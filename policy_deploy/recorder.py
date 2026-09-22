"""把每一局真机 rollout 录下来。

为什么要录：**行为对不对肉眼能看出来，模型到底"看到"了什么看不出来。**
插不进去的时候，是模型判断错了，还是我们喂给它的触觉/画面/位姿本身就是坏的？
不录就只能猜。录了才能事后重放，甚至在自己机器上把那一局重新跑一遍。

分两级存，避免磁盘爆掉：

  * **小流**（每局都存，几百 KB）：状态、动作、触觉格子、时间戳、每拍耗时、有没有漏拍、触觉基准。
    位姿轨迹、动作曲线、触觉信号面板都能从这里画出来。
  * **原图**（默认只有失败的局才落盘）：三路相机的原始画面。跑的时候先留在内存里，
    一局结束时按成败决定写不写。**代价不小**：内存里最多留 90 秒 = 1438 帧，
    两路 512×910 加一路 240×320 就是 **4.35 GB 常驻**，写盘时 np.stack 再复制一份、
    峰值约 8.7 GB，落盘也是 4.35 GB 一局。不想要就 `PolicyRunner(..., record_images=False)`。
    注意 `finish_episode()` 不传 success（默认 None）时**每局都会落图**——那是有意的
    （结果未知要留证据），但别忘了传。

用法（`PolicyRunner` 已经接好，一般不用直接碰这个）：
    rec = Recorder("runs/eval_0806", keep_images_on_success=False)
    rec.start("ep_003")
    ...每拍 rec.log(...)
    rec.finish(success=False)        # 失败 → 连原图一起写
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

MAX_BUFFER_SEC = 90        # 内存里最多留这么久的原图，防止一局跑飞把内存吃光


class Recorder:
    def __init__(self, out_dir: str, keep_images_on_success: bool = False,
                 fps: float = 15.98):
        self.root = Path(out_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.keep_on_success = keep_images_on_success
        self.fps = fps
        self._maxlen = int(MAX_BUFFER_SEC * fps)
        self.ep = None
        self._meta: dict = {}          # start() 之前就可以先记（比如用的哪个存档）
        self._writers: list[threading.Thread] = []

    # ------------------------------------------------------------------
    def start(self, name: str) -> None:
        self.ep = name
        self._small = {k: [] for k in
                       ("t", "t_wall", "state", "action", "tactile_grid", "eef_pose",
                        "eef_pose_cmd", "gripper_width", "step_ms", "recomputed")}
        self._imgs = deque(maxlen=self._maxlen)
        self._t0_wall = time.time()

    def log(self, *, t, state, action, tactile_grid, eef_pose, gripper_width,
            eef_pose_cmd=None, t_wall=None, third_bgr=None, wrist_bgr=None,
            tactile_bgr=None, step_ms=0.0, recomputed=False) -> None:
        if self.ep is None:
            return
        s = self._small
        # 两个钟都要：
        #   t      单调钟（perf_counter），起点随进程走，只有差值有意义 —— 算循环周期用它，
        #          它不会被系统对时/夏令时往回拨。
        #   t_wall 绝对时间（epoch 秒）—— 跟相机时间戳、机械臂日志、录像文件对齐用它。
        #          只有它能回答"这一拍到底是几点几分几秒发生的"。
        s["t"].append(np.float64(t))
        s["t_wall"].append(np.float64(time.time() if t_wall is None else t_wall))
        s["state"].append(np.asarray(state, np.float32))
        s["action"].append(np.asarray(action, np.float32))
        s["tactile_grid"].append(np.asarray(tactile_grid, np.float32))
        s["eef_pose"].append(np.asarray(eef_pose, np.float32))
        # 控制器**下发**的目标位姿（franka 的 O_T_EE_c，转成 pos+quat xyzw）。
        # 为什么必须记它：实测位姿在下降过程中俯仰会自己掉 5~13 度，而策略全程只命令
        # 约 1 度旋转。光有实测位姿，分不出「目标本身在往下漂」还是「目标是平的、手臂
        # 没跟上」—— 这两种的修法完全相反。没传就填 nan，事后一眼能看出这一局没记。
        s["eef_pose_cmd"].append(
            np.full(7, np.nan, np.float32) if eef_pose_cmd is None
            else np.asarray(eef_pose_cmd, np.float32))
        s["gripper_width"].append(np.float32(gripper_width))
        s["step_ms"].append(np.float32(step_ms))
        s["recomputed"].append(bool(recomputed))
        if third_bgr is not None:
            # 必须拷贝：真机那侧多半会复用同一块相机缓冲，不拷的话存下来全是最后一帧
            self._imgs.append((third_bgr.copy(), wrist_bgr.copy(), tactile_bgr.copy()))

    @property
    def n_frames(self) -> int:
        return len(self._small["action"]) if self.ep is not None else 0

    def note(self, **kw) -> None:
        """记一些一局一个的东西，比如触觉基准、用的哪个存档。"""
        self._meta.update(kw)

    # ------------------------------------------------------------------
    def finish(self, success: bool | None = None, **extra) -> Path:
        """一局结束。写盘在后台线程做，不挡下一局开始。"""
        if self.ep is None:
            raise RuntimeError("还没 start()")
        d = self.root / self.ep
        d.mkdir(parents=True, exist_ok=True)
        small = {k: np.asarray(v) for k, v in self._small.items()}
        # 每拍之间隔了多久（毫秒）。第一拍没有上一拍，补 nan。
        t = small["t"].astype(np.float64)
        small["dt_ms"] = (np.concatenate([[np.nan], np.diff(t) * 1000.0])
                          if len(t) else np.zeros(0))
        dt = small["dt_ms"][1:]
        t_end = time.time()
        meta = dict(self._meta, success=success, n_frames=len(small["action"]),
                    fps=self.fps,
                    t_start_wall=self._t0_wall, t_end_wall=t_end,
                    started_at=datetime.fromtimestamp(self._t0_wall).isoformat(timespec="seconds"),
                    duration_s=round(t_end - self._t0_wall, 3),
                    # 实际跑出来的节奏。hz_actual 明显低于 fps，或 dt_max_ms 远大于 1000/fps，
                    # 就是循环被什么东西拖住了 —— 手臂速度会跟着变，别只看行为好不好看。
                    hz_actual=(round(1000.0 / float(np.median(dt)), 2) if len(dt) else None),
                    dt_max_ms=(round(float(np.max(dt)), 1) if len(dt) else None),
                    **extra)
        np.savez_compressed(d / "trace.npz", **small)

        want_imgs = (success is False) or (success is None) or self.keep_on_success
        imgs = list(self._imgs) if want_imgs else []
        # ★图像和数字流不等长时，必须把偏移写下来。
        # 数字流是无上限的 list，图像是 maxlen=90秒 的环形缓冲；一局超过 90 秒（实测示教
        # 最长的一集就有 97.9 秒），图像只剩最后 1438 帧，而 trace.npz 是全长。
        # 这时按下标 i 去对 trace 的第 i 拍和 third.npy 的第 i 帧，对到的差 img_start 帧
        # （实测差 162 帧 = 10 秒），而且不会有任何报错。
        # 正确的对法：third.npy[j] 对应 trace 的第 img_start + j 拍。
        meta["n_images"] = len(imgs)
        meta["img_start"] = len(small["action"]) - len(imgs) if imgs else None
        (d / "meta.json").write_text(json.dumps(meta, indent=1, default=str),
                                     encoding="utf-8")
        self._imgs = deque(maxlen=self._maxlen)
        self.ep = None
        if imgs:
            th = threading.Thread(target=self._write_images, args=(d, imgs), daemon=True)
            th.start()
            self._writers.append(th)
        return d

    @staticmethod
    def _write_images(d: Path, imgs: list) -> None:
        third = np.stack([a for a, _, _ in imgs])
        wrist = np.stack([b for _, b, _ in imgs])
        tac = np.stack([c for _, _, c in imgs])
        # 不压缩：一局 1.4 G，但写得快、读回来重放也快。要省地方就事后转视频。
        np.save(d / "third.npy", third)
        np.save(d / "wrist.npy", wrist)
        np.save(d / "tactile.npy", tac)

    def wait(self, timeout: float = 300.0) -> None:
        """收工前调，确认后台写盘都落完了。"""
        for th in self._writers:
            th.join(timeout=timeout)
        self._writers = []
