"""给它原始数据，还你 7 维动作。真机那侧只要收发，别的都在这个文件里。

    from policy_runner import PolicyRunner
    runner = PolicyRunner("<存档目录>")

    runner.reset()                       # 每次试验开始
    while not done:                      # 控制循环默认按 11 Hz 跑
        action = runner.step(third_bgr, wrist_bgr, tactile_bgr, eef_pose, gripper_width)
        发给机械臂(action)                # 7 个数都在【机器人基座系】，见 README

这个文件包含四块：
    ① 触觉原图 → 10×12×3 格子（含每局自动标定零点）      TactileEncoder
    ② 相机图裁剪/转色、位姿+夹爪 → 11 维状态             ObservationBuilder
    ③ 触觉历史、提前算、动作块管理                        PolicyRunner
    ④ 跑模型                                              PolicyRunner._sample_chunk
①② 是把训练时那套离线转换（convert_realrobot_to_lerobot.py）改成一帧一帧的在线版，
公式一字未改，已在真实录像上逐帧核对过（差 0.000e+00）。

---------------------------------------------------------------------------
为什么要提前算（默认开着）
---------------------------------------------------------------------------
模型一次算未来 50 步，只执行前 10 步就重算。算一次约 120 毫秒，而默认 11 Hz 下每拍约
90.9 毫秒——等算完再发仍会造成额外停顿，手臂走走停停。

所以队列还剩两步时就在后台线程开算，边发边算。算完时已经过去几拍，就把那几步跳过、
从对得上当前时刻的那一步接上，动作和时间仍然对齐。实测 0/195 漏拍。

---------------------------------------------------------------------------
为什么触觉历史要自己管
---------------------------------------------------------------------------
模型要看往回九个时刻的触觉（最远 48 帧）。LeRobot 内部有个队列在攒，但那份是边跑边改的
——后台线程正在读，主线程还在往里塞，一次推理里会前半段用旧历史、后半段用新历史，
结果错了还不报错。所以这里自己存一份，开算时快照给后台，两边不共享会变的东西。
"""
from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

DATASET_HZ = 15.98      # 训练数据采样率；仅用于说明，部署不重采样
CONTROL_HZ = 11.0       # 真机部署默认节拍；给上一条动作更多执行时间
# 队列还剩这么多步时开算。11 Hz 下 2 拍约 182 毫秒，足够覆盖约 120 毫秒推理。
# 不设更大：提前得越多，动作用的画面越旧，反应越迟钝，对毫米级对准是实打实的代价。
# 换到更慢的显卡上先跑 selftest.py，漏拍了再往上调。
TRIGGER_LEFT = 2
D94_STATE_HISTORY_OFFSETS = (-48, -43, -37, -32, -27, -21, -16, -11, -5, 0)
D94_STATE_DIM = 11
D94_STATE_HISTORY_HIDDEN = 64

THIRD_SIZE = (512, 910)       # 第三视角原图尺寸 (高, 宽)
WRIST_SIZE = (512, 910)
TACTILE_SIZE = (240, 320)
GRID_H, GRID_W = 10, 12
N_BASE = 50            # 定基准用多少帧（训练默认值）
AUTO_FRAMES = 110      # 每局开头攒这么多帧来标定触觉零点（约 7 秒）
PROVISIONAL_N = 10     # 攒够这么多帧先算个临时零点顶着（只算一次）


# ==========================================================================
# ① 触觉：原图 → 10×12×3 格子，含每局自动标定
# ==========================================================================
class TactileEncoder:
    """触觉传感器是"摄像头从里面拍一块软胶垫"。什么都没碰时拍出来也不是一片黑，
    而是胶垫本身（灯不均匀、有花纹、有旧痕）。所以要先减掉"什么都没压时长什么样"
    ——就是电子秤的去皮；再除以摄像头自己抖动的幅度，把读数单位统一成"比自身抖动大几倍"。

    皮会变（灯随温度飘、曝光自动调、胶垫留旧痕、传感器拆装），所以每局重取。训练时
    就是每一集重取一次的。这里每局开头自动攒 110 帧算皮，算完冻住，不需要任何人配合。
    唯一前提：开头这七秒夹爪没碰到东西（采集时从开始到抓取有 22~38 秒，很宽裕）。
    """

    def __init__(self, n_base: int = N_BASE, auto_frames: int = AUTO_FRAMES):
        self.n_base, self.auto_frames = n_base, auto_frames
        self.reset()

    def reset(self) -> None:
        # generation：上一局的标定线程可能还在跑（约 240 毫秒）。它写回时会核对代号，
        # 对不上就丢弃。不这么做的话，开局七秒左右中止重开时，新一局会静默沿用上一局的
        # 零点、并且因为 frozen 已被置 True 而**再也不会重新标定**，整局用错皮还没有任何提示。
        self._gen = getattr(self, "_gen", 0) + 1
        self._buf: list[np.ndarray] = []
        self.base = None
        self.sigma = None
        self.frozen = False
        self._prov = None            # 攒够之前顶着用的临时零点，只算一次
        self._calib_thread = None

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        """(240,320,3) BGR -> (1,10,12,3) = [x梯度, y梯度, 形变响应强度]"""
        f = np.asarray(frame_bgr, dtype=np.float32)
        if not self.frozen:
            # 必须真拷贝：np.asarray(uint8数组, dtype=uint8) 返回的是**同一个对象**。
            # 相机那侧多半复用同一块缓冲，不拷的话 110 帧全指向最后一帧 ——
            # base 变成那一帧、sigma 各通道 std=0 被钳到 1e-3，格子幅值放大约 2900 倍，
            # 而且一声不吭。recorder.py 里防了这个坑，这里原来没防。
            self._buf.append(np.array(frame_bgr, dtype=np.uint8, copy=True))
            if len(self._buf) >= self.auto_frames and self._calib_thread is None:
                # 定零点要在 110 帧上求中位数，约 240 毫秒，当场算会把这一拍拖爆。
                # 丢后台算，算完再冻住，这期间继续用临时零点。
                buf = np.stack(self._buf).astype(np.float32)
                self._calib_thread = threading.Thread(
                    target=self._calibrate_from, args=(buf, self._gen), daemon=True)
                self._calib_thread.start()
            if not self.frozen:
                # 临时零点**只算一次**。每帧都拿全部已攒帧重算的话，攒得越多越慢
                # （实测会从 100 毫秒一路涨到 200 毫秒），把控制循环拖垮。
                if self._prov is None:
                    if len(self._buf) < PROVISIONAL_N:
                        return np.zeros((1, GRID_H, GRID_W, 3), dtype=np.float32)
                    cur = np.stack(self._buf).astype(np.float32)
                    self._prov = (
                        np.median(cur, axis=0).astype(np.float32),
                        np.maximum(cur.std(axis=0).mean(axis=(0, 1)), 1e-3).astype(np.float32),
                    )
                return self._grid(f, *self._prov)
        return self._grid(f, self.base, self.sigma)

    def _calibrate_from(self, tac: np.ndarray, gen: int) -> None:
        n = len(tac)
        # 后半段定零点，前半段定噪声尺度，两段不重叠 —— 和训练的取法一致
        cut = max(5, n - self.n_base)
        base = np.median(tac[cut:], axis=0).astype(np.float32)
        noise = tac[max(0, cut - self.n_base):cut]
        if len(noise) < 5:
            noise = tac[cut:]
        sigma = np.maximum(noise.std(axis=0).mean(axis=(0, 1)), 1e-3).astype(np.float32)
        if gen != self._gen:
            return                     # 这一局已经被 reset 掉了，结果作废
        # 一次性赋值再置 frozen，读的那一侧不会看到只改了一半的状态
        self.base, self.sigma = base, sigma
        self.frozen = True
        self._buf = []

    @staticmethod
    def _grid(f: np.ndarray, base: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        d = (f - base) / sigma
        mag = np.linalg.norm(d, axis=2) / np.sqrt(3.0)                  # (240,320)
        small = cv2.resize(mag, (GRID_W, GRID_H), interpolation=cv2.INTER_AREA)
        gy, gx = np.gradient(small.astype(np.float32))
        return np.stack([gx, gy, small], axis=-1)[None, ...].astype(np.float32)


# ==========================================================================
# ② 观测：裁图、转色、拼状态（几何函数抄自训练时的转换脚本，一字未改）
# ==========================================================================
def quat_xyzw_to_R(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rot6d_from_R(R: np.ndarray) -> np.ndarray:
    """六维旋转 = 旋转矩阵的前两**列**。列不是行，抄错了模型会拿到转置过的姿态。"""
    return R[:, :2].T.reshape(-1)


def R_from_rot6d(v) -> np.ndarray:
    """六维旋转 -> 旋转矩阵。rot6d 是矩阵的前两**列**（与 rot6d_from_R 互逆）。

    模型吐出来的两列不保证正交、也不保证单位长，所以必须先施密特正交化再叉乘补第三列。
    直接拿去当旋转矩阵用会得到一个带缩放/剪切的矩阵，转成四元数后姿态是错的。
    """
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    c0, c1 = v[0:3], v[3:6]
    n0 = np.linalg.norm(c0)
    if n0 < 1e-9:
        raise ValueError(f"rot6d 第一列退化：{v.tolist()}")
    c0 = c0 / n0
    c1 = c1 - np.dot(c1, c0) * c0                       # 去掉与 c0 平行的分量
    n1 = np.linalg.norm(c1)
    if n1 < 1e-9:
        raise ValueError(f"rot6d 两列共线：{v.tolist()}")
    c1 = c1 / n1
    return np.column_stack([c0, c1, np.cross(c0, c1)])


def quat_xyzw_from_R(R: np.ndarray) -> np.ndarray:
    """旋转矩阵 -> 四元数 xyzw（Franka/ROS 的次序）。

    按迹的四个分支切，不用单一公式：迹接近 -1 时那个公式的分母趋零，误差会炸。
    """
    R = np.asarray(R, dtype=np.float64)
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    return q / np.linalg.norm(q)


def _to_chw(rgb_uint8: np.ndarray) -> torch.Tensor:
    """(H,W,3) uint8 -> (3,H,W) float32 [0,1]，跟数据集读出来的一模一样。"""
    t = torch.from_numpy(np.ascontiguousarray(rgb_uint8)).permute(2, 0, 1)
    return t.to(torch.float32) / 255.0


class ObservationBuilder:
    def __init__(self, task: str, device: str = "cpu"):
        self.task, self.device = task, device
        self.tactile = TactileEncoder()

    def reset(self) -> None:
        self.tactile.reset()

    def light(self, tactile_bgr, eef_pose, gripper_width):
        """每帧都做的那部分：触觉转格子 + 拼状态。图像不碰，所以很便宜。

        分成 light / full 两半，是因为图像只有真要跑模型的那一帧才用得上（每 10 帧一次），
        其余九帧转了也白转 —— 两路 512×910 的图转一次要 4 毫秒，占每拍预算一大截。
        """
        if tactile_bgr.shape[:2] != TACTILE_SIZE:
            raise ValueError(f"触觉图应为 {TACTILE_SIZE}，收到 {tactile_bgr.shape[:2]}")
        if tactile_bgr.dtype != np.uint8:
            raise ValueError(f"触觉图必须是 uint8，收到 {tactile_bgr.dtype}")
        # 位姿只要有一个 NaN/inf，整块 50 步动作全会变成 NaN，而且不报错、照样发给机械臂。
        # 位姿话题掉一帧脏数据、或四元数全零（下面要除以范数）都会触发。这里当场拦掉。
        ep = np.asarray(eef_pose, dtype=np.float64)
        if ep.shape != (7,) or not np.isfinite(ep).all():
            raise ValueError(f"eef_pose 必须是 7 个有限数，收到 {eef_pose}")
        if np.linalg.norm(ep[3:7]) < 1e-6:
            raise ValueError(f"四元数范数接近零：{ep[3:7]}")
        if not np.isfinite(gripper_width):
            raise ValueError(f"gripper_width 不是有限数：{gripper_width}")
        R = quat_xyzw_to_R(np.asarray(eef_pose)[3:7])
        half = float(gripper_width) / 2.0        # 两根手指各自的位置 = 开口的一半
        state = np.concatenate([
            np.asarray(eef_pose, dtype=np.float32)[:3],
            rot6d_from_R(R),
            [half, half],
        ]).astype(np.float32)
        return self.tactile(tactile_bgr), state

    def full(self, third_bgr, wrist_bgr, grid, state) -> dict:
        """要跑模型的那一帧才调：把图像也转好，拼成完整观测。

        尺寸不对直接报错，不悄悄缩放 —— 缩放会让画面和训练时对不上，而模型不会告诉你。
        """
        if third_bgr.shape[:2] != THIRD_SIZE:
            raise ValueError(f"第三视角应为 {THIRD_SIZE}，收到 {third_bgr.shape[:2]}")
        if wrist_bgr.shape[:2] != WRIST_SIZE:
            raise ValueError(f"腕部相机应为 {WRIST_SIZE}，收到 {wrist_bgr.shape[:2]}")
        h, w = third_bgr.shape[:2]
        x0 = (w - h) // 2                        # 512x910 取中间正方形，列 199:711
        third_rgb = np.ascontiguousarray(third_bgr[:, x0:x0 + h, ::-1])
        # 腕部不裁：夹爪在每一帧都越过左右边界，裁了就把爪子切掉了
        wrist_rgb = np.ascontiguousarray(wrist_bgr[:, :, ::-1])
        d = self.device
        return {
            "task": [self.task],
            "observation.state": torch.from_numpy(state)[None].to(d),
            "observation.images.camera1": _to_chw(third_rgb)[None].to(d),
            "observation.images.camera2": _to_chw(wrist_rgb)[None].to(d),
            "observation.tactile.force_grid": torch.from_numpy(grid)[None].to(d),
        }

    @property
    def tactile_ready(self) -> bool:
        return self.tactile.frozen


# ==========================================================================
# ③④ 总入口
# ==========================================================================
class PolicyRunner:
    def __init__(self, ckpt_dir: str, device: str | None = None,
                 task: str = "insert the plug into the power strip",
                 pipelined: bool = True, trigger_left: int = TRIGGER_LEFT,
                 record_dir: str | None = None, record_images: bool = True,
                 rtc: bool = True):
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.factory import make_pre_post_processors

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = Path(ckpt_dir)
        self.policy = SmolVLAPolicy.from_pretrained(ckpt).to(self.device).eval()
        cfg = self.policy.config
        # RTC（real-time chunking，默认开）：换动作块时，把上一块**还没执行完的尾巴**
        # 作为软约束喂给去噪过程，让新块的开头跟旧块正在执行的部分接得上，压掉块交界的
        # 跳变。纯推理期机制，任何存档都能用，不改训练。关掉：PolicyRunner(..., rtc=False)。
        self.rtc = bool(rtc)
        if self.rtc:
            from lerobot.policies.rtc.configuration_rtc import RTCConfig
            cfg.rtc_config = RTCConfig(enabled=True,
                                       execution_horizon=int(cfg.n_action_steps))
            self.policy.init_rtc_processor()
        # 归一化的均值方差沿用存档里那一份（训练集算出来的），只换设备。
        # 换成别的统计量等于给模型换了把尺子，输入输出全偏，而且不报错。
        self.pre, self.post = make_pre_post_processors(
            cfg, pretrained_path=str(ckpt),
            preprocessor_overrides={"device_processor": {"device": self.device}},
        )
        # 观测先在 CPU 上拼，等真要算了才搬上显卡 —— 否则主线程每拍的小搬运会排在
        # 后台那 120 毫秒的大计算后面，提前算就白做了。
        self.obs = ObservationBuilder(task=task, device="cpu")

        # 动作空间由存档自己说了算，不用调用方指定：
        #   7 维  = 增量  [dx dy dz | 轴角 drx dry drz | 夹爪]
        #   10 维 = 绝对  [x y z | rot6d(6) | 夹爪]
        # 两种存档的调用方式完全一样，区别只在 step() 返回什么、以及怎么下发（见 README）。
        self.action_dim = int(cfg.action_feature.shape[0])
        self.absolute = (self.action_dim == 10)
        self.uses_tactile = bool(getattr(cfg, "use_tactile", False))
        self.n_action_steps = int(cfg.n_action_steps)
        self.chunk_size = int(cfg.chunk_size)
        self.pipelined = pipelined
        self.trigger_left = int(trigger_left)
        # 录制：默认关。开了之后每局都存状态/动作/触觉，原图只在失败的局落盘。
        self.rec = None
        if record_dir:
            from recorder import Recorder
            self.rec = Recorder(record_dir)
            self.rec.note(ckpt=str(ckpt), uses_tactile=self.uses_tactile)
        self.record_images = record_images
        self._ep_id = 0
        self._period = 1.0 / CONTROL_HZ             # wait_for_next_tick 的节拍长度
        self.hist_offsets = [int(o) for o in getattr(cfg, "tactile_ibr_history_offsets", [])]
        self.needs_hist = self.uses_tactile and len(self.hist_offsets) > 0
        self._hist_len = (abs(min(self.hist_offsets)) + 1) if self.needs_hist else 1
        state_taps = (getattr(cfg, "observation_delta_indices_by_key", None) or {}).get(
            "observation.state", []
        )
        self.state_hist_offsets = [int(o) for o in state_taps]
        self.needs_state_hist = bool(
            getattr(cfg, "tactile_state_history_enabled", False)
        )
        if self.needs_state_hist:
            expected_offsets = list(D94_STATE_HISTORY_OFFSETS)
            if self.state_hist_offsets != expected_offsets:
                raise RuntimeError(
                    "this deploy package only accepts the audited D94 state taps: "
                    f"expected={expected_offsets}, got={self.state_hist_offsets}"
                )
            if not self.uses_tactile or not self.needs_hist:
                raise RuntimeError("D94 state history requires the tactile history path")
            if self.state_hist_offsets != self.hist_offsets + [0]:
                raise RuntimeError(
                    "state and tactile history taps must match exactly: "
                    f"state={self.state_hist_offsets}, tactile={self.hist_offsets + [0]}"
                )
            state_shape = tuple(cfg.input_features["observation.state"].shape)
            if state_shape != (D94_STATE_DIM,):
                raise RuntimeError(
                    f"D94 state shape must be ({D94_STATE_DIM},), got {state_shape}"
                )
            if int(getattr(cfg, "tactile_state_history_hidden", -1)) != D94_STATE_HISTORY_HIDDEN:
                raise RuntimeError(
                    "D94 state-history hidden size must be "
                    f"{D94_STATE_HISTORY_HIDDEN}"
                )
            if bool(getattr(cfg, "tactile_state_history_repeat_current", True)):
                raise RuntimeError("D94 requires real state history, not repeat_current")
            if getattr(self.policy.model, "state_history_encoder", None) is None:
                raise RuntimeError("checkpoint enabled state history but built no encoder")
        self._state_hist_len = (
            1 - self.state_hist_offsets[0] if self.needs_state_hist else 1
        )
        self._pending = None
        self._warmup()
        self.reset()
        import atexit, weakref
        # 用弱引用注册：atexit 存 bound method 会一直强引用 self，实例永远不释放。
        # selftest.py 连开四个 PolicyRunner 做对比，那样会四份模型全常驻、小显存卡必 OOM。
        _wc = weakref.WeakMethod(self.close)
        atexit.register(lambda: (_wc() or (lambda: None))())

    def _warmup(self) -> None:
        """先空跑一次。第一次推理要建 CUDA 上下文、编译核函数，比后面慢两三倍；
        不预热的话试验第一拍会晚三百多毫秒。用全零假数据跑，不碰任何真实状态。"""
        n = len(self.hist_offsets) + 1 if self.needs_hist else 1
        raw = {
            "task": [self.obs.task],
            "observation.state": torch.zeros(1, 11),
            "observation.images.camera1": torch.zeros(1, 3, 512, 512),
            "observation.images.camera2": torch.zeros(1, 3, 512, 910),
            "observation.tactile.force_grid": torch.zeros(1, 1, 10, 12, 3),
        }
        tactile_hist = tactile_pad = state_hist = None
        if self.needs_hist:
            tactile_hist = torch.zeros(1, n, 1, 10, 12, 3)
            tactile_pad = torch.zeros(1, n, dtype=torch.bool)
        if self.needs_state_hist:
            state_hist = torch.zeros(1, len(self.state_hist_offsets), 11)
        try:
            self._infer(raw, (tactile_hist, tactile_pad, state_hist))
        except Exception as e:
            if self.needs_state_hist:
                raise RuntimeError(
                    "D94 state-history warmup failed; refusing to start real control"
                ) from e
            # Legacy checkpoints historically allowed a failed dummy warmup and paid
            # the one-time latency on the first real chunk instead.
            print(f"[提示] 预热没跑成（{e}），第一拍可能会慢一点")

    # ------------------------------------------------------------------ 生命周期
    def reset(self) -> None:
        """每次试验开始必须调。触觉零点、触觉历史、动作队列一起清掉。"""
        # 上一局可能还有后台线程在算，先收干净再清状态，否则它会往已经作废的盒子里写。
        pending = getattr(self, "_pending", None)
        if pending is not None:
            pending[0].join(timeout=5.0)
            if pending[0].is_alive():
                raise RuntimeError(
                    "previous episode inference is still running after 5 seconds; "
                    "refusing to reset shared model state"
                )
        self.obs.reset()
        self._hist = deque(maxlen=self._hist_len)   # 自己管的触觉历史
        self._state_hist = deque(maxlen=self._state_hist_len)
        self._n_seen = 0
        self._acts = deque()                        # 待执行的动作
        self._tick = 0
        self._pending = None                        # (线程, 结果盒子, 开算那一拍的序号)
        # 下面三个都必须在这里清掉。少一个就是「平时跑得好好的，出事那一刻才崩」：
        #   _t_last / _fast_run 只在 _check_rate 里用，而 _check_rate 只有循环跑太快时才走到
        #     那个分支 —— 也就是手臂正在超速的时候崩，正是最不能崩的时候。
        #   _t_next 是 wait_for_next_tick 的节拍锚点，不清的话第一拍就 AttributeError。
        self._t_last = None
        self._fast_run = 0
        self._t_next = None
        self._last_chunk = None                     # RTC：上一块的归一化输出（GPU 上）
        self._last_t0 = 0                           # 上一块对应的起始拍号
        self.policy.reset()
        if self.rec is not None:
            # 三种情况必须分清，否则会静默覆盖已录好的一局：
            #   ① 当前有一局但一帧没录（构造函数开的空壳）→ 原地重用，不换号
            #   ② 当前有一局且录了东西（上一局忘了 finish）→ 按"结果未知"存下来，换号
            #   ③ 当前没有局（上一局已 finish）           → 直接换号
            if self.rec.ep is not None and self.rec.n_frames == 0:
                pass
            else:
                if self.rec.ep is not None:
                    self.rec.finish(success=None)
                self._ep_id += 1
            self.rec.start(f"ep_{self._ep_id:03d}")

    # ------------------------------------------------------------------ 主循环
    def step(self, third_bgr, wrist_bgr, tactile_bgr, eef_pose, gripper_width,
             eef_pose_cmd=None) -> np.ndarray:
        """一帧进，一个动作出。维数由存档决定，见 self.absolute：

        7 维（增量存档）  [dx dy dz, drx dry drz, 夹爪]  基座系增量，左乘
        10 维（绝对存档） [x y z, rot6d(6), 夹爪]        基座系**绝对目标位姿**

        绝对存档建议直接用 step_absolute()，它把 rot6d 换成四元数、并做好安全检查。
        """
        t_enter = time.perf_counter()
        t_wall = time.time()               # 绝对时间，用来跟相机/机械臂那侧的日志对齐
        # 相机图的 dtype 在这里查，不能放进 full()：full() 只有真要跑模型那一帧才调用，
        # 其余九拍走不到，float32 的图会被静默吃掉（再除一次 255 → 近全黑，不报错）。
        if third_bgr.dtype != np.uint8 or wrist_bgr.dtype != np.uint8:
            raise ValueError(f"相机图必须是 uint8，收到 {third_bgr.dtype}/{wrist_bgr.dtype}")
        grid, state = self.obs.light(tactile_bgr, eef_pose, gripper_width)
        if self.needs_hist:
            self._hist.append(grid)
        if self.needs_state_hist:
            # ObservationBuilder reuses no state storage today, but copy here makes the
            # asynchronous snapshot contract explicit and safe against future changes.
            self._state_hist.append(np.asarray(state).copy())
        self._n_seen += 1

        need_now = len(self._acts) == 0
        launch = self.pipelined and self._pending is None and \
            len(self._acts) <= self.trigger_left
        if need_now or launch:
            raw = self.obs.full(third_bgr, wrist_bgr, grid, state)
            if need_now and self._pending is None:
                # 队列空了还没人在算（同步模式，或者第一拍）：只能当场算
                self._absorb(self._infer(raw, self._stack_history(),
                                         self._rtc_kwargs(), self._tick), self._tick)
            elif need_now:
                self._absorb(*self._join())            # 等后台那份，正常不会走到这里
                if launch and self._pending is None:
                    self._launch(raw)
            else:
                self._launch(raw)

        if not self._acts:                             # 兜底，理论上到不了
            self._absorb(*self._join())

        act = self._acts.popleft()
        if not np.isfinite(act).all():
            raise RuntimeError(f"模型输出含 NaN/inf：{act}。绝不能发给机械臂。")
        self._tick += 1
        self._check_rate()
        if self.rec is not None:
            im = (third_bgr, wrist_bgr, tactile_bgr) if self.record_images else (None, None, None)
            self.rec.log(t=t_enter, t_wall=t_wall, state=state, action=act, tactile_grid=grid,
                         eef_pose=eef_pose, eef_pose_cmd=eef_pose_cmd,
                         gripper_width=gripper_width,
                         third_bgr=im[0], wrist_bgr=im[1], tactile_bgr=im[2],
                         step_ms=(time.perf_counter() - t_enter) * 1000,
                         recomputed=(need_now or launch))
        return act

    # ---- 绝对位姿存档专用 --------------------------------------------------
    # 只限「这一拍相对上一拍能跳多远」，不限「能去哪儿」。
    # 增量存档天生有前者的保护（每拍最多几毫米），绝对存档没有 —— 一个跑飞的目标
    # 会让手臂以高速大幅扫掠。所以按训练里单步的实际量级（位移最大 4.24 毫米、
    # 转角最大 0.39 度）留两倍以上余量，只挡住数量级异常的那一拍。
    #
    # **刻意不设工作空间范围限制**：那会把「模型走到训练范围之外」当成故障拦下来，
    # 而那正是泛化测试要观察的行为。要不要限位置范围是实验设计的事，不该写死在这里。
    MAX_STEP_MM = 10.0
    MAX_STEP_DEG = 2.0

    def step_absolute(self, third_bgr, wrist_bgr, tactile_bgr, eef_pose, gripper_width,
                      eef_pose_cmd=None):
        """绝对位姿存档用这个。返回 (p_target(3,米), quat_xyzw(4), gripper, ok)。

        ok=False 表示这一拍的目标没通过安全检查，**不要下发**，保持上一次目标即可。
        典型原因：模型进了没见过的状态、或者实测位姿流断了。
        """
        if not self.absolute:
            raise RuntimeError("这是增量存档（7 维），请用 step()")
        a = self.step(third_bgr, wrist_bgr, tactile_bgr, eef_pose, gripper_width,
                      eef_pose_cmd=eef_pose_cmd)
        p = np.asarray(a[:3], dtype=np.float64)
        R = R_from_rot6d(a[3:9])
        q = quat_xyzw_from_R(R)
        grip = float(a[9])

        cur = np.asarray(eef_pose, dtype=np.float64)
        d_mm = float(np.linalg.norm(p - cur[:3]) * 1000.0)
        R_cur = quat_xyzw_to_R(cur[3:7])
        c = float(np.clip((np.trace(R @ R_cur.T) - 1.0) / 2.0, -1.0, 1.0))
        d_deg = float(np.degrees(np.arccos(c)))
        ok = (d_mm <= self.MAX_STEP_MM) and (d_deg <= self.MAX_STEP_DEG)
        if not ok:
            why = []
            if d_mm > self.MAX_STEP_MM: why.append(f"单拍位移 {d_mm:.1f}mm>{self.MAX_STEP_MM}")
            if d_deg > self.MAX_STEP_DEG: why.append(f"单拍转角 {d_deg:.2f}°>{self.MAX_STEP_DEG}")
            print(f"[安全] 第 {self._tick} 拍目标被拒：{'；'.join(why)}")
        return p, q, grip, ok

    def close(self) -> None:
        """收工。等后台那次推理算完再走。

        不收的话：提前算用的是后台线程，进程退出时它可能正卡在一次前向里，被硬砍掉会让
        底层库抛 terminate、进程带着 core dump 退出 —— 东西其实跑完了，但看着像崩了。
        已经注册进 atexit，正常不用自己调。
        """
        pending = getattr(self, "_pending", None)
        if pending is not None:
            pending[0].join(timeout=10.0)
            self._pending = None
        if getattr(self, "rec", None) is not None:
            self.rec.wait(timeout=60.0)

    def finish_episode(self, success: bool | None = None, **extra):
        """一局结束时调，把这一局落盘。success=False 的局会连原图一起存。"""
        if self.rec is None:
            return None
        self.rec.note(tactile_base_frozen=bool(self.obs.tactile_ready))
        return self.rec.finish(success=success, **extra)

    # ------------------------------------------------------------------ 内部
    def _stack_history(self):
        """冻结触觉和state历史；开局不足的时刻都重复本局第一帧。"""
        g = m = s = None
        if self.needs_hist:
            cur = self._hist[-1]
            frames, pad = [], []
            for o in self.hist_offsets:
                idx = len(self._hist) - 1 + o
                enough = (self._n_seen - 1 + o) >= 0 and idx >= 0
                frames.append(self._hist[idx] if enough else self._hist[0])
                pad.append(not enough)
            frames.append(cur)                             # 最后一帧是当前
            pad.append(False)                              # 当前帧永远是有的
            g = torch.from_numpy(np.stack(frames))[None]   # (1,10,1,10,12,3)
            m = torch.tensor(pad, dtype=torch.bool)[None]  # (1,10)，模型只取前九个
        if self.needs_state_hist:
            frames = []
            for o in self.state_hist_offsets:
                idx = len(self._state_hist) - 1 + o
                enough = (self._n_seen - 1 + o) >= 0 and idx >= 0
                frames.append(
                    self._state_hist[idx] if enough else self._state_hist[0]
                )
            s = torch.from_numpy(np.stack(frames))[None]   # (1,10,11)
        return g, m, s

    def _rtc_kwargs(self) -> dict:
        """RTC 的两个输入，在**主线程**上算好再交给推理线程（避免竞态）：
        prev_chunk_left_over = 上一块从「当前拍」起还没执行的尾巴（归一化空间）；
        inference_delay      = 推理期间还会被执行掉的步数 = 此刻队列里剩几步。"""
        if not self.rtc or self._last_chunk is None:
            return {}
        off = self._tick - self._last_t0
        if off < 0 or off >= self._last_chunk.shape[1]:
            return {}
        return {"prev_chunk_left_over": self._last_chunk[:, off:, :],
                "inference_delay": int(len(self._acts))}

    def _launch(self, raw: dict) -> None:
        hist = self._stack_history()
        box: dict = {}
        t0 = self._tick
        rtc_kw = self._rtc_kwargs()
        th = threading.Thread(target=lambda: box.update(
            a=self._infer(raw, hist, rtc_kw, t0)), daemon=True)
        th.start()
        self._pending = (th, box, t0)

    def _join(self):
        th, box, t0 = self._pending
        th.join()
        self._pending = None
        if "a" not in box:
            raise RuntimeError("后台推理线程挂了")
        return box["a"], t0

    @torch.no_grad()
    def _sample_chunk(self, batch: dict, rtc_kw: dict | None = None) -> torch.Tensor:
        """跑一次模型，出 50 步动作（还是归一化空间的值）。

        走的是**训练时那条路**，不是 LeRobot 的在线接口 predict_action_chunk。
        在线接口假定触觉一帧帧喂进去、自己在内部攒历史；我们要提前算，历史必须自己
        快照。训练那条路直接吃摞好的历史，正合适。两条路的输出逐个数对过：
        位移最大差 0.005 毫米、转角最大差 0.00007 度，是浮点噪声。
        """
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
        p = self.policy
        images, img_masks = p.prepare_images(batch)
        state = p.prepare_state(batch)
        tac_grid, tac_force = p.prepare_tactile(batch)
        kw = {"tactile_force_grid": tac_grid, "tactile_resultant_force": tac_force}
        if (getattr(p.config, "tactile_r1_enabled", False)
                or getattr(p.config, "tactile_ibr_enabled", False)
                or p._needs_r1_history_mask()):
            kw["tactile_history_is_pad"] = p._r1_training_history_mask(batch, tac_grid)
        if rtc_kw:
            kw.update(rtc_kw)
        out = p.model.sample_actions(
            images, img_masks, batch[f"{OBS_LANGUAGE_TOKENS}"],
            batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"], state, noise=None, **kw)
        return out          # 完整宽度（max_action_dim），RTC 的 prev 要用；切片在 _infer 做

    @torch.no_grad()
    def _infer(self, raw: dict, hist, rtc_kw: dict | None = None,
               t0: int = 0) -> np.ndarray:
        batch = dict(raw)
        g, m, s = hist
        if self.needs_hist:
            key = self.policy.config.tactile_force_grid_key
            batch[key] = g
            batch[f"{key}_is_pad"] = m
        if self.needs_state_hist:
            if s is None or tuple(s.shape[1:]) != (len(self.state_hist_offsets), 11):
                raise RuntimeError(
                    "state-history snapshot has wrong shape: "
                    f"{None if s is None else tuple(s.shape)}"
                )
            # Put the complete sparse history into the normal preprocessing path so
            # every tap receives the checkpoint's observation.state normalization.
            batch["observation.state"] = s
        batch = self.pre(batch)
        if self.needs_hist:                            # 预处理不搬这个键，自己搬
            batch[f"{key}_is_pad"] = batch[f"{key}_is_pad"].to(self.device)
        full = self._sample_chunk(batch, rtc_kw or {})     # (1, 50, max_action_dim) 归一化
        if self.rtc:
            self._last_chunk = full.detach()               # 下一次 RTC 的 prev
            self._last_t0 = t0                             # 该块 step k <-> 拍 t0+k
        out = self.post(full[:, :, : self.policy.config.action_feature.shape[0]])
        return out.squeeze(0).float().cpu().numpy()    # (50, 7)

    def _absorb(self, chunk: np.ndarray, t0: int) -> None:
        """把新算出来的一串接上。跳过开算之后已经过去的那几步，保证动作和时间对得上。"""
        skip = max(0, self._tick - t0)
        take = chunk[skip: skip + self.n_action_steps]
        if len(take) == 0:
            # 到不了这儿：_join() 里的 join 没有超时，后台算不完主循环就卡着等，
            # 循环卡住 _tick 就不涨，所以 skip 上界是「开算时队列还剩几个」= trigger_left。
            # 实测把推理拖到 8 秒（慢 100 倍）skip 仍然只有 2。真到了这儿说明前提被改了
            # （比如有人给 join 加了超时），那就得重新想对齐，不能默默拿旧计划的末段顶上。
            raise RuntimeError(
                f"动作块对齐越界：skip={skip} 超过块长 {len(chunk)}。"
                "说明后台推理和主循环的节拍前提变了，请重新检查 _launch/_join/_absorb。")
        self._acts.extend(take)

    # ------------------------------------------------------------------ 节拍
    def wait_for_next_tick(self) -> None:
        """在循环最开头调，睡到这一拍该开始的时刻。

            while not done:
                runner.wait_for_next_tick()      # ← 放在最前面
                读传感器()                        # 睡完再读，数据才是新鲜的
                action = runner.step(...)
                发给机械臂(action)

        为什么必须掐表：step() 本身只要 2 毫秒，循环不掐表会跑到几百赫兹。
        模型输出的是"这一拍走多远"，发得越频繁手臂越快——几百赫兹意味着手臂
        以训练时十几倍的速度撞上去。这是真机上最危险的一种错法。

        为什么不做在 step() 里：那样得在读完传感器之后再睡，等于拿一拍之前的
        画面去算动作。放在循环开头睡完再读，数据才是新鲜的。

        落后了不补：宁可这一拍晚一点，也不要为了追进度连发几拍把手臂甩出去。
        """
        now = time.perf_counter()
        if self._t_next is None or now > self._t_next + self._period:
            self._t_next = now + self._period       # 第一次，或者落后太多，重新起算
            return
        if now < self._t_next:
            time.sleep(self._t_next - now)
        self._t_next += self._period

    def _check_rate(self) -> None:
        now = time.perf_counter()
        # 开头几拍在追赶第一次推理的耗时，节奏本来就不准，不查
        if self._tick > 8 and self._t_last is not None:
            hz = 1.0 / max(now - self._t_last, 1e-9)
            if hz > CONTROL_HZ * 2:
                self._fast_run += 1
                # 连着这么多拍都快一倍以上 = 循环根本没掐表。手臂正在以数倍速度冲，
                # 这时候停下来远比继续跑安全，所以直接抛异常而不是打印警告。
                if self._fast_run >= 20:
                    raise RuntimeError(
                        f"控制循环跑到 {hz:.0f} Hz，部署目标是 {CONTROL_HZ:.2f} Hz —— "
                        "手臂会以数倍速度运动，已停下。请在循环开头调 "
                        "runner.wait_for_next_tick()，或自己掐表到约 11 Hz。")
            else:
                self._fast_run = 0
            if (hz < CONTROL_HZ * 0.8 or hz > CONTROL_HZ * 1.25) and self._tick % 100 == 9:
                print(f"[警告] 控制频率 {hz:.1f} Hz，部署目标是 {CONTROL_HZ:.2f} Hz。"
                      "慢一点没关系，快了危险。")
        self._t_last = now

    @property
    def tactile_ready(self) -> bool:
        return self.obs.tactile_ready
