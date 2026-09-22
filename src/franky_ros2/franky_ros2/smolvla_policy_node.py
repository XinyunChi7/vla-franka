"""Bridges the realrobot-plug-smolvla PolicyRunner to the live franky_ros2 topics.

Not runnable via `ros2 run` — it needs torch/transformers/the vendored lerobot_fork,
which live in the ~/lerobot-env venv, not the ROS-managed Python. Launch it directly
with that interpreter after sourcing the workspace so rclpy is still importable:

    cd /home/xinyun/vlm-franka
    source install/setup.bash
    /home/xinyun/lerobot-env/bin/python3 src/franky_ros2/franky_ros2/smolvla_policy_node.py \\
        --ckpt-dir checkpoints/realrobot-plug-smolvla/ckpt_vision

Defaults to dry_run=True: computes and logs the absolute target pose + gripper decision
every tick but publishes nothing, so you can eyeball behavior before it can move the arm.
Arm it with --live once you trust it. Before ever running --live, do the checkpoint
README's 2-minute sign-convention check by hand (feed the arm a stationary pose_topic_absolute
= current pose + [0.01, 0, 0] and confirm it moves 1cm along the *base* X axis).

Run loop is gated by /teleop_command ("start_policy" / "stop_policy"), matching the
start_episode/stop_episode idiom episode_recorder_node.py already uses.
"""
import argparse
import math
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, String
from scipy.spatial.transform import Rotation

CAMERA_RESIZE = (910, 512)  # (width, height), must match training preprocessing (see episode_recorder_node.RESIZE_SIZE)


def imgmsg_to_bgr(msg: Image) -> np.ndarray:
    """Decode a sensor_msgs/Image to a BGR ndarray without cv_bridge: cv_bridge's compiled
    extension is built against NumPy 1.x and hard-crashes (ABI mismatch) under the NumPy 2.x
    that torch/transformers pull into ~/lerobot-env. Only rgb8/bgr8 are handled -- the only
    encodings the realsense and gelsight drivers here actually publish -- and anything else
    raises rather than silently mis-coloring the image (the checkpoint README is explicit that
    swapped channel order won't error, it'll just quietly feed the model garbage)."""
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    if msg.encoding == 'bgr8':
        return arr
    if msg.encoding == 'rgb8':
        return arr[:, :, ::-1]
    raise ValueError(f"Unsupported image encoding {msg.encoding!r} on topic (expected rgb8/bgr8)")


class SmolVLAPolicyNode(Node):
    def __init__(self, ckpt_dir: str, deploy_pkg_dir: str, task: str, dry_run: bool, record_dir: str | None,
                 tick_log_path: str | None = None, *, gripper_close_confirm_ticks: int = 3,
                 gripper_open_confirm_ticks: int = 6, disable_gripper_debounce: bool = False,
                 control_hz: float = 11.0):
        super().__init__('smolvla_policy_node')
        if not math.isfinite(control_hz) or control_hz <= 0:
            raise ValueError(f"control_hz must be finite and positive, got {control_hz}")
        self.dry_run = dry_run
        self.tick_log_path = tick_log_path
        self._tick_wall_times: list[float] = []

        sys.path.insert(0, str(Path(deploy_pkg_dir) / "lerobot_fork" / "src"))
        sys.path.insert(0, str(deploy_pkg_dir))
        from policy_runner import PolicyRunner, DATASET_HZ  # noqa: E402  (path must be extended first)
        from gripper_debouncer import GripperDebouncer  # noqa: E402
        self.TRAINING_HZ = DATASET_HZ
        self.CONTROL_HZ = float(control_hz)
        self.gripper_debounce_enabled = not disable_gripper_debounce
        self.gripper_gate = GripperDebouncer(
            close_confirm_ticks=gripper_close_confirm_ticks if self.gripper_debounce_enabled else 1,
            open_confirm_ticks=gripper_open_confirm_ticks if self.gripper_debounce_enabled else 1,
        )

        self.declare_parameter('wrist_camera_topic', '/camera/realsense_node/color/image_raw')
        self.declare_parameter('thirdview_camera_topic', '/thirdview/realsense_thirdview_node/color/image_raw')
        self.declare_parameter('tactile_image_topic', '/tactile/gelsight/image_raw')
        self.declare_parameter('eef_pose_topic', '/franka_state/eef_pose')
        self.declare_parameter('gripper_width_topic', '/franka_state/gripper_width')
        topic = lambda name: self.get_parameter(name).get_parameter_value().string_value

        self.latest_wrist = None
        self.latest_thirdview = None
        self.latest_tactile = None
        self.latest_eef_pose = None      # (x,y,z,qx,qy,qz,qw)
        self.latest_eef_pose_cmd = None  # 控制器下发的目标位姿，同格式，仅录制用
        self.latest_gripper_width = None

        # Sensor callbacks (incl. two cv2.resize calls per camera frame) and the control
        # tick must NOT share a callback group: with the default single-threaded rclpy.spin(),
        # every image callback serializes with the tick timer on one thread, and if cameras
        # publish faster than the control loop, that alone can drag the achieved rate down.
        # Mirrors the separate callback group franky_control_node_orientation.py
        # already uses to keep state publishing from being starved by blocking robot.move() calls.
        self.sensor_callback_group = MutuallyExclusiveCallbackGroup()
        self.control_callback_group = MutuallyExclusiveCallbackGroup()

        self.create_subscription(Image, topic('wrist_camera_topic'), self._wrist_cb, 10,
                                  callback_group=self.sensor_callback_group)
        self.create_subscription(Image, topic('thirdview_camera_topic'), self._thirdview_cb, 10,
                                  callback_group=self.sensor_callback_group)
        self.create_subscription(Image, topic('tactile_image_topic'), self._tactile_cb, 10,
                                  callback_group=self.sensor_callback_group)
        self.create_subscription(PoseStamped, topic('eef_pose_topic'), self._eef_pose_cb, 10,
                                  callback_group=self.sensor_callback_group)
        # 控制器下发的目标位姿，只用于录制对照（诊断姿态漂移在哪一层）。
        # 收不到也不影响控制：录制里会存成 nan，事后一眼看得出这一局没记。
        self.create_subscription(PoseStamped, '/franka_state/eef_pose_commanded',
                                  self._eef_pose_cmd_cb, 10,
                                  callback_group=self.sensor_callback_group)
        self.create_subscription(Float32, topic('gripper_width_topic'), self._gripper_width_cb, 10,
                                  callback_group=self.sensor_callback_group)
        self.create_subscription(String, '/teleop_command', self._command_cb, 10,
                                  callback_group=self.sensor_callback_group)

        # /pose_topic (relative, franky's native + proven-at-high-frequency path), not
        # /pose_topic_absolute -- see the note in _tick() for why.
        self.pose_pub = self.create_publisher(Pose, '/pose_topic', 10)
        # 绝对位姿存档（action 10 维）走这条。哪条生效由 self.runner.absolute 决定，
        # 而它是 PolicyRunner 从存档里读出来的 —— 换存档不用改代码，也不会用错。
        self.pose_abs_pub = self.create_publisher(Pose, '/pose_topic_absolute', 10)
        self.gripper_pub = self.create_publisher(Bool, '/gripper_topic', 10)

        self.get_logger().info(f"Loading PolicyRunner from {ckpt_dir} (dry_run={dry_run}) ...")
        self.runner = PolicyRunner(ckpt_dir, task=task, record_dir=record_dir)
        mode = "ABSOLUTE pose -> /pose_topic_absolute" if self.runner.absolute \
            else "DELTA -> /pose_topic"
        self.get_logger().info(
            f"Loaded. action_dim={self.runner.action_dim} mode={mode} "
            f"uses_tactile={self.runner.uses_tactile} device={self.runner.device}")

        self.running = False
        self.timer = None
        if self.gripper_debounce_enabled:
            debounce_summary = (
                f"enabled close<=0 x{gripper_close_confirm_ticks}, "
                f"open>0 x{gripper_open_confirm_ticks}, reverse_lockout=off"
            )
        else:
            debounce_summary = "DISABLED (immediate sign edge)"
        self.get_logger().info(f"Gripper debounce: {debounce_summary}")
        self.get_logger().info(
            f"Policy publish rate: {self.CONTROL_HZ:.2f}Hz "
            f"(training data: {self.TRAINING_HZ:.2f}Hz)"
        )

    # --- cached sensor state ---
    def _wrist_cb(self, msg):
        img = imgmsg_to_bgr(msg)
        self.latest_wrist = cv2.resize(img, CAMERA_RESIZE, interpolation=cv2.INTER_AREA)

    def _thirdview_cb(self, msg):
        img = imgmsg_to_bgr(msg)
        self.latest_thirdview = cv2.resize(img, CAMERA_RESIZE, interpolation=cv2.INTER_AREA)

    def _tactile_cb(self, msg):
        self.latest_tactile = imgmsg_to_bgr(msg)

    def _eef_pose_cb(self, msg: PoseStamped):
        p, o = msg.pose.position, msg.pose.orientation
        self.latest_eef_pose = np.array([p.x, p.y, p.z, o.x, o.y, o.z, o.w], dtype=np.float32)

    def _eef_pose_cmd_cb(self, msg: PoseStamped):
        p, o = msg.pose.position, msg.pose.orientation
        self.latest_eef_pose_cmd = np.array([p.x, p.y, p.z, o.x, o.y, o.z, o.w], dtype=np.float32)

    def _gripper_width_cb(self, msg: Float32):
        self.latest_gripper_width = msg.data

    # --- start/stop gating, mirrors episode_recorder_node's start_episode/stop_episode ---
    def _command_cb(self, msg: String):
        if msg.data == "start_policy":
            self.start()
        elif msg.data == "stop_policy":
            self.stop(success=None)

    def start(self):
        if self.running:
            self.get_logger().warn("Policy loop already running; ignoring start_policy.")
            return
        missing = [n for n, v in [
            ("wrist", self.latest_wrist), ("thirdview", self.latest_thirdview),
            ("tactile", self.latest_tactile), ("eef_pose", self.latest_eef_pose),
            ("gripper_width", self.latest_gripper_width),
        ] if v is None]
        if missing:
            self.get_logger().error(f"Cannot start: no data yet on {missing}. Is teleop_record.launch.py fully up?")
            return
        self.runner.reset()
        # Initialize from the measured physical width, not from the first noisy
        # model output. D52-compatible open/closed plateaus are 81.2/27.0 mm.
        self.gripper_gate.reset(is_closed=float(self.latest_gripper_width) < 0.055)
        self.running = True
        self.timer = self.create_timer(1.0 / self.CONTROL_HZ, self._tick,
                                        callback_group=self.control_callback_group)
        self.get_logger().info(f"Policy loop started (dry_run={self.dry_run}).")

    def stop(self, success: bool | None):
        if not self.running:
            return
        self.running = False
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None
        self.runner.finish_episode(success=success)
        self._report_tick_rate()
        self.get_logger().info("Policy loop stopped.")

    def _report_tick_rate(self):
        ts = self._tick_wall_times
        if len(ts) < 10:
            return
        dt = np.diff(np.asarray(ts[5:]))  # drop the first few (warmup skews the mean)
        hz = 1.0 / dt
        self.get_logger().info(
            f"[tick-rate] n={len(dt)} mean={hz.mean():.2f}Hz median={np.median(hz):.2f}Hz "
            f"min={hz.min():.2f}Hz p05={np.percentile(hz,5):.2f}Hz target={self.CONTROL_HZ:.2f}Hz")
        if self.tick_log_path:
            np.save(self.tick_log_path, np.asarray(ts))
            self.get_logger().info(f"[tick-rate] raw timestamps saved to {self.tick_log_path}")
        self._tick_wall_times = []

    # --- main control tick ---
    def _tick(self):
        # This callback runs inside a ROS timer: if step()/publish raises, the executor
        # catches it at the callback boundary, logs a traceback to stderr (easy to miss in a
        # scrolling terminal), and just calls _tick() again next cycle -- the timer looks
        # perfectly healthy (still firing at the target rate) while nothing reaches the robot.
        # Catch and log loudly here instead of relying on that.
        self._tick_wall_times.append(time.perf_counter())
        try:
            self._tick_body()
        except Exception:
            # rclpy 的 logger 不认 logging 模块的 exc_info：传了会自己抛 TypeError，
            # 把真正的异常和栈整个吞掉 —— 这个兜底原本在出事那一刻才发作。
            self.get_logger().error(
                "Exception in _tick(), this cycle's action was NOT sent:\n"
                + traceback.format_exc())

    def _tick_delta(self, eef_pose):
        """增量存档（7 维）：基座系增量 -> 末端系 -> /pose_topic (Relative)。原有逻辑不变。"""
        action = self.runner.step(
            self.latest_thirdview, self.latest_wrist, self.latest_tactile,
            eef_pose, self.latest_gripper_width,
            eef_pose_cmd=self.latest_eef_pose_cmd,
        )
        dp, drot, gripper_val = action[:3], action[3:6], float(action[6])
        self.get_logger().info(f"[delta] dp={dp} drot={drot} grip={gripper_val:+.3f}")
        r_current = Rotation.from_quat(eef_pose[3:7])
        dp_local = r_current.inv().apply(dp)
        q_local = (r_current.inv() * Rotation.from_rotvec(drot) * r_current).as_quat()
        if self.dry_run:
            self.get_logger().info(f"[dry-run] dp_local={dp_local} q_local={q_local}")
        else:
            m = Pose()
            m.position.x, m.position.y, m.position.z = (float(v) for v in dp_local)
            m.orientation.x, m.orientation.y, m.orientation.z, m.orientation.w = (float(v) for v in q_local)
            self.pose_pub.publish(m)
        return gripper_val

    def _tick_absolute(self, eef_pose):
        """绝对存档（10 维）：直接发基座系绝对目标 -> /pose_topic_absolute (Absolute)。

        这里没有任何坐标系换算 —— 目标本来就在基座系。增量模式那个换算用的是**实测**
        姿态、而 franky 还原时用的是**指令**姿态，两者一旦不一致（正是姿态漂移时的情形），
        每条位移都会被整体转掉一个角度；绝对模式从结构上没有这个问题。

        step_absolute() 内部已经做完：rot6d 施密特正交化 -> 四元数；单拍目标相对实测
        跳变超过 10mm / 2 度 -> ok=False（只挡数量级异常的一拍，**不限制去哪儿**）。
        拒发时保持上一次目标，不要回退成「实测 + 增量」，那等于把刚去掉的耦合又加回来。
        """
        p, q, gripper_val, ok = self.runner.step_absolute(
            self.latest_thirdview, self.latest_wrist, self.latest_tactile,
            eef_pose, self.latest_gripper_width,
            eef_pose_cmd=self.latest_eef_pose_cmd,
        )
        if not ok:
            self.get_logger().warn(f"[abs] 目标被安全检查拒发，本拍不下发: p={p}")
            return None
        self.get_logger().info(f"[abs] p={p} q={q} grip={gripper_val:+.3f}")
        if self.dry_run:
            self.get_logger().info(f"[dry-run] 绝对目标 p={p} q={q}")
        else:
            m = Pose()
            m.position.x, m.position.y, m.position.z = (float(v) for v in p)
            m.orientation.x, m.orientation.y, m.orientation.z, m.orientation.w = (float(v) for v in q)
            self.pose_abs_pub.publish(m)
        return gripper_val

    def _tick_body(self):
        eef_pose = self.latest_eef_pose
        if self.runner.absolute:
            gripper_val = self._tick_absolute(eef_pose)
        else:
            gripper_val = self._tick_delta(eef_pose)
        if gripper_val is None:
            return                      # 这一拍被安全检查拦下，夹爪也不动

        # Gripper is a state, not a delta. Preserve the original sign rule but
        # require sustained evidence before publishing an edge when enabled.
        should_close = self.gripper_gate.update(gripper_val)
        if should_close is not None:
            if not self.dry_run:
                self.gripper_pub.publish(Bool(data=should_close))
            self.get_logger().info(
                f"Gripper debounced edge: {'close' if should_close else 'open'} "
                f"(model={gripper_val:+.3f})"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True, help="e.g. checkpoints/realrobot-plug-smolvla/ckpt_vision")
    ap.add_argument("--deploy-pkg-dir", default=None,
                     help="Defaults to the parent of --ckpt-dir (the realrobot-plug-smolvla directory).")
    ap.add_argument("--task", default="insert the plug into the power strip")
    ap.add_argument("--live", action="store_true", help="Actually publish motion/gripper commands. Default is dry-run (log only).")
    ap.add_argument("--record-dir", default=None)
    ap.add_argument("--tick-log", default=None,
                     help="Path to save raw per-tick wall-clock timestamps (.npy) on stop, for control-frequency analysis.")
    ap.add_argument("--gripper-close-confirm-ticks", type=int, default=3)
    ap.add_argument("--gripper-open-confirm-ticks", type=int, default=6)
    ap.add_argument("--disable-gripper-debounce", action="store_true",
                    help="Disable confirmation filtering and switch on the first sign edge.")
    ap.add_argument("--control-hz", type=float, default=11.0,
                    help="ROS policy publish rate. Defaults to 11Hz to give the arm more tracking time.")
    args, ros_args = ap.parse_known_args()

    deploy_pkg_dir = args.deploy_pkg_dir or str(Path(args.ckpt_dir).resolve().parent)

    rclpy.init(args=ros_args)
    node = SmolVLAPolicyNode(
        ckpt_dir=args.ckpt_dir, deploy_pkg_dir=deploy_pkg_dir, task=args.task,
        dry_run=not args.live, record_dir=args.record_dir, tick_log_path=args.tick_log,
        gripper_close_confirm_ticks=args.gripper_close_confirm_ticks,
        gripper_open_confirm_ticks=args.gripper_open_confirm_ticks,
        disable_gripper_debounce=args.disable_gripper_debounce,
        control_hz=args.control_hz,
    )
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop(success=None)
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
