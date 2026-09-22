import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from std_msgs.msg import Bool, String
import math
import time
from pynput import keyboard


TRANSLATION_STEP = 0.001   # meters per tick
# TRANSLATION_STEP = 0.0025   # meters per tick
ROTATION_STEP = 0.05      # radians per tick

# Translation bindings: (dx, dy, dz)
translation_bindings = {
    'w': (TRANSLATION_STEP, 0.0, 0.0),
    's': (-TRANSLATION_STEP, 0.0, 0.0),
    'a': (0.0, TRANSLATION_STEP, 0.0),
    'd': (0.0, -TRANSLATION_STEP, 0.0),
    'q': (0.0, 0.0, TRANSLATION_STEP),
    'e': (0.0, 0.0, -TRANSLATION_STEP),
}

# Rotation bindings: (axis, angle)
rotation_bindings = {
    'u': ((1.0, 0.0, 0.0), ROTATION_STEP),   # roll +
    'o': ((1.0, 0.0, 0.0), -ROTATION_STEP),  # roll -
    'i': ((0.0, 1.0, 0.0), ROTATION_STEP),   # pitch +
    'k': ((0.0, 1.0, 0.0), -ROTATION_STEP),  # pitch -
    'n': ((0.0, 0.0, 1.0), ROTATION_STEP),   # yaw +
    'm': ((0.0, 0.0, 1.0), -ROTATION_STEP),  # yaw -
}

GRIPPER_KEY = 'j'
PRINT_POSE_KEY = 'p'
PREDEFINED_POSE_KEYS = set('0123456789')  # goto:0 .. goto:9
RECORD_TOGGLE_KEY = 'r'
DISCARD_EPISODE_KEY = 'x'
TELEOP_MODE_KEY = 't'

# Keys that count as a manual correction: while any of these is held, a running VLA policy
# (smolvla_policy_node.py) is told to pause so it stops fighting the operator's input on
# /pose_topic. Predefined-pose/record/print keys are deliberate, one-shot commands rather
# than corrections, so they don't trigger a pause.
POLICY_OVERRIDE_KEYS = set(translation_bindings) | set(rotation_bindings) | {GRIPPER_KEY}
POLICY_RESUME_DEBOUNCE_SEC = 0.5

pressed_keys = set()


def axis_angle_quaternion(axis, angle):
    ax, ay, az = axis
    half = angle / 2.0
    s = math.sin(half)
    return (ax * s, ay * s, az * s, math.cos(half))


def quat_mult(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


class KeyboardPoseOrientationPublisher(Node):
    def __init__(self):
        super().__init__('keyboard_pose_orientation_publisher')
        self.pose_publisher_ = self.create_publisher(Pose, '/pose_topic', 10)
        self.gripper_publisher_ = self.create_publisher(Bool, '/gripper_topic', 10)
        self.command_publisher_ = self.create_publisher(String, '/teleop_command', 10)
        # Mirrors of the two publishers above, purely so episode_recorder_node.py can log
        # "what did the operator actually send" separately from policy commands sharing the
        # same /pose_topic//gripper_topic. The robot controller doesn't subscribe to these.
        self.teleop_action_pose_publisher_ = self.create_publisher(Pose, '/teleop_action/pose_delta', 10)
        self.teleop_action_gripper_publisher_ = self.create_publisher(Bool, '/teleop_action/gripper', 10)

        self.is_gripped = False
        self.is_recording = False
        self.policy_paused = False
        self.last_override_time = None
        self.teleop_only_mode = False

        self.get_logger().info(
            "Keyboard teleop (with orientation) started. "
            "WASDQE = translate, UO/IK/NM = roll/pitch/yaw, J = toggle gripper, "
            "P = print current pose, 0-9 = go to predefined pose, "
            "R = start/stop recording episode, X = discard current episode, "
            "T = toggle teleop-only mode (fully stops the policy until pressed again), "
            "CTRL-C = exit. "
            "Holding a movement/gripper key pauses a running VLA policy for "
            f"correction; releasing it for {POLICY_RESUME_DEBOUNCE_SEC:.1f}s resumes it -- "
            "unless teleop-only mode is on, which stays paused regardless."
        )

        self.timer = self.create_timer(0.1, self.update_pose)

    def update_policy_override(self):
        """Auto-pause/resume smolvla_policy_node.py's control loop around manual corrections."""
        if self.teleop_only_mode:
            # Teleop-only mode already holds the policy paused for as long as it's on --
            # the debounced auto-pause/resume below is redundant here and would only add
            # confusing "resumed" log lines while the operator is still fully in control.
            return
        now = time.monotonic()
        if pressed_keys & POLICY_OVERRIDE_KEYS:
            self.last_override_time = now
            if not self.policy_paused:
                self.policy_paused = True
                self.command_publisher_.publish(String(data="pause_policy"))
                self.get_logger().info("Movement key held -> pause_policy")
        elif (self.policy_paused and self.last_override_time is not None
                and (now - self.last_override_time) >= POLICY_RESUME_DEBOUNCE_SEC):
            self.policy_paused = False
            self.command_publisher_.publish(String(data="resume_policy"))
            self.get_logger().info("Override released -> resume_policy")

    def update_pose(self):
        self.update_policy_override()

        if not pressed_keys:
            return

        dx = dy = dz = 0.0
        quat = (0.0, 0.0, 0.0, 1.0)

        for key in list(pressed_keys):
            if key in translation_bindings:
                tdx, tdy, tdz = translation_bindings[key]
                dx += tdx
                dy += tdy
                dz += tdz
            elif key in rotation_bindings:
                axis, angle = rotation_bindings[key]
                quat = quat_mult(axis_angle_quaternion(axis, angle), quat)
            elif key == GRIPPER_KEY:
                self.is_gripped = not self.is_gripped
                gripper_msg = Bool()
                gripper_msg.data = self.is_gripped
                self.gripper_publisher_.publish(gripper_msg)
                self.teleop_action_gripper_publisher_.publish(gripper_msg)
                self.get_logger().info(f"Gripper toggled: {'closed' if self.is_gripped else 'open'}")
                # 'j' is momentary — remove it so the toggle only fires once per press.
                pressed_keys.discard(GRIPPER_KEY)
            elif key == PRINT_POSE_KEY:
                self.command_publisher_.publish(String(data="print_pose"))
                pressed_keys.discard(PRINT_POSE_KEY)
            elif key in PREDEFINED_POSE_KEYS:
                self.command_publisher_.publish(String(data=f"goto:{key}"))
                self.get_logger().info(f"Requested predefined pose '{key}'")
                pressed_keys.discard(key)
            elif key == RECORD_TOGGLE_KEY:
                self.is_recording = not self.is_recording
                self.command_publisher_.publish(
                    String(data="start_episode" if self.is_recording else "stop_episode")
                )
                self.get_logger().info(f"Recording {'started' if self.is_recording else 'stopped'}")
                pressed_keys.discard(RECORD_TOGGLE_KEY)
            elif key == DISCARD_EPISODE_KEY:
                self.is_recording = False
                self.command_publisher_.publish(String(data="discard_episode"))
                self.get_logger().info("Discarding current episode")
                pressed_keys.discard(DISCARD_EPISODE_KEY)
            elif key == TELEOP_MODE_KEY:
                self.teleop_only_mode = not self.teleop_only_mode
                if self.teleop_only_mode:
                    self.policy_paused = True
                    self.last_override_time = None
                    self.command_publisher_.publish(String(data="teleop_mode_on"))
                    self.get_logger().info("Teleop-only mode ON -- policy fully stopped")
                else:
                    self.policy_paused = False
                    self.command_publisher_.publish(String(data="teleop_mode_off"))
                    self.get_logger().info("Teleop-only mode OFF -- policy back in control")
                pressed_keys.discard(TELEOP_MODE_KEY)
            elif key == '\x03':  # CTRL-C
                rclpy.shutdown()
                return

        if (dx, dy, dz) == (0.0, 0.0, 0.0) and quat == (0.0, 0.0, 0.0, 1.0):
            return

        pose = Pose()
        pose.position.x = dx
        pose.position.y = dy
        pose.position.z = dz
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quat
        self.pose_publisher_.publish(pose)
        self.teleop_action_pose_publisher_.publish(pose)
        self.get_logger().info(f"Published delta pose: position={pose.position}, orientation={pose.orientation}")


def on_press(key):
    try:
        k = key.char
        pressed_keys.add(k)
    except AttributeError:
        pass


def on_release(key):
    try:
        k = key.char
        if k in pressed_keys:
            pressed_keys.remove(k)
    except AttributeError:
        pass


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardPoseOrientationPublisher()
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
