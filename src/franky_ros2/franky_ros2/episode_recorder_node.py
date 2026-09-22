import os
import re
from datetime import datetime

import cv2
import h5py
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Pose, PoseStamped, WrenchStamped
from std_msgs.msg import Bool, Float32, String


def quat_mult(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


class EpisodeRecorderNode(Node):
    RESIZE_SIZE = (910, 512)  # (width, height) — 16:9 downscale of the 1280x720 raw feed (910.22 rounded, already even)

    def __init__(self):
        super().__init__('episode_recorder_node')

        self.declare_parameter('output_dir', 'episodes')
        self.declare_parameter('sample_rate_hz', 20.0)
        self.declare_parameter('wrist_camera_topic', '/camera/realsense_node/color/image_raw')
        self.declare_parameter('thirdview_camera_topic', '/thirdview/realsense_thirdview_node/color/image_raw')
        self.declare_parameter('tactile_image_topic', '/tactile/gelsight/image_raw')
        self.declare_parameter('tactile_force_topic', '/tactile/gelsight/force')
        self.declare_parameter('joint_state_topic', '/franka_state/joint_states')
        self.declare_parameter('eef_pose_topic', '/franka_state/eef_pose')
        self.declare_parameter('gripper_action_topic', '/gripper_topic')
        self.declare_parameter('gripper_width_topic', '/franka_state/gripper_width')
        # Source-tagged, published only by smolvla_policy_node.py / keyboard_teleop_pose_orientation.py
        # respectively (see those files), so a policy-issued action and a keyboard correction are
        # always recorded separately rather than composed into one number.
        self.declare_parameter('policy_action_pose_topic', '/policy_action/pose_delta')
        self.declare_parameter('policy_action_gripper_topic', '/policy_action/gripper')
        self.declare_parameter('teleop_action_pose_topic', '/teleop_action/pose_delta')
        self.declare_parameter('teleop_action_gripper_topic', '/teleop_action/gripper')

        self.output_dir = self.get_parameter('output_dir').get_parameter_value().string_value
        self.sample_rate_hz = self.get_parameter('sample_rate_hz').get_parameter_value().double_value

        self.bridge = CvBridge()

        self.latest_wrist_image = None
        self.latest_thirdview_image = None
        self.latest_tactile_image = None
        self.latest_tactile_force = None  # (fx, fy, fz, magnitude)
        self.latest_joint_state = None    # (position list, velocity list)
        self.latest_eef_pose = None       # (x, y, z, qx, qy, qz, qw)
        self.latest_gripper_action = False
        self.latest_gripper_width = None

        # Accumulates the delta pose(s) published to each source's own action topic since the
        # last record_step, rather than caching "latest" like the state fields above: a movement
        # command is a momentary action, not a persisted state, so it must be consumed (reset to
        # identity) once recorded -- otherwise the same commanded delta would be re-written on
        # every following step until the next key press, misrepresenting one action as many.
        # Multiple commands from the *same* source landing within one record tick are composed
        # together (translation summed, quaternion multiplied) so none are dropped, matching the
        # accumulation the teleop node itself does before publishing. Kept one per source (rather
        # than one shared accumulator) so a policy tick and a keyboard correction landing in the
        # same recorder tick are never blended into one indistinguishable number.
        self.pending_policy_pose_action = None
        self.pending_teleop_pose_action = None
        # Gripper is a persisted commanded state, not a momentary action (mirrors
        # latest_gripper_action above), tracked per source so it's always known which side
        # last asked for open/close even on ticks with no new message.
        self.latest_policy_gripper_action = False
        self.latest_teleop_gripper_action = False
        # True only on a tick where the operator's gripper key actually fired since the last
        # record_step -- unlike latest_teleop_gripper_action (persisted open/closed state),
        # this is consumed (reset to False) every tick, same reasoning as the pending pose
        # accumulators above, so it marks the instant of the keypress, not the state after it.
        self.teleop_gripper_fired_this_tick = False
        # Set from the same pause_policy/resume_policy commands smolvla_policy_node.py acts on
        # (see keyboard_teleop_pose_orientation.py) -- lets a consumer tell "policy was driving"
        # steps apart from "operator had taken over" ones without inferring it from the actions.
        # This is a debounced *state* (stays 1 for POLICY_RESUME_DEBOUNCE_SEC after the key is
        # released) -- see teleop_override below for the un-debounced ground-truth moment.
        self.policy_paused = False

        self.recording = False
        self.episode_path = None
        self.h5_file = None
        self.step_index = 0
        self.record_timer = None
        # True when the *current* recording was auto-started by start_policy below, so the
        # matching stop_policy knows it's the one that should end it. If the operator had
        # already started a recording by hand (R key) before start_policy fired, this stays
        # False and stop_policy leaves that manual recording alone -- only R/X can end it.
        self._policy_owns_episode = False

        self.video_paths = {}    # stream name -> file path, set at start_episode
        self.video_writers = {}  # stream name -> cv2.VideoWriter, created lazily on first frame

        topic = lambda name: self.get_parameter(name).get_parameter_value().string_value

        self.create_subscription(Image, topic('wrist_camera_topic'), self.wrist_image_callback, 10)
        self.create_subscription(Image, topic('thirdview_camera_topic'), self.thirdview_image_callback, 10)
        self.create_subscription(Image, topic('tactile_image_topic'), self.tactile_image_callback, 10)
        self.create_subscription(WrenchStamped, topic('tactile_force_topic'), self.tactile_force_callback, 10)
        self.create_subscription(JointState, topic('joint_state_topic'), self.joint_state_callback, 10)
        self.create_subscription(PoseStamped, topic('eef_pose_topic'), self.eef_pose_callback, 10)
        self.create_subscription(Bool, topic('gripper_action_topic'), self.gripper_action_callback, 10)
        self.create_subscription(Float32, topic('gripper_width_topic'), self.gripper_width_callback, 10)
        self.create_subscription(Pose, topic('policy_action_pose_topic'), self.policy_pose_action_callback, 10)
        self.create_subscription(Bool, topic('policy_action_gripper_topic'), self.policy_gripper_action_callback, 10)
        self.create_subscription(Pose, topic('teleop_action_pose_topic'), self.teleop_pose_action_callback, 10)
        self.create_subscription(Bool, topic('teleop_action_gripper_topic'), self.teleop_gripper_action_callback, 10)
        self.create_subscription(String, '/teleop_command', self.command_callback, 10)

        self.get_logger().info(
            "Episode recorder ready. Send 'start_episode' / 'stop_episode' / 'discard_episode' "
            "on /teleop_command to control recording."
        )

    # --- stream callbacks: just cache the latest sample of each stream ---

    def wrist_image_callback(self, msg: Image):
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.latest_wrist_image = cv2.resize(image, self.RESIZE_SIZE, interpolation=cv2.INTER_AREA)

    def thirdview_image_callback(self, msg: Image):
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.latest_thirdview_image = cv2.resize(image, self.RESIZE_SIZE, interpolation=cv2.INTER_AREA)

    def tactile_image_callback(self, msg: Image):
        self.latest_tactile_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def tactile_force_callback(self, msg: WrenchStamped):
        fx, fy, fz = msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z
        magnitude = (fx ** 2 + fy ** 2 + fz ** 2) ** 0.5
        self.latest_tactile_force = (fx, fy, fz, magnitude)

    def joint_state_callback(self, msg: JointState):
        self.latest_joint_state = (list(msg.position), list(msg.velocity))

    def eef_pose_callback(self, msg: PoseStamped):
        p, o = msg.pose.position, msg.pose.orientation
        self.latest_eef_pose = (p.x, p.y, p.z, o.x, o.y, o.z, o.w)

    @staticmethod
    def _accumulate_pose_delta(existing, msg: Pose):
        """Compose one more delta-pose command onto whatever's accumulated so far this record
        tick, matching the accumulation the teleop node itself does before publishing (see
        pending_policy_pose_action's docstring above)."""
        dx, dy, dz = msg.position.x, msg.position.y, msg.position.z
        quat = (msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)
        if existing is None:
            return (dx, dy, dz, *quat)
        px, py, pz, qx, qy, qz, qw = existing
        combined_quat = quat_mult(quat, (qx, qy, qz, qw))
        return (px + dx, py + dy, pz + dz, *combined_quat)

    def gripper_action_callback(self, msg: Bool):
        self.latest_gripper_action = msg.data

    def gripper_width_callback(self, msg: Float32):
        self.latest_gripper_width = msg.data

    def policy_pose_action_callback(self, msg: Pose):
        self.pending_policy_pose_action = self._accumulate_pose_delta(self.pending_policy_pose_action, msg)

    def policy_gripper_action_callback(self, msg: Bool):
        self.latest_policy_gripper_action = msg.data

    def teleop_pose_action_callback(self, msg: Pose):
        self.pending_teleop_pose_action = self._accumulate_pose_delta(self.pending_teleop_pose_action, msg)

    def teleop_gripper_action_callback(self, msg: Bool):
        self.latest_teleop_gripper_action = msg.data
        self.teleop_gripper_fired_this_tick = True

    # --- recording control ---

    def command_callback(self, msg: String):
        command = msg.data
        if command == "start_episode":
            self.start_episode()
        elif command == "stop_episode":
            self.stop_episode()
        elif command == "discard_episode":
            self.discard_episode()
        elif command == "pause_policy":
            self.policy_paused = True
        elif command == "resume_policy":
            self.policy_paused = False
        elif command == "teleop_mode_on":
            # T-key teleop-only toggle in keyboard_teleop_pose_orientation.py -- same
            # "policy isn't driving" signal as pause_policy, just latched instead of debounced.
            self.policy_paused = True
        elif command == "teleop_mode_off":
            self.policy_paused = False
        elif command == "start_policy":
            # Mirrors runs/ recording (Recorder inside PolicyRunner, gated the same way via
            # record_dir/start_policy in smolvla_policy_node.py) so a live rollout produces
            # both an episodes/episode_NN.h5 and a runs/ep_NNN without an extra R keypress.
            # No-op if a manual (R-key) recording is already running -- see _policy_owns_episode.
            if not self.recording:
                self.start_episode()
                self._policy_owns_episode = True
                self.get_logger().info("Recording auto-started for policy rollout (start_policy).")
        elif command == "stop_policy":
            if self._policy_owns_episode:
                self.stop_episode()

    def _next_episode_index(self) -> int:
        """Scan output_dir for existing episode_NN(.h5|_<stream>.mp4) files and
        return the next free index, so restarts continue the sequence instead
        of overwriting it."""
        pattern = re.compile(r'^episode_(\d+)(?:\.h5$|_(?:wrist|thirdview|tactile)\.mp4$)')
        existing = [
            int(match.group(1))
            for entry in os.listdir(self.output_dir)
            if (match := pattern.match(entry))
        ]
        return max(existing, default=-1) + 1

    def start_episode(self):
        if self.recording:
            self.get_logger().warn("Already recording an episode; ignoring start_episode.")
            return

        os.makedirs(self.output_dir, exist_ok=True)
        episode_name = f"episode_{self._next_episode_index():02d}"
        self.episode_path = os.path.join(self.output_dir, f"{episode_name}.h5")
        self.video_paths = {
            "wrist": os.path.join(self.output_dir, f"{episode_name}_wrist.mp4"),
            "thirdview": os.path.join(self.output_dir, f"{episode_name}_thirdview.mp4"),
            "tactile": os.path.join(self.output_dir, f"{episode_name}_tactile.mp4"),
        }
        self.video_writers = {}

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.h5_file = h5py.File(self.episode_path, "w")
        self.h5_file.attrs['created_at'] = timestamp
        self.h5_file.attrs['sample_rate_hz'] = self.sample_rate_hz
        self.h5_file.attrs['eef_pose_columns'] = "x,y,z,qx,qy,qz,qw"
        self.h5_file.attrs['action_policy_eef_pose_delta_columns'] = "dx,dy,dz,qx,qy,qz,qw"
        self.h5_file.attrs['action_teleop_eef_pose_delta_columns'] = "dx,dy,dz,qx,qy,qz,qw"
        self.h5_file.attrs['tactile_force_columns'] = "fx,fy,fz,magnitude"
        self.h5_file.attrs['image_channel_order'] = "bgr"

        self.step_index = 0
        self.pending_policy_pose_action = None
        self.pending_teleop_pose_action = None
        self.teleop_gripper_fired_this_tick = False
        self.recording = True
        self.record_timer = self.create_timer(1.0 / self.sample_rate_hz, self.record_step)
        self.get_logger().info(f"Recording started: {self.episode_path}")

    def stop_episode(self):
        if not self.recording:
            self.get_logger().warn("Not currently recording; ignoring stop_episode.")
            return
        episode_path = self.episode_path
        step_count = self.step_index
        self._finalize_recording()
        self.episode_path = None
        self.get_logger().info(f"Recording stopped: {episode_path} ({step_count} steps)")

    def discard_episode(self):
        # Only an episode currently being recorded can be discarded — once
        # stop_episode() has saved one, 'x' must not be able to delete it.
        if not self.recording:
            self.get_logger().warn("No active episode to discard.")
            return
        episode_path = self.episode_path
        video_paths = list(self.video_paths.values())
        self._finalize_recording()
        self.episode_path = None
        if episode_path and os.path.isfile(episode_path):
            os.remove(episode_path)
        for video_path in video_paths:
            if os.path.isfile(video_path):
                os.remove(video_path)
        self.get_logger().info(f"Recording discarded: {episode_path}")

    def _finalize_recording(self):
        self.recording = False
        self._policy_owns_episode = False
        if self.record_timer is not None:
            self.record_timer.cancel()
            self.record_timer = None
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None
        for writer in self.video_writers.values():
            writer.release()
        self.video_writers = {}
        self.video_paths = {}

    # --- HDF5 writing ---

    def _append(self, name: str, array):
        """Write one step's sample into a resizable dataset at /<name>,
        creating it (chunked, extendable along axis 0) the first time it's seen.
        `array=None` means this step has no sample for a stream whose shape isn't
        known yet (e.g. no camera frame has arrived at all so far) — skip it; once
        the dataset is created from the first real sample, earlier skipped steps
        are backfilled with zeros so every dataset stays aligned by step index."""
        if array is None:
            return
        dataset = self.h5_file.get(name)
        if dataset is None:
            dataset = self.h5_file.create_dataset(
                name,
                shape=(0, *array.shape),
                maxshape=(None, *array.shape),
                dtype=array.dtype,
                chunks=(1, *array.shape) if array.shape else (1,),
                compression="gzip" if array.ndim >= 2 else None,
            )
            if self.step_index > 0:
                dataset.resize(self.step_index, axis=0)  # backfill zeros for already-elapsed steps
        dataset.resize(dataset.shape[0] + 1, axis=0)
        dataset[-1] = array

    # --- video writing ---

    def _write_video_frame(self, name: str, frame):
        """Write one BGR frame into the mp4 for stream `name`, creating the
        writer (sized to this frame) the first time a real frame shows up.
        Like `_append`, frames before the stream's first real sample are
        simply not writable to video (no random-access rewrite like HDF5
        backfill), so those steps are just skipped for this stream."""
        if frame is None:
            return
        writer = self.video_writers.get(name)
        if writer is None:
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(self.video_paths[name], fourcc, self.sample_rate_hz, (width, height))
            self.video_writers[name] = writer
        writer.write(frame)

    def record_step(self):
        timestamp = self.get_clock().now().nanoseconds / 1e9

        self._append("images/wrist", self.latest_wrist_image)
        self._append("images/thirdview", self.latest_thirdview_image)
        self._append("images/tactile", self.latest_tactile_image)

        # self._write_video_frame("wrist", self.latest_wrist_image)
        # self._write_video_frame("thirdview", self.latest_thirdview_image)
        # self._write_video_frame("tactile", self.latest_tactile_image)

        joint_pos, joint_vel = self.latest_joint_state if self.latest_joint_state else ([np.nan] * 7, [np.nan] * 7)
        eef = self.latest_eef_pose if self.latest_eef_pose else [np.nan] * 7
        tactile = self.latest_tactile_force if self.latest_tactile_force else [np.nan] * 4

        self._append("joint_pos", np.asarray(joint_pos, dtype=np.float32))
        self._append("joint_vel", np.asarray(joint_vel, dtype=np.float32))
        self._append("eef_pose", np.asarray(eef, dtype=np.float32))
        # Zero translation + identity quaternion == "no movement commanded this step",
        # consistent with the relative CartesianMotion the controller would apply for it.
        NO_DELTA = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

        # Kept separate per source (never composed into one merged number) so a policy tick and
        # a keyboard correction landing in the same recorder tick stay individually attributable.
        policy_action = self.pending_policy_pose_action if self.pending_policy_pose_action is not None else NO_DELTA
        self._append("action/policy/eef_pose_delta", np.asarray(policy_action, dtype=np.float32))
        self.pending_policy_pose_action = None
        self._append("action/policy/gripper", np.asarray(int(self.latest_policy_gripper_action), dtype=np.uint8))

        # True the instant a teleop pose delta or gripper toggle actually landed this tick --
        # un-debounced ground truth for "the operator overrode the policy right now", unlike
        # policy_paused below which stays 1 for POLICY_RESUME_DEBOUNCE_SEC after the key is
        # released (see keyboard_teleop_pose_orientation.py).
        teleop_override = self.pending_teleop_pose_action is not None or self.teleop_gripper_fired_this_tick
        self._append("teleop_override", np.asarray(int(teleop_override), dtype=np.uint8))

        teleop_action = self.pending_teleop_pose_action if self.pending_teleop_pose_action is not None else NO_DELTA
        self._append("action/teleop/eef_pose_delta", np.asarray(teleop_action, dtype=np.float32))
        self.pending_teleop_pose_action = None
        self._append("action/teleop/gripper", np.asarray(int(self.latest_teleop_gripper_action), dtype=np.uint8))
        self.teleop_gripper_fired_this_tick = False

        # Debounced *state*: 1 while the operator is correcting and for a short tail after
        # release (see teleop_override above for the un-debounced per-tick moment instead).
        self._append("policy_paused", np.asarray(int(self.policy_paused), dtype=np.uint8))

        self._append("tactile_force", np.asarray(tactile, dtype=np.float32))
        self._append("gripper_action", np.asarray(int(self.latest_gripper_action), dtype=np.uint8))
        self._append(
            "gripper_width",
            np.asarray(self.latest_gripper_width if self.latest_gripper_width is not None else np.nan, dtype=np.float32)
        )
        self._append("timestamp", np.asarray(timestamp, dtype=np.float64))

        self.step_index += 1
        # Flush HDF5 metadata to disk ~once a second so a hard crash (kill -9,
        # power loss) leaves a readable file up to the last flush instead of an
        # unreadable one -- h5py/HDF5 only guarantees a valid file on close().
        if self.step_index % max(int(self.sample_rate_hz), 1) == 0:
            self.h5_file.flush()


def main(args=None):
    rclpy.init(args=args)
    node = EpisodeRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.recording:
            node.stop_episode()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
