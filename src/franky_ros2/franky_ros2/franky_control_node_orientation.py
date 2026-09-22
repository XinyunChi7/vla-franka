import math
import threading

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32, String

from franky import Robot, Gripper, CartesianMotion, ReferenceType, Affine

JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
STATE_PUBLISH_RATE_HZ = 20.0


def quaternion_to_euler_deg(quaternion):
    x, y, z, w = quaternion
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return tuple(math.degrees(a) for a in (roll, pitch, yaw))


class FrankaOrientationController(Node):
    def __init__(self):
        super().__init__('franka_orientation_controller_node')

        self.pose_subscription = self.create_subscription(
            Pose,
            '/pose_topic',
            self.pose_callback,
            1        # 深度 1：回调卡住时不攒旧命令。相对位移是叠加的，攒 10 条解阻塞后
                     # 会合成一个 10 倍位移的目标冲出去。
        )
        # Absolute-pose sibling of /pose_topic: franky's ReferenceType.Relative composes
        # in the end-effector frame (confirmed against upstream franky docs), which is the
        # opposite of the base-frame convention used by e.g. the SmolVLA policy runner.
        # Callers that already compute a base-frame absolute target (current pose + base-frame
        # delta) publish here instead of re-deriving an EE-frame-relative message.
        self.pose_absolute_subscription = self.create_subscription(
            Pose,
            '/pose_topic_absolute',
            self.pose_absolute_callback,
            1        # 同上；绝对目标虽不叠加，但攒一串旧目标同样没有意义
        )
        self.gripper_subscription = self.create_subscription(
            Bool,
            '/gripper_topic',
            self.gripper_callback,
            10
        )
        # 急停必须能插队。位姿/夹爪回调默认落在同一个互斥回调组里串行执行，
        # 一旦 robot.move() 阻塞，estop 会排在后面等 —— 那正是最需要它的时候。
        self.command_callback_group = ReentrantCallbackGroup()
        self.command_subscription = self.create_subscription(
            String,
            '/teleop_command',
            self.command_callback,
            10,
            callback_group=self.command_callback_group
        )
        self.declare_parameter('dynamics_factor', 0.10)
        self.is_gripped = False
        # The state timer (publish_robot_state) and gripper_callback run on different
        # executor threads and both talk to the same Gripper's UDP connection -- libfranka's
        # gripper socket isn't safe for concurrent access from two threads, so a width poll
        # landing mid-command causes one side to miss its reply and time out. Serialize them.
        self._gripper_lock = threading.Lock()
        self.get_logger().info("Initializing Franka robot...")

        try:
            self.robot = Robot("192.168.1.11")
            self.gripper = Gripper("192.168.1.11")
            # 动力学系数，改成 ROS 参数，不用改代码就能调：
            #   -p dynamics_factor:=0.07   （用 launch 启动时要写进 launch 的 parameters）
            # franky 里三个系数是**相乘**的（waypoint × motion × robot全局），这一行会乘到
            # 每一条运动上 —— delta 和绝对两条路都受它控制。参数只在启动时读一次。
            # 原值 0.01 太低：每条命令都在一拍内走不完就被下一条顶掉，六局 rollout 实测
            # 执行增益落在 0.29~0.63（口径不同 0.40~0.90；录制里没存当时的系数，无法逐档归因）。
            # 0.10 更接近 1.0，代价是速度脉动更大、手臂可能一顿一顿；嫌抖就调回 0.07~0.08。
            self._dyn = float(self.get_parameter('dynamics_factor').value)
            self.robot.relative_dynamics_factor = self._dyn
            self.get_logger().info("Franka robot initialized successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to connect to robot: {e}")
            raise e

        # Robot state is only readable from the process holding the FCI
        # connection (this node), so it's republished here for anything
        # else (e.g. the episode recorder) that needs joint/eef/gripper state.
        self.joint_state_publisher_ = self.create_publisher(JointState, '/franka_state/joint_states', 10)
        self.eef_pose_publisher_ = self.create_publisher(PoseStamped, '/franka_state/eef_pose', 10)
        self.gripper_width_publisher_ = self.create_publisher(Float32, '/franka_state/gripper_width', 10)
        # 诊断话题：指令俯仰 vs 实测俯仰（单位度）。录一条集就能定位姿态漂移在哪一层。
        # 控制器下发的目标位姿。实测位姿已经在 /franka_state/eef_pose 上了，这条是它的对照。
        self.eef_pose_cmd_publisher_ = self.create_publisher(PoseStamped, '/franka_state/eef_pose_commanded', 10)
        # Runs in its own callback group so a blocking robot.move() triggered
        # from command_callback (e.g. goto_predefined_pose) can't starve state
        # publishing — otherwise the episode recorder sees frozen joint/eef
        # state for the whole duration of the move even though the arm is moving.
        self.state_callback_group = MutuallyExclusiveCallbackGroup()
        self.state_timer = self.create_timer(
            1.0 / STATE_PUBLISH_RATE_HZ,
            self.publish_robot_state,
            callback_group=self.state_callback_group,
        )

        # Predefined absolute poses reachable via the '0'-'9' keys, keyed by the
        # same digit string the teleop node sends ("goto:<key>"). Fill in with
        # translation ([x, y, z] meters) / quaternion ([x, y, z, w]) values
        # read from the 'P' (print current pose) command once you've jogged
        # somewhere useful.
        self.predefined_poses = {
            "0": (  # home
                [0.4741425656053237, 0.025099392108070592, 0.550046364510172],
                [0.9993035821497035, 0.022181745314996112, 0.029999036360078602, -0.0006153820100336882],
            ),
            "1": (  # init
                [0.4849618397189546, 0.025348210954546608, 0.35476096348264663],
                [0.9993871411761418, 0.021970554309668582, 0.027245563175736672, -0.0005622120073828814],
            ),
            # "2": (
            #     [0.3662100382766127, 0.005461479093704009, 0.08632916806871843],
            #     [0.9993035821497035, 0.022181745314996112, 0.029999036360078602, -0.0006153820100336882],
            # ),
            "2": (  # pregrasp
                [0.36137722734238753, 0.005220264960954153, 0.16001683851772802],
                [0.9993422789312056, 0.021736683240371726, 0.02902060662072909, -0.0005749205657895846],
            ),
            "3": (  # pre insert
                [0.5328935265592352, -0.022386839553292348, 0.1159267274474045],
                [0.9998948317233309, -0.009373716883951961, -0.00991598987469203, -0.004912440285262815],
            ),
        }


    def pose_callback(self, msg: Pose):
        translation = [msg.position.x, msg.position.y, msg.position.z]
        quaternion = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        try:
            motion = CartesianMotion(
                Affine(translation, quaternion),
                reference_type=ReferenceType.Relative,
                # 系数 0.01 时每条命令一拍内跑不完，franky 走的是廉价的「只换目标」；
                # 提到 0.10 之后 40~60ms 就跑完了，每拍都会「等旧线程退出+重进实时循环」，
                # 一秒十几次，会引发通信约束违例或不连续性反射。加这个就不会结束运动。
                return_when_finished=False,
            )
            self.robot.move(motion, asynchronous=True)
        except Exception as e:
            self.get_logger().error(f"Motion command failed: {e}")

    def pose_absolute_callback(self, msg: Pose):
        translation = [msg.position.x, msg.position.y, msg.position.z]
        quaternion = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        try:
            motion = CartesianMotion(
                Affine(translation, quaternion),
                reference_type=ReferenceType.Absolute,
                # 这里**不要**再传 relative_dynamics_factor：它会和上面的全局值相乘。
                # 原来写 0.15，实际生效的是 0.15 × 0.01 = 0.0015，比相对路径还慢 6.7 倍，
                # 手臂几乎不动 —— 之前「绝对模式净位移≈0」就是这么来的。
                # 慢速段每条目标 40ms 内就跑完，不加这个会每秒重启 16 次控制会话。
                return_when_finished=False,
            )
            self.robot.move(motion, asynchronous=True)
        except Exception as e:
            self.get_logger().error(f"Absolute motion command failed: {e}")

    def command_callback(self, msg: String):
        command = msg.data
        try:
            if command == "print_pose":
                self.print_current_pose()
            elif command == "estop":
                # Hard motion abort at the controller level -- unlike just withholding new
                # pose_topic/pose_topic_absolute messages, this cancels whatever motion the
                # robot is currently mid-execution on, rather than letting it run to completion.
                # Still secondary to the physical e-stop button, which cuts power/engages
                # brakes regardless of software state -- keep that within reach too.
                self.robot.stop()
                self.get_logger().warn("ESTOP: robot.stop() called. Robot may now be in a reflex/error "
                                        "state -- send 'recover' on /teleop_command before further motion.")
            elif command == "recover":
                ok = self.robot.recover_from_errors()
                self.get_logger().info(f"recover_from_errors() -> {ok}")
            elif command.startswith("goto:"):
                self.goto_predefined_pose(command.split(":", 1)[1])
            elif command.startswith("moveto:"):
                self.moveto_pose(command.split(":", 1)[1])
            else:
                self.get_logger().warn(f"Unknown teleop command: {command}")
        except Exception as e:
            self.get_logger().error(f"Command handling failed: {e}")

    def print_current_pose(self):
        pose = self.robot.current_pose.end_effector_pose
        translation = tuple(pose.translation)
        quaternion = tuple(pose.quaternion)
        roll, pitch, yaw = quaternion_to_euler_deg(quaternion)
        self.get_logger().info(
            f"Current pose -> translation (m): {translation}, "
            f"quaternion (x,y,z,w): {quaternion}, "
            f"rpy (deg): ({roll:.1f}, {pitch:.1f}, {yaw:.1f})"
        )

    def goto_predefined_pose(self, name: str):
        if name not in self.predefined_poses:
            self.get_logger().warn(
                f"No predefined pose '{name}'. Known poses: {sorted(self.predefined_poses.keys())}"
            )
            return
        translation, quaternion = self.predefined_poses[name]
        self.get_logger().info(f"Moving to predefined pose '{name}': {translation}, {quaternion}")
        # 全局系数从 0.01 提到 0.10 之后，这里补一个 0.10，
        # 相乘后仍是 0.01 —— 预设点位的移动速度和改动前完全一样。
        motion = CartesianMotion(
            Affine(translation, quaternion),
            reference_type=ReferenceType.Absolute,
            # 预设点位保持原来的慢速：相乘后恒为 0.01，与全局系数怎么调无关
            relative_dynamics_factor=min(1.0, 0.01 / max(1e-6, self._dyn))
        )
        self.robot.move(motion, asynchronous=True)

    def publish_robot_state(self):
        try:
            state = self.robot.state
            now = self.get_clock().now().to_msg()

            joint_msg = JointState()
            joint_msg.header.stamp = now
            joint_msg.name = JOINT_NAMES
            joint_msg.position = [float(v) for v in state.q]
            joint_msg.velocity = [float(v) for v in state.dq]
            self.joint_state_publisher_.publish(joint_msg)

            # ★ 控制器**下发**的目标位姿 O_T_EE_c（franky 里是 Affine）。
            # 为什么必须发出来：实测位姿在下降过程中俯仰会自己掉 5~13 度，而策略全程
            # 只命令约 1 度旋转。光看实测分不出「目标本身在往下漂」还是「目标是平的、
            # 手臂没跟上」—— 这两种的修法完全相反。录制那边会把它和实测位姿存在同一个
            # 数组里（eef_pose_cmd），事后两条曲线一画就定位了。
            try:
                cmd = state.O_T_EE_c
                cmsg = PoseStamped()
                cmsg.header.stamp = now
                cx, cy, cz = cmd.translation
                cqx, cqy, cqz, cqw = cmd.quaternion
                cmsg.pose.position.x = float(cx)
                cmsg.pose.position.y = float(cy)
                cmsg.pose.position.z = float(cz)
                cmsg.pose.orientation.x = float(cqx)
                cmsg.pose.orientation.y = float(cqy)
                cmsg.pose.orientation.z = float(cqz)
                cmsg.pose.orientation.w = float(cqw)
                self.eef_pose_cmd_publisher_.publish(cmsg)
            except Exception as e:
                self.get_logger().warn(f"O_T_EE_c 取不到（诊断话题不发，不影响控制）: {e}",
                                       once=True)

            eef_pose = self.robot.current_pose.end_effector_pose
            pose_msg = PoseStamped()
            pose_msg.header.stamp = now
            tx, ty, tz = eef_pose.translation
            qx, qy, qz, qw = eef_pose.quaternion
            pose_msg.pose.position.x = float(tx)
            pose_msg.pose.position.y = float(ty)
            pose_msg.pose.position.z = float(tz)
            pose_msg.pose.orientation.x = float(qx)
            pose_msg.pose.orientation.y = float(qy)
            pose_msg.pose.orientation.z = float(qz)
            pose_msg.pose.orientation.w = float(qw)
            self.eef_pose_publisher_.publish(pose_msg)

            with self._gripper_lock:
                width = self.gripper.width
            self.gripper_width_publisher_.publish(Float32(data=float(width)))
        except Exception as e:
            self.get_logger().error(f"Robot state publish failed: {e}")

    def gripper_callback(self, msg: Bool):
        try:
            if msg.data and not self.is_gripped:
                self.is_gripped = True
                with self._gripper_lock:
                    self.gripper.grasp_async(0.0, 0.1, 50.0, epsilon_outer=0.2)
            elif not msg.data and self.is_gripped:
                self.is_gripped = False
                with self._gripper_lock:
                    self.gripper.open_async(0.1)
        except Exception as e:
            self.get_logger().error(f"Gripper command failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = FrankaOrientationController()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
