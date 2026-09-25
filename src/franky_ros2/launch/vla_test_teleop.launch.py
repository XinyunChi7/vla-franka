from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition

# vla_test.launch.py, plus keyboard_teleop_pose_orientation.py and episode_recorder_node.py
# running alongside the policy so an operator can jog/correct the arm mid-rollout instead of
# only watching it. Holding a movement/gripper key on the keyboard teleop node auto-publishes
# "pause_policy" on /teleop_command (see keyboard_teleop_pose_orientation.py); releasing it for
# ~0.5s auto-publishes "resume_policy". This only stops smolvla_policy_node.py from sending new
# /pose_topic(_absolute)/gripper_topic commands for that window -- it does not e-stop the arm --
# so the keyboard's own commands land without the policy re-issuing a conflicting target on the
# very next control tick. Press T to fully toggle teleop-only mode instead: publishes
# "teleop_mode_on"/"teleop_mode_off" once, which holds the policy paused regardless of the
# debounce above -- useful when teleop commands are sent intermittently and the 0.5s auto-resume
# would otherwise let the policy sneak an action in between keystrokes. Press T again to hand
# control back to the policy. Still start the policy loop manually ("Terminal 2" in the README):
#   ros2 topic pub /teleop_command std_msgs/String "data: start_policy" --once

REPO_ROOT = '/home/xinyun/vlm-franka'
POLICY_SCRIPT = REPO_ROOT + '/src/franky_ros2/franky_ros2/smolvla_policy_node.py'


def generate_launch_description():
    camera1_serial = LaunchConfiguration('camera1_serial')
    camera2_serial = LaunchConfiguration('camera2_serial')
    use_feats = LaunchConfiguration('use_feats')
    feats_model_path = LaunchConfiguration('feats_model_path')
    feats_norm_path = LaunchConfiguration('feats_norm_path')
    venv_python = LaunchConfiguration('venv_python')
    ckpt_dir = LaunchConfiguration('ckpt_dir')
    deploy_pkg_dir = LaunchConfiguration('deploy_pkg_dir')
    task = LaunchConfiguration('task')
    live = LaunchConfiguration('live')
    record_dir = LaunchConfiguration('record_dir')
    visualize_cameras = LaunchConfiguration('visualize_cameras')
    dynamics_factor = LaunchConfiguration('dynamics_factor')
    output_dir = LaunchConfiguration('output_dir')
    sample_rate_hz = LaunchConfiguration('sample_rate_hz')

    return LaunchDescription([
        DeclareLaunchArgument(
            'camera1_serial',
            default_value="'317222072625'",
            description='Serial number of the wrist-mounted RealSense D435i'
        ),
        DeclareLaunchArgument(
            'camera2_serial',
            default_value="'243322073128'",
            description='Serial number of the third-person RealSense D435i'
        ),
        DeclareLaunchArgument(
            'use_feats',
            default_value='true',
            description='Use the FEATS neural force estimator instead of the '
                         'marker-displacement proxy (requires feats_model_path/feats_norm_path)'
        ),
        DeclareLaunchArgument(
            'feats_model_path',
            default_value=REPO_ROOT + '/gsrobotics/feats/src/feats/models/unet_09042025_124903_80.pt',
            description='Only used when use_feats:=true'
        ),
        DeclareLaunchArgument(
            'feats_norm_path',
            default_value=REPO_ROOT + '/gsrobotics/feats/src/feats/data/labels/normalization_08042025_122519.npy',
            description='Only used when use_feats:=true'
        ),
        DeclareLaunchArgument(
            'venv_python',
            default_value='/home/xinyun/lerobot-env/bin/python3',
            description='Interpreter with torch/transformers/lerobot_fork for smolvla_policy_node.py'
        ),
        DeclareLaunchArgument(
            'ckpt_dir',
            default_value='checkpoints/realrobot-plug-smolvla/ckpt_vision',
            description='Policy checkpoint dir, passed to smolvla_policy_node.py --ckpt-dir'
        ),
        DeclareLaunchArgument(
            'deploy_pkg_dir',
            default_value='checkpoints/realrobot-plug-smolvla',
            description="Deploy package root (has policy_runner.py, lerobot_fork/), passed to "
                        "smolvla_policy_node.py --deploy-pkg-dir. Needed explicitly because "
                        "newer checkpoints live several levels under this root (e.g. "
                        "checkpoints/D93_idle_tail_v4_jpeg/vision/20k), so the node can no "
                        "longer infer it from ckpt_dir's immediate parent."
        ),
        DeclareLaunchArgument(
            'task',
            default_value='insert the plug into the power strip',
            description='Task string, passed to smolvla_policy_node.py --task'
        ),
        DeclareLaunchArgument(
            'live',
            default_value='false',
            description='Actually publish motion/gripper commands. Default is dry-run (log only).'
        ),
        DeclareLaunchArgument(
            'record_dir',
            default_value='',
            description='If set, record each rollout (state/action/tactile every step, plus '
                         'camera images unless stopped as a known success) under this dir via '
                         'PolicyRunner/Recorder, e.g. runs/eval_0816. Empty disables recording.'
        ),
        DeclareLaunchArgument(
            'visualize_cameras',
            default_value='true',
            description='Open rqt_image_view windows on the wrist and thirdview camera feeds'
        ),
        DeclareLaunchArgument(
            'dynamics_factor',
            default_value='0.10',
            description='Robot-global relative_dynamics_factor for franky_control_node_orientation. '
                         'Multiplies with per-motion coefficients (see node source for details); '
                         'must be in (0, 1] or the node fails to connect. ros2 run -p does not work '
                         'here since this node is launched via this file, so this is the way to override it.'
        ),
        DeclareLaunchArgument(
            'output_dir',
            default_value='episodes',
            description='Directory episode_recorder_node saves full rollouts (incl. any '
                         'keyboard corrections) into, keyed by the R key start/stop toggle -- '
                         'independent of the policy-only recording under record_dir.'
        ),
        DeclareLaunchArgument(
            'sample_rate_hz',
            default_value='20.0',
            description='Rate at which episode_recorder_node samples steps while recording'
        ),
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='realsense_node',
            output='screen',
            parameters=[{
                'serial_no': camera1_serial,
                'enable_color': True,
                'enable_depth': True,
                'enable_infra1': False,
                'enable_infra2': False,
                'enable_gyro': False,
                'enable_accel': False,
            }]
        ),
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='realsense_thirdview_node',
            namespace='thirdview',
            output='screen',
            parameters=[{
                'serial_no': camera2_serial,
                'enable_color': True,
                'enable_depth': True,
                'enable_infra1': False,
                'enable_infra2': False,
                'enable_gyro': False,
                'enable_accel': False,
            }]
        ),
        Node(
            package='tactile_ros',
            executable='gelsight_node',
            name='gelsight_node',
            output='screen',
            parameters=[{
                'use_feats': use_feats,
                'feats_model_path': feats_model_path,
                'feats_norm_path': feats_norm_path,
            }]
        ),
        Node(
            package='franky_ros2',
            executable='franky_control_node_orientation',
            name='franky_control_node_orientation',
            output='screen',
            parameters=[{
                'dynamics_factor': dynamics_factor,
            }]
        ),
        Node(
            package='franky_ros2',
            executable='keyboard_teleop_pose_orientation',
            name='keyboard_teleop_pose_orientation',
            output='screen',
            parameters=[]
        ),
        Node(
            package='franky_ros2',
            executable='episode_recorder_node',
            name='episode_recorder_node',
            output='screen',
            parameters=[{
                'output_dir': output_dir,
                'sample_rate_hz': sample_rate_hz,
            }]
        ),
        Node(
            package='rqt_image_view',
            executable='rqt_image_view',
            name='rqt_wrist_view',
            arguments=['/camera/realsense_node/color/image_raw'],
            condition=IfCondition(visualize_cameras),
        ),
        Node(
            package='rqt_image_view',
            executable='rqt_image_view',
            name='rqt_thirdview_view',
            arguments=['/thirdview/realsense_thirdview_node/color/image_raw'],
            condition=IfCondition(visualize_cameras),
        ),
        # smolvla_policy_node.py can't run under `ros2 run` (needs the ~/lerobot-env
        # venv, not the ROS-managed Python) so it's launched directly with that
        # interpreter instead of a launch_ros Node action; cwd matches the README's
        # `cd /home/xinyun/vlm-franka` so the default relative ckpt_dir resolves.
        ExecuteProcess(
            cmd=[venv_python, POLICY_SCRIPT, '--ckpt-dir', ckpt_dir,
                 '--deploy-pkg-dir', deploy_pkg_dir, '--task', task,
                 '--record-dir', record_dir, '--live'],
            cwd=REPO_ROOT,
            output='screen',
            condition=IfCondition(live),
        ),
        ExecuteProcess(
            cmd=[venv_python, POLICY_SCRIPT, '--ckpt-dir', ckpt_dir,
                 '--deploy-pkg-dir', deploy_pkg_dir, '--task', task,
                 '--record-dir', record_dir],
            cwd=REPO_ROOT,
            output='screen',
            condition=UnlessCondition(live),
        ),
    ])
