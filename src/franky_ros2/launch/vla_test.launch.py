from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition

# Merges "Terminal 1" (teleop_record.launch.py's sensors/arm) and "Terminal 2"
# (smolvla_policy_node.py) from the README's "vla test" section into one launch
# file, with the policy node driving the arm instead of a keyboard/spacemouse
# teleop node. Still start the policy loop manually afterwards ("Terminal 3"):
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
    task = LaunchConfiguration('task')
    live = LaunchConfiguration('live')
    record_dir = LaunchConfiguration('record_dir')
    visualize_cameras = LaunchConfiguration('visualize_cameras')
    dynamics_factor = LaunchConfiguration('dynamics_factor')

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
            description='franky robot.relative_dynamics_factor, passed to franky_control_node_orientation'
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
            cmd=[venv_python, POLICY_SCRIPT, '--ckpt-dir', ckpt_dir, '--task', task,
                 '--record-dir', record_dir, '--live'],
            cwd=REPO_ROOT,
            output='screen',
            condition=IfCondition(live),
        ),
        ExecuteProcess(
            cmd=[venv_python, POLICY_SCRIPT, '--ckpt-dir', ckpt_dir, '--task', task,
                 '--record-dir', record_dir],
            cwd=REPO_ROOT,
            output='screen',
            condition=UnlessCondition(live),
        ),
    ])
