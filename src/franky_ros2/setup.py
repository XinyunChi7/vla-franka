from setuptools import find_packages, setup

package_name = 'franky_ros2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/vla_test.launch.py']),
        ('share/' + package_name + '/launch', ['launch/vla_test_teleop.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='xinyun',
    maintainer_email='xinyun.chi22@gmail.com',
    description='Franka arm control node and SmolVLA policy bridge for the realrobot-plug-smolvla checkpoint rollout.',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'franky_control_node_orientation = franky_ros2.franky_control_node_orientation:main',
            'keyboard_teleop_pose_orientation = franky_ros2.keyboard_teleop_pose_orientation:main',
            'episode_recorder_node = franky_ros2.episode_recorder_node:main',
        ],
    },
)
