"""Hardware-free hello-world / demo launch for cv_tool.

Replays a recorded RealSense RGB-D rosbag and starts the cv_tool action server, so the
full detection pipeline can be exercised without a real camera or the original industrial
setup. The rosbag is distributed separately as an external download (see
``examples/bags/README.md``) because raw RGB-D recordings are too large to keep in git.

Usage (inside the Docker container, with the bag mounted at /cv_tool_ws/bags):

    ros2 launch cv_tool cv_tool_replay.launch.py bag_path:=/cv_tool_ws/bags/boxes_0.db3

Then, in a second shell, send a goal:

    ros2 action send_goal /detect_tool cv_tool_interfaces/action/Detect \
        "{tool_name: screwdriver}" --feedback
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_path = PathJoinSubstitution([
        FindPackageShare('cv_tool'),
        'config',
        'config.yaml',
    ])

    bag_path = LaunchConfiguration('bag_path')
    rate = LaunchConfiguration('rate')

    return LaunchDescription([
        DeclareLaunchArgument(
            'bag_path',
            default_value='/cv_tool_ws/bags/boxes_0.db3',
            description='Path to the downloaded RGB-D rosbag (directory with metadata.yaml '
                        'or a .db3 file). See examples/bags/README.md.',
        ),
        DeclareLaunchArgument(
            'rate',
            default_value='1.0',
            description='Playback rate multiplier passed to ros2 bag play.',
        ),

        # Replay the recorded RealSense streams (looped) on the topics cv_tool subscribes to.
        ExecuteProcess(
            cmd=['ros2', 'bag', 'play', bag_path, '--read-ahead-queue-size', '500',
                 '--rate', rate, '--loop'],
            output='screen',
        ),

        # The detection action server.
        Node(
            package='cv_tool',
            executable='cv_tool',
            name='cv_tool_action_server',
            output='screen',
            arguments=['--config', config_path],
        ),
    ])
