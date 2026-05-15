from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_path = PathJoinSubstitution([
        FindPackageShare('cv_tool'),
        'config',
        'config.yaml',
    ])

    return LaunchDescription([
        Node(
            package='cv_tool',
            executable='cv_tool',
            name='cv_tool_action_server',
            output='screen',
            arguments=[
                '--config', config_path,
            ],
        )
    ])
