from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='cv_tool',
            executable='cv_tool',
            name='cv_tool_action_server',
            output='screen',
            arguments=[
                '--rgb_topic', '/camera/camera/color/image_raw',
                '--depth_topic', '/camera/camera/depth/image_rect_raw',
                '--model', '11n_int8',
                '--buffer_size', '8',
                '--conf_thres', '0.2',
                '--margin_x', '70',
                '--margin_y', '70',
                '--verbose'
            ],
        )
    ])
