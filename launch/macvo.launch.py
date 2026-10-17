import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('MACVO_ROS2')
    default_config_path = os.path.join(pkg_share, 'config', 'robot_macvo.yaml')
    default_rviz_path = os.path.join(pkg_share, 'rviz', 'macvo.rviz')

    use_sim_time = LaunchConfiguration('use_sim_time')
    config_file = LaunchConfiguration('config')
    imageL_topic = LaunchConfiguration('imageL_topic')
    imageR_topic = LaunchConfiguration('imageR_topic')
    publish_tf = LaunchConfiguration('publish_tf')
    odom_frame = LaunchConfiguration('odom_frame')
    camera_frame = LaunchConfiguration('camera_frame')
    rviz = LaunchConfiguration('rviz')
    rviz_config = LaunchConfiguration('rviz_config')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (bag) clock if true'
        ),
        DeclareLaunchArgument(
            'config',
            default_value=default_config_path,
            description='Path to MAC-VO YAML config file'
        ),
        DeclareLaunchArgument(
            'imageL_topic',
            default_value='/camera/left/image_raw',
            description='Topic name for Left camera image'
        ),
        DeclareLaunchArgument(
            'imageR_topic',
            default_value='/camera/right/image_raw',
            description='Topic name for Right camera image'
        ),
        DeclareLaunchArgument(
            'publish_tf',
            default_value='true',
            description='Whether to broadcast TF transform'
        ),
        DeclareLaunchArgument(
            'odom_frame',
            default_value='odom',
            description='Odometry parent frame id'
        ),
        DeclareLaunchArgument(
            'camera_frame',
            default_value='camera_left_optical_frame',
            description='Camera child frame id'
        ),
        DeclareLaunchArgument(
            'rviz',
            default_value='false',
            description='Launch RViz2 if true'
        ),
        DeclareLaunchArgument(
            'rviz_config',
            default_value=default_rviz_path,
            description='Path to RViz configuration file'
        ),
        Node(
            package='MACVO_ROS2',
            executable='MACVO',
            name='macvo',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'config': config_file,
                'imageL_topic': imageL_topic,
                'imageR_topic': imageR_topic,
                'publish_tf': publish_tf,
                'odom_frame': odom_frame,
                'camera_frame': camera_frame,
            }]
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            parameters=[{'use_sim_time': use_sim_time}],
            condition=IfCondition(rviz),
            output='screen'
        )
    ])
