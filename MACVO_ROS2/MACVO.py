import rclpy
import torch
import pypose as pp
import cv2
import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path as NavPath
from message_filters import ApproximateTimeSynchronizer, Subscriber
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Time
import tf2_ros

from pathlib import Path
from typing import TYPE_CHECKING
import os, sys
import argparse
import logging

from .MessageFactory import to_stamped_pose, from_image, to_pointcloud, to_image

# Add the src directory to the Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, src_path)
if TYPE_CHECKING:
    from src.Odometry.MACVO import MACVO
    from src.DataLoader import StereoFrame, StereoData, SmartResizeFrame
    from src.Utility.Config import load_config
    from src.Utility.PrettyPrint import Logger
    from src.Utility.Timer import Timer
else:
    import DataLoader
    from Odometry.MACVO import MACVO                
    from DataLoader import StereoFrame, StereoData, SmartResizeFrame
    from Utility.Config import load_config
    from Utility.PrettyPrint import Logger
    from Utility.Timer import Timer

PACKAGE_NAME = "MACVO_ROS2"


class MACVONode(Node):
    def __init__(
        self,
        config: Path,
        imageL_subscribe: str,
        imageR_subscribe: str,
        pose_publish: str,
        map_pc_publish: str,
        imageL_publish: str,
        odom_publish: str = "/macvo/odometry",
        path_publish: str = "/macvo/path",
        odom_frame: str = "odom",
        camera_frame: str = "camera_link",
        publish_tf: bool = True
    ) -> None:
        super().__init__("macvo")
        self.get_logger().set_level(logging.INFO)
        self.get_logger().info(f"Working Directory: {os.getcwd()}")

        self.odom_frame = odom_frame
        self.camera_frame = camera_frame
        self.publish_tf = publish_tf
        self.coord_frame = odom_frame

        if self.publish_tf:
            self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        imageL_recv = Subscriber(self, Image, imageL_subscribe, qos_profile=1)
        imageR_recv = Subscriber(self, Image, imageR_subscribe, qos_profile=1)
        self.stereo_recv = ApproximateTimeSynchronizer([imageL_recv, imageR_recv], queue_size=20, slop=0.08)
        self.stereo_recv.registerCallback(self.receive_stereo)

        self.pose_send = self.create_publisher(PoseStamped, pose_publish, qos_profile=1)
        self.map_send  = self.create_publisher(PointCloud, map_pc_publish, qos_profile=1)
        self.img_send  = self.create_publisher(Image, imageL_publish, qos_profile=1)
        self.odom_send = self.create_publisher(Odometry, odom_publish, qos_profile=1)
        self.path_send = self.create_publisher(NavPath, path_publish, qos_profile=1)

        self.path_msg = NavPath()
        self.path_msg.header.frame_id = self.odom_frame

        # Start MAC-VO
        cfg, _ = load_config(config)
        self.frame_id, self.camera = 0, cfg.Camera

        # Resolve weight path
        weight_path = Path(cfg.Odometry.frontend.args.weight)
        if not weight_path.is_file():
            try:
                share_dir = Path(get_package_share_directory(PACKAGE_NAME))
                if (share_dir / weight_path).is_file():
                    cfg.Odometry.frontend.args.weight = str(share_dir / weight_path)
            except Exception:
                pass

        # Load model from the resources directory automatically.
        original_cwd = os.getcwd()
        try:
            try:
                share_dir = get_package_share_directory(PACKAGE_NAME)
                os.chdir(share_dir)
            except Exception:
                pass
            self.odometry = MACVO[StereoFrame].from_config(cfg)
        finally:
            os.chdir(original_cwd)

        # Setup image resizing/cropping (default 320x320)
        resize_h = getattr(self.camera, "target_height", 320)
        resize_w = getattr(self.camera, "target_width", 320)
        self.frame_fn = SmartResizeFrame({"height": resize_h, "width": resize_w, "interp": "bilinear"})

        # Check stereo rectification configuration
        self.do_rectify = getattr(self.camera, "rectify", False)
        if self.do_rectify:
            try:
                K0 = np.array(self.camera.K0, dtype=np.float64).reshape((3, 3))
                D0 = np.array(self.camera.D0, dtype=np.float64)
                K1 = np.array(self.camera.K1, dtype=np.float64).reshape((3, 3))
                D1 = np.array(self.camera.D1, dtype=np.float64)
                R  = np.array(self.camera.R, dtype=np.float64).reshape((3, 3))
                T  = np.array(self.camera.T, dtype=np.float64)
                w  = int(getattr(self.camera, "width", 640))
                h  = int(getattr(self.camera, "height", 480))
                R0, R1, P0, P1, _, _, _ = cv2.stereoRectify(
                    K0, D0, K1, D1, (w, h), R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
                )
                self.mapL1, self.mapL2 = cv2.initUndistortRectifyMap(K0, D0, R0, P0, (w, h), cv2.CV_32FC1)
                self.mapR1, self.mapR2 = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (w, h), cv2.CV_32FC1)
                self.get_logger().info("Stereo rectification initialized successfully.")
            except Exception as e:
                self.get_logger().warn(f"Failed to initialize rectification: {e}. Running without rectification.")
                self.do_rectify = False

        self.odometry.register_on_optimize_finish(self.publish_data)
        self.get_logger().info(f"MAC-VO Node successfully initialized (listening to {imageL_subscribe} & {imageR_subscribe})")

    def publish_data(self, system: MACVO):
        # Latest pose
        pose = pp.SE3(system.graph.frames.data["pose"][-1])
        time_ns = int(system.graph.frames.data["time_ns"][-1].item())
        time = Time()
        time.sec = time_ns // 1_000_000_000
        time.nanosec = time_ns % 1_000_000_000
        pose_msg = to_stamped_pose(pose, self.coord_frame, time)

        # Publish PoseStamped
        self.pose_send.publish(pose_msg)

        # Publish NavPath
        self.path_msg.header.stamp = time
        self.path_msg.poses.append(pose_msg)
        if len(self.path_msg.poses) > 5000:
            self.path_msg.poses.pop(0)
        self.path_send.publish(self.path_msg)

        # Publish Odometry
        odom_msg = Odometry()
        odom_msg.header.stamp = time
        odom_msg.header.frame_id = self.odom_frame
        odom_msg.child_frame_id = self.camera_frame
        odom_msg.pose.pose = pose_msg.pose
        self.odom_send.publish(odom_msg)

        # Publish TF
        if self.publish_tf:
            tf_msg = TransformStamped()
            tf_msg.header.stamp = time
            tf_msg.header.frame_id = self.odom_frame
            tf_msg.child_frame_id = self.camera_frame
            tf_msg.transform.translation.x = pose_msg.pose.position.x
            tf_msg.transform.translation.y = pose_msg.pose.position.y
            tf_msg.transform.translation.z = pose_msg.pose.position.z
            tf_msg.transform.rotation = pose_msg.pose.orientation
            self.tf_broadcaster.sendTransform(tf_msg)

        # Latest map
        if system.mapping:
            points = system.graph.get_frame2map(system.graph.frames[-2:-1])
        else:
            points = system.graph.get_match2point(system.graph.get_frame2match(system.graph.frames[-1:]))

        map_pc_msg = to_pointcloud(
            position  = points.data["pos_Tw"],
            keypoints = None,
            frame_id  = self.coord_frame,
            colors    = points.data["color"],
            time      = time,
        )
        self.map_send.publish(map_pc_msg)

    def receive_stereo(self, msg_imageL: Image, msg_imageR: Image) -> None:
        imageL, timestamp = from_image(msg_imageL), msg_imageL.header.stamp
        imageR            = from_image(msg_imageR)

        # Ensure 3-channel RGB format
        if msg_imageL.encoding == "mono8" or imageL.ndim == 2 or (imageL.ndim == 3 and imageL.shape[2] == 1):
            if imageL.ndim == 3:
                imageL = imageL[:, :, 0]
            imageL = cv2.cvtColor(imageL, cv2.COLOR_GRAY2RGB)
        elif msg_imageL.encoding in ("bgr8", "8UC3"):
            imageL = cv2.cvtColor(imageL, cv2.COLOR_BGR2RGB)

        if msg_imageR.encoding == "mono8" or imageR.ndim == 2 or (imageR.ndim == 3 and imageR.shape[2] == 1):
            if imageR.ndim == 3:
                imageR = imageR[:, :, 0]
            imageR = cv2.cvtColor(imageR, cv2.COLOR_GRAY2RGB)
        elif msg_imageR.encoding in ("bgr8", "8UC3"):
            imageR = cv2.cvtColor(imageR, cv2.COLOR_BGR2RGB)

        # Apply stereo rectification if enabled
        if self.do_rectify:
            imageL = cv2.remap(imageL, self.mapL1, self.mapL2, cv2.INTER_LINEAR)
            imageR = cv2.remap(imageR, self.mapR1, self.mapR2, cv2.INTER_LINEAR)

        time_ns = int(timestamp.sec * 1_000_000_000 + timestamp.nanosec)

        # Instantiate a frame and scale to the desired height & width
        stereo_frame = self.frame_fn(StereoFrame(
            idx    =[self.frame_id],
            time_ns=[time_ns],
            stereo =StereoData(
                T_BS=pp.identity_SE3(1),
                K   =torch.tensor([[
                    [self.camera.fx, 0.            , self.camera.cx],
                    [0.            , self.camera.fy, self.camera.cy],
                    [0.            , 0.            , 1.            ]
                ]]),
                baseline=torch.tensor([self.camera.bl]),
                time_ns=[time_ns],
                height=imageL.shape[0],
                width=imageL.shape[1],
                imageL=torch.tensor(imageL)[..., :3].float().permute(2, 0, 1).unsqueeze(0) / 255.,
                imageR=torch.tensor(imageR)[..., :3].float().permute(2, 0, 1).unsqueeze(0) / 255.,
            )
        ))
        self.odometry.run(stereo_frame)

        self.frame_id += 1

    def destroy_node(self):
        self.odometry.terminate()


def main():
    rclpy.init()

    # Create dummy node to read ROS parameters if available
    temp_node = rclpy.create_node("_temp_param_reader")
    temp_node.declare_parameter("config", "")
    temp_node.declare_parameter("imageL_topic", "")
    temp_node.declare_parameter("imageR_topic", "")
    temp_node.declare_parameter("pose_topic", "")
    temp_node.declare_parameter("map_topic", "")
    temp_node.declare_parameter("odom_topic", "")
    temp_node.declare_parameter("path_topic", "")
    temp_node.declare_parameter("odom_frame", "")
    temp_node.declare_parameter("camera_frame", "")
    temp_node.declare_parameter("publish_tf", True)

    p_config = temp_node.get_parameter("config").value
    p_imageL = temp_node.get_parameter("imageL_topic").value
    p_imageR = temp_node.get_parameter("imageR_topic").value
    p_pose = temp_node.get_parameter("pose_topic").value
    p_map = temp_node.get_parameter("map_topic").value
    p_odom = temp_node.get_parameter("odom_topic").value
    p_path = temp_node.get_parameter("path_topic").value
    p_odom_frame = temp_node.get_parameter("odom_frame").value
    p_camera_frame = temp_node.get_parameter("camera_frame").value
    p_publish_tf = temp_node.get_parameter("publish_tf").value
    temp_node.destroy_node()

    # CLI Arguments
    args = argparse.ArgumentParser()
    args.add_argument("--config", type=str, default="")
    args.add_argument("--imageL_topic", type=str, default="")
    args.add_argument("--imageR_topic", type=str, default="")
    args.add_argument("--pose_topic", type=str, default="")
    args.add_argument("--map_topic", type=str, default="")
    args.add_argument("--odom_topic", type=str, default="")
    args.add_argument("--path_topic", type=str, default="")
    args.add_argument("--odom_frame", type=str, default="")
    args.add_argument("--camera_frame", type=str, default="")
    args.add_argument("--timing", action="store_true", help="Record timing for system")
    parsed, _ = args.parse_known_args()

    # Determine default config path
    try:
        share_dir = Path(get_package_share_directory(PACKAGE_NAME))
        default_config = share_dir / "config" / "robot_macvo.yaml"
    except Exception:
        default_config = Path(__file__).resolve().parent / "config" / "robot_macvo.yaml"

    config_path = parsed.config or p_config or str(default_config)
    imageL_topic = parsed.imageL_topic or p_imageL or "/camera/left/image_raw"
    imageR_topic = parsed.imageR_topic or p_imageR or "/camera/right/image_raw"
    pose_topic = parsed.pose_topic or p_pose or "/macvo/pose"
    map_topic = parsed.map_topic or p_map or "/macvo/map"
    odom_topic = parsed.odom_topic or p_odom or "/macvo/odometry"
    path_topic = parsed.path_topic or p_path or "/macvo/path"
    odom_frame = parsed.odom_frame or p_odom_frame or "odom"
    camera_frame = parsed.camera_frame or p_camera_frame or "camera_link"

    # Optional Timer setup
    Timer.setup(active=parsed.timing)

    # Create Node and spin
    node = MACVONode(
        config=Path(config_path),
        imageL_subscribe=imageL_topic,
        imageR_subscribe=imageR_topic,
        pose_publish=pose_topic,
        map_pc_publish=map_topic,
        imageL_publish="/macvo/img",
        odom_publish=odom_topic,
        path_publish=path_topic,
        odom_frame=odom_frame,
        camera_frame=camera_frame,
        publish_tf=bool(p_publish_tf)
    )
    print("MACVO Node created and spinning...")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
