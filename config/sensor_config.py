"""
传感器详细配置
为每种传感器提供详细的参数定义
"""

from typing import Dict, List, Tuple


class SensorConfig:
    """
    传感器配置类
    统一管理所有传感器的参数
    """

    def __init__(
        self,
        # 相机
        num_cameras: int = 4,
        camera_names: List[str] = None,
        camera_resolution: Tuple[int, int] = (1920, 1080),
        camera_input_size: Tuple[int, int] = (224, 224),
        image_tokens_per_camera: int = 196,

        # 时序
        num_history_frames: int = 3,
        frame_skip: int = 1,
        frame_spacing: int = 1,

        # 激光雷达
        enable_lidar: bool = False,
        lidar_num_points: int = 16384,
        lidar_encoding: str = "range_image",
        lidar_hidden_size: int = 512,

        # 毫米波雷达
        enable_radar: bool = False,
        radar_num_detections: int = 100,
        radar_hidden_size: int = 64,

        # GPS/IMU
        enable_gps_imu: bool = False,
        gps_imu_dims: int = 6,
    ):
        self.camera_names = camera_names or ["front", "left", "right", "rear"]
        assert len(self.camera_names) == num_cameras, \
            f"camera_names length ({len(self.camera_names)}) must match num_cameras ({num_cameras})"

        self.num_cameras = num_cameras
        self.camera_resolution = camera_resolution
        self.camera_input_size = camera_input_size
        self.image_tokens_per_camera = image_tokens_per_camera
        self.total_image_tokens = num_cameras * image_tokens_per_camera

        self.num_history_frames = num_history_frames
        self.frame_skip = frame_skip
        self.frame_spacing = frame_spacing

        self.enable_lidar = enable_lidar
        self.lidar_num_points = lidar_num_points
        self.lidar_encoding = lidar_encoding
        self.lidar_hidden_size = lidar_hidden_size

        self.enable_radar = enable_radar
        self.radar_num_detections = radar_num_detections
        self.radar_hidden_size = radar_hidden_size

        self.enable_gps_imu = enable_gps_imu
        self.gps_imu_dims = gps_imu_dims

    @property
    def total_input_tokens(self) -> int:
        """计算总输入 token 数（视觉 + 文本）"""
        return self.total_image_tokens

    def get_camera_config(self) -> Dict:
        """获取相机配置字典"""
        return {
            "num_cameras": self.num_cameras,
            "camera_names": self.camera_names,
            "resolution": self.camera_resolution,
            "input_size": self.camera_input_size,
            "tokens_per_camera": self.image_tokens_per_camera,
        }

    def get_temporal_config(self) -> Dict:
        """获取时序配置字典"""
        return {
            "num_frames": self.num_history_frames,
            "frame_skip": self.frame_skip,
            "frame_spacing": self.frame_spacing,
        }

    def get_sensor_summary(self) -> Dict:
        """获取所有传感器配置摘要"""
        summary = {
            "cameras": self.get_camera_config(),
            "temporal": self.get_temporal_config(),
        }
        if self.enable_lidar:
            summary["lidar"] = {
                "num_points": self.lidar_num_points,
                "encoding": self.lidar_encoding,
                "hidden_size": self.lidar_hidden_size,
            }
        if self.enable_radar:
            summary["radar"] = {
                "num_detections": self.radar_num_detections,
                "hidden_size": self.radar_hidden_size,
            }
        if self.enable_gps_imu:
            summary["gps_imu"] = {
                "dims": self.gps_imu_dims,
            }
        return summary

    def __repr__(self):
        return (f"SensorConfig(cameras={self.num_cameras}, frames={self.num_history_frames}, "
                f"lidar={self.enable_lidar}, radar={self.enable_radar}, gps_imu={self.enable_gps_imu})")


def get_default_sensor_config() -> SensorConfig:
    """获取默认传感器配置"""
    return SensorConfig(
        num_cameras=4,
        camera_names=["front", "left", "right", "rear"],
        num_history_frames=3,
        enable_lidar=False,
        enable_radar=False,
        enable_gps_imu=False,
    )
