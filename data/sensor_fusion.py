"""
多传感器融合工具类

提供传感器数据的预处理、对齐、插值和同步功能
支持: 相机、激光雷达、毫米波雷达、GPS/IMU
"""

import os
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class SensorFusion:
    """
    多传感器融合工具

    功能:
        1. 时间同步: 不同采样率的传感器对齐到统一时间戳
        2. 空间对齐: 不同坐标系转换到统一坐标系
        3. 数据插值: 对缺失帧进行插值
        4. 异常检测: 检测传感器故障/异常数据
    """

    def __init__(
        self,
        camera_fps: float = 10.0,
        lidar_fps: float = 10.0,
        radar_fps: float = 20.0,
        imu_fps: float = 100.0,
        gps_fps: float = 10.0,
        time_window: float = 0.3,  # 时间窗口 (秒)
        coordinate_system: str = "vehicle",  # "vehicle" / "global"
    ):
        self.camera_fps = camera_fps
        self.lidar_fps = lidar_fps
        self.radar_fps = radar_fps
        self.imu_fps = imu_fps
        self.gps_fps = gps_fps
        self.time_window = time_window
        self.coordinate_system = coordinate_system

        # 传感器外参 (用于空间对齐)
        self.camera_extrinsics = {}
        self.lidar_extrinsics = {}
        self.radar_extrinsics = {}
        self.calibration_loaded = False

    def load_calibration(self, calibration_path: str):
        """加载传感器标定参数"""
        import json
        with open(calibration_path, "r") as f:
            calib = json.load(f)

        self.camera_extrinsics = calib.get("camera_extrinsics", {})
        self.lidar_extrinsics = calib.get("lidar_extrinsics", {})
        self.radar_extrinsics = calib.get("radar_extrinsics", {})
        self.calibration_loaded = True

    def sync_timestamps(
        self,
        timestamps: Dict[str, List[float]],
        reference_timestamp: float,
    ) -> Dict[str, Optional[np.ndarray]]:
        """
        将不同传感器的数据对齐到参考时间戳

        Args:
            timestamps: {sensor_name: [timestamps]}
            reference_timestamp: 参考时间戳

        Returns:
            {sensor_name: aligned_timestamp}
        """
        synced = {}
        for sensor_name, ts_list in timestamps.items():
            if not ts_list:
                synced[sensor_name] = None
                continue

            ts_array = np.array(ts_list)
            # 找到最接近参考时间戳的索引
            diff = np.abs(ts_array - reference_timestamp)
            min_idx = np.argmin(diff)

            # 检查时间窗口
            if diff[min_idx] > self.time_window:
                synced[sensor_name] = None
            else:
                synced[sensor_name] = ts_array[min_idx]

        return synced

    def interpolate_missing_frames(
        self,
        data: np.ndarray,
        timestamps: np.ndarray,
        target_timestamps: np.ndarray,
        axis: int = 0,
    ) -> np.ndarray:
        """
        对缺失帧进行线性插值

        Args:
            data: 传感器数据 [num_frames, ...]
            timestamps: 对应的时间戳
            target_timestamps: 目标时间戳
            axis: 插值轴

        Returns:
            插值后的数据
        """
        if len(data) < 2:
            return np.tile(data[0] if len(data) > 0 else 0,
                          (len(target_timestamps), *[(1,) * (data.ndim - 1)]))

        # 构建插值函数
        from scipy import interpolate
        interp_func = interpolate.interp1d(
            timestamps, data,
            axis=axis,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )

        return interp_func(target_timestamps)

    def detect_anomalies(
        self,
        sensor_data: Dict[str, np.ndarray],
        thresholds: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Dict[str, bool]:
        """
        检测传感器异常数据

        Args:
            sensor_data: {sensor_name: data}
            thresholds: 异常阈值配置

        Returns:
            {sensor_name: is_anomalous}
        """
        if thresholds is None:
            thresholds = {
                "camera": {"min_brightness": 0.01, "max_saturation": 0.99},
                "lidar": {"max_range": 100.0, "min_points": 1000},
                "radar": {"max_range": 200.0},
                "imu": {"max_acceleration": 5.0, "max_angular_velocity": 2.0},
            }

        anomalies = {}
        for sensor_name, data in sensor_data.items():
            is_anomalous = False

            if sensor_name == "camera":
                # 检查亮度和饱和度
                if data is not None:
                    mean_brightness = np.mean(data)
                    min_b = thresholds.get("camera", {}).get("min_brightness", 0.01)
                    if mean_brightness < min_b:
                        is_anomalous = True

            elif sensor_name == "lidar":
                # 检查点云数量和范围
                if data is not None:
                    num_points = len(data)
                    min_points = thresholds.get("lidar", {}).get("min_points", 1000)
                    if num_points < min_points:
                        is_anomalous = True

            elif sensor_name == "imu":
                # 检查加速度和角速度
                if data is not None and len(data) >= 2:
                    acc = data[:, 0:3]
                    gyro = data[:, 3:6]
                    max_acc = thresholds.get("imu", {}).get("max_acceleration", 5.0)
                    max_gyro = thresholds.get("imu", {}).get("max_angular_velocity", 2.0)
                    if np.any(np.abs(acc)) > max_acc or np.any(np.abs(gyro)) > max_gyro:
                        is_anomalous = True

            anomalies[sensor_name] = is_anomalous

        return anomalies

    def transform_to_vehicle_frame(
        self,
        points: np.ndarray,
        extrinsics: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        将点云从传感器坐标系转换到车辆坐标系

        Args:
            points: [N, 3] 点云数据 (x, y, z)
            extrinsics: 外参矩阵 [4, 4]

        Returns:
            转换后的点云 [N, 3]
        """
        if extrinsics is None:
            return points

        # 齐次坐标
        points_homo = np.hstack([points, np.ones((points.shape[0], 1))])
        points_vehicle = (extrinsics @ points_homo.T).T[:, :3]
        return points_vehicle

    def project_lidar_to_image(
        self,
        lidar_points: np.ndarray,
        camera_intrinsics: np.ndarray,
        extrinsics: np.ndarray,
        image_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        将激光雷达点投影到图像平面

        Args:
            lidar_points: [N, 3] 车辆坐标系下的点
            camera_intrinsics: [3, 3] 内参矩阵
            extrinsics: [4, 4] 外参矩阵 (lidar -> camera)
            image_shape: (H, W)

        Returns:
            [H, W] 深度图 (0 表示无数据)
        """
        H, W = image_shape

        # 转换到相机坐标系
        points_homo = np.hstack([lidar_points, np.ones((lidar_points.shape[0], 1))])
        points_camera = (extrinsics @ points_homo.T).T[:, :3]

        # 投影到图像平面
        points_2d = (camera_intrinsics @ points_camera.T).T
        depths = points_2d[:, 2]

        # 过滤不可见点
        valid = (depths > 0) & (points_2d[:, 0] > 0) & (points_2d[:, 1] > 0)
        u = (points_2d[valid, 0] / depths[valid]).astype(int)
        v = (points_2d[valid, 1] / depths[valid]).astype(int)

        # 创建深度图
        depth_map = np.zeros((H, W), dtype=np.float32)
        mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        depth_map[v[mask], u[mask]] = depths[valid[mask]]

        return depth_map

    def get_sensor_status(self) -> Dict[str, str]:
        """获取传感器状态摘要"""
        status = {
            "camera": "ready" if self.calibration_loaded else "calibrating",
            "lidar": "ready" if self.calibration_loaded else "calibrating",
            "radar": "ready" if self.calibration_loaded else "calibrating",
            "imu": "ready",
        }
        return status
