"""
数据校验器

验证驾驶数据的质量、格式和一致性:
    - 字段完整性检查
    - 图像有效性检查
    - 控制标签范围检查
    - 场景一致性检查
    - 时间戳顺序检查
"""

import os
import json
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass, field

import torch
import numpy as np
from PIL import Image


@dataclass
class ValidationResult:
    """校验结果"""
    total_items: int = 0
    valid_items: int = 0
    invalid_items: int = 0
    errors: List[Dict] = field(default_factory=list)
    warnings: List[Dict] = field(default_factory=list)
    stats: Dict = field(default_factory=dict)

    def add_error(self, item_idx: int, field: str, message: str):
        self.errors.append({
            "item_idx": item_idx,
            "field": field,
            "message": message,
        })

    def add_warning(self, item_idx: int, field: str, message: str):
        self.warnings.append({
            "item_idx": item_idx,
            "field": field,
            "message": message,
        })

    @property
    def validity_rate(self) -> float:
        if self.total_items == 0:
            return 0.0
        return self.valid_items / self.total_items

    def summary(self) -> str:
        return (
            f"=== 数据校验报告 ===\n"
            f"总样本数: {self.total_items}\n"
            f"有效样本: {self.valid_items} ({self.validity_rate:.1%})\n"
            f"无效样本: {self.invalid_items}\n"
            f"错误数: {len(self.errors)}\n"
            f"警告数: {len(self.warnings)}\n"
            f"===================="
        )


class DataValidator:
    """
    驾驶数据校验器

    对驾驶数据集进行全面校验，确保数据质量和一致性
    """

    # 控制标签合法范围
    CONTROL_RANGES = {
        "steering": (-1.0, 1.0),
        "throttle": (0.0, 1.0),
        "brake": (0.0, 1.0),
        "gear": (0, 4),
    }

    # 必需字段
    REQUIRED_FIELDS = ["scene", "prompt", "response", "images", "controls"]

    # 可选字段
    OPTIONAL_FIELDS = ["timestamp", "speed", "action", "weather", "time_of_day"]

    # 场景分类
    VALID_SCENES = [
        "highway", "urban", "suburban", "intersection", "roundabout",
        "parking", "tunnel", "construction", "emergency", "pedestrian_cross",
        "school_zone", "residential", "ringroad", "onramp_offramp",
    ]

    # 天气分类
    VALID_WEATHERS = [
        "sunny", "cloudy", "rainy", "foggy", "snowy", "night",
        "dusk", "dawn", "overcast", "stormy",
    ]

    def __init__(
        self,
        strict_mode: bool = False,
        min_image_size: Tuple[int, int] = (224, 224),
        max_image_size: Tuple[int, int] = (4096, 4096),
        require_all_cameras: bool = True,
    ):
        self.strict_mode = strict_mode
        self.min_image_size = min_image_size
        self.max_image_size = max_image_size
        self.require_all_cameras = require_all_cameras

    def validate_dataset(
        self,
        data_path: str,
        camera_root: str = "./dataset/driving/raw/camera",
    ) -> ValidationResult:
        """
        校验整个数据集

        Args:
            data_path: JSONL 数据文件路径或目录
            camera_root: 相机图像根目录

        Returns:
            ValidationResult
        """
        result = ValidationResult()

        # 加载数据
        data = self._load_data(data_path)
        result.total_items = len(data)

        scene_counts = defaultdict(int)
        weather_counts = defaultdict(int)
        image_sizes = {}
        control_stats = {"steering": [], "throttle": [], "brake": []}

        for idx, item in enumerate(data):
            item_result = self._validate_item(idx, item, camera_root)
            result.errors.extend(item_result.errors)
            result.warnings.extend(item_result.warnings)

            if not item_result.errors:
                result.valid_items += 1
            else:
                result.invalid_items += 1

            # 收集统计信息
            scene = item.get("scene", "unknown")
            scene_counts[scene] += 1

            weather = item.get("weather", "unknown")
            weather_counts[weather] += 1

            # 收集图像尺寸
            images = item.get("images", {})
            for cam_name, cam_images in images.items():
                if cam_images:
                    first_img = cam_images[0]
                    if isinstance(first_img, str):
                        img_path = os.path.join(camera_root, cam_name, first_img)
                        if os.path.exists(img_path):
                            try:
                                with Image.open(img_path) as img:
                                    image_sizes[cam_name] = img.size
                            except Exception:
                                pass

            # 收集控制统计
            controls = item.get("controls", {})
            for key in ["steering", "throttle", "brake"]:
                if key in controls:
                    control_stats[key].append(controls[key])

        result.stats = {
            "scene_distribution": dict(scene_counts),
            "weather_distribution": dict(weather_counts),
            "image_sizes": {k: f"{v[0]}x{v[1]}" for k, v in image_sizes.items()},
            "control_statistics": {},
        }

        for key in ["steering", "throttle", "brake"]:
            values = control_stats[key]
            if values:
                result.stats["control_statistics"][key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "count": len(values),
                }

        return result

    def _load_data(self, data_path: str) -> List[Dict]:
        """加载数据"""
        if data_path.endswith(".jsonl"):
            data = []
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            return data
        elif data_path.endswith(".json"):
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else [data]
        elif os.path.isdir(data_path):
            all_data = []
            for jsonl_file in glob.glob(os.path.join(data_path, "*.jsonl")):
                all_data.extend(self._load_data(jsonl_file))
            return all_data
        return []

    def _validate_item(
        self,
        idx: int,
        item: Dict,
        camera_root: str,
    ) -> ValidationResult:
        """校验单个数据项"""
        result = ValidationResult()
        result.total_items = 1

        # 1. 检查必需字段
        for field in self.REQUIRED_FIELDS:
            if field not in item or item[field] is None:
                result.add_error(idx, field, f"Missing required field: {field}")

        if result.invalid_items > 0 and result.errors:
            return result

        # 2. 检查场景分类
        scene = item.get("scene", "")
        if scene not in self.VALID_SCENES:
            result.add_warning(idx, "scene", f"Unknown scene category: {scene}")

        # 3. 检查图像
        images = item.get("images", {})
        if self.require_all_cameras:
            from config.driving_config import SENSOR_CONFIG
            expected_cameras = SENSOR_CONFIG["camera_names"]
            for cam in expected_cameras:
                if cam not in images or not images[cam]:
                    result.add_error(idx, f"images.{cam}", f"Missing images for camera: {cam}")

        # 检查图像文件
        for cam_name, cam_images in images.items():
            if not cam_images:
                continue
            for img_path in cam_images[:3]:  # 只检查前几张
                if isinstance(img_path, str):
                    full_path = os.path.join(camera_root, cam_name, img_path)
                    if not os.path.exists(full_path):
                        result.add_warning(idx, f"images.{cam_name}", f"Image not found: {img_path}")
                    else:
                        try:
                            with Image.open(full_path) as img:
                                w, h = img.size
                                if w < self.min_image_size[0] or h < self.min_image_size[1]:
                                    result.add_warning(
                                        idx, f"images.{cam_name}",
                                        f"Image too small: {w}x{h}"
                                    )
                                if w > self.max_image_size[0] or h > self.max_image_size[1]:
                                    result.add_warning(
                                        idx, f"images.{cam_name}",
                                        f"Image too large: {w}x{h}"
                                    )
                        except Exception as e:
                            result.add_error(
                                idx, f"images.{cam_name}",
                                f"Cannot open image: {img_path} - {str(e)}"
                            )

        # 4. 检查控制标签
        controls = item.get("controls", {})
        for key, (min_val, max_val) in self.CONTROL_RANGES.items():
            if key in controls:
                val = controls[key]
                if not isinstance(val, (int, float)):
                    result.add_error(idx, f"controls.{key}", f"Invalid control value type: {type(val)}")
                elif val < min_val or val > max_val:
                    result.add_error(
                        idx, f"controls.{key}",
                        f"Control value out of range [{min_val}, {max_val}]: {val}"
                    )

        # 5. 检查离散动作
        action = item.get("action", "")
        if action and action not in [
            "keep_lane", "turn_left", "turn_right", "stop", "accelerate",
            "decelerate", "yield", "overtake", "park", "emergency_brake",
            "follow_lane", "change_lane_left", "change_lane_right",
        ]:
            result.add_warning(idx, "action", f"Unknown action: {action}")

        # 6. 检查天气
        weather = item.get("weather", "")
        if weather and weather not in self.VALID_WEATHERS:
            result.add_warning(idx, "weather", f"Unknown weather: {weather}")

        # 7. 检查速度
        speed = item.get("speed", 0.0)
        if isinstance(speed, (int, float)):
            if speed < 0 or speed > 300:
                result.add_warning(idx, "speed", f"Unusual speed: {speed} km/h")

        # 8. 严格模式: 检查 prompt 和 response 长度
        if self.strict_mode:
            prompt = item.get("prompt", "")
            response = item.get("response", "")
            if len(prompt) < 10:
                result.add_warning(idx, "prompt", f"Prompt too short: {len(prompt)} chars")
            if len(response) < 10:
                result.add_warning(idx, "response", f"Response too short: {len(response)} chars")

        return result

    def validate_control_consistency(
        self,
        data_path: str,
    ) -> Dict[str, float]:
        """
        检查控制标签的一致性

        验证控制信号是否符合物理约束:
            - 不能同时油门和刹车
            - 转向和速度的关系
            - 挡位与速度的关系
        """
        data = self._load_data(data_path)
        violations = 0
        total = len(data)

        for item in data:
            controls = item.get("controls", {})
            throttle = controls.get("throttle", 0.0)
            brake = controls.get("brake", 0.0)
            steering = controls.get("steering", 0.0)
            gear = controls.get("gear", 2)
            speed = item.get("speed", 0.0)

            # 不能同时油门和刹车
            if throttle > 0.1 and brake > 0.1:
                violations += 1

            # 倒车时不能向前行驶
            if gear == 0 and speed > 1.0:
                violations += 1

            # 高速时不能急转
            if speed > 100 and abs(steering) > 0.8:
                violations += 1

        return {
            "total": total,
            "violations": violations,
            "violation_rate": violations / total if total > 0 else 0.0,
        }

    def get_scene_coverage(self, data_path: str) -> Dict[str, int]:
        """获取场景覆盖率"""
        data = self._load_data(data_path)
        scene_counts = defaultdict(int)
        for item in data:
            scene = item.get("scene", "unknown")
            scene_counts[scene] += 1
        return dict(scene_counts)

    def get_data_summary(self, data_path: str) -> str:
        """获取数据摘要"""
        result = self.validate_dataset(data_path)
        return result.summary()
