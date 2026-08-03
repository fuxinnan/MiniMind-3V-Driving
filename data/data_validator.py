"""Validation for the canonical driving sample schema."""

import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

from config.driving_config import (
    CAMERA_NAMES,
    CONTROL_RANGES,
    DATA_FORMAT_CONFIG,
    DISCRETE_ACTIONS,
    SCENE_CONFIG,
)
from .driving_dataset import DrivingSample, load_driving_records


@dataclass
class ValidationResult:
    total_items: int = 0
    valid_items: int = 0
    invalid_items: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, item_idx: int, field_name: str, message: str):
        self.errors.append(
            {"item_idx": item_idx, "field": field_name, "message": message}
        )

    def add_warning(self, item_idx: int, field_name: str, message: str):
        self.warnings.append(
            {"item_idx": item_idx, "field": field_name, "message": message}
        )

    @property
    def validity_rate(self) -> float:
        return self.valid_items / self.total_items if self.total_items else 0.0

    def summary(self) -> str:
        return (
            "=== 数据校验报告 ===\n"
            f"总样本数: {self.total_items}\n"
            f"有效样本: {self.valid_items} ({self.validity_rate:.1%})\n"
            f"无效样本: {self.invalid_items}\n"
            f"错误数: {len(self.errors)}\n"
            f"警告数: {len(self.warnings)}\n"
            "===================="
        )


class DataValidator:
    def __init__(
        self,
        strict_mode: bool = False,
        min_image_size: Tuple[int, int] = (224, 224),
        max_image_size: Tuple[int, int] = (4096, 4096),
        require_all_cameras: bool = True,
        num_frames: int = 3,
    ):
        self.strict_mode = strict_mode
        self.min_image_size = min_image_size
        self.max_image_size = max_image_size
        self.require_all_cameras = require_all_cameras
        self.num_frames = num_frames

    def validate_dataset(
        self,
        data_path: str,
        camera_root: str = "./dataset/driving/raw/camera",
    ) -> ValidationResult:
        data = load_driving_records(data_path)
        result = ValidationResult(total_items=len(data))
        scenes = defaultdict(int)
        labels = defaultdict(int)
        control_stats = {key: [] for key in ("steering", "throttle", "brake")}
        previous_timestamp: Dict[str, int] = {}

        for index, item in enumerate(data):
            item_result = self._validate_item(index, item, camera_root)
            result.errors.extend(item_result.errors)
            result.warnings.extend(item_result.warnings)
            if item_result.errors:
                result.invalid_items += 1
            else:
                result.valid_items += 1
            scene = item.get("scene", "unknown")
            scenes[scene] += 1
            labels[str(item.get("label_source", "missing"))] += 1
            controls = item.get("controls") or {}
            for key in control_stats:
                if isinstance(controls.get(key), (int, float)):
                    control_stats[key].append(controls[key])
            sequence_id = str((item.get("metadata") or {}).get("scene_token", scene))
            timestamp = item.get("timestamp")
            if isinstance(timestamp, (int, float)):
                old = previous_timestamp.get(sequence_id)
                if old is not None and timestamp < old:
                    result.add_warning(
                        index, "timestamp",
                        f"timestamp decreased within sequence {sequence_id}",
                    )
                previous_timestamp[sequence_id] = timestamp

        result.stats = {
            "scene_distribution": dict(scenes),
            "label_source_distribution": dict(labels),
            "control_statistics": {
                key: {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "count": len(values),
                }
                for key, values in control_stats.items() if values
            },
        }
        return result

    def _validate_item(
        self, index: int, item: Dict[str, Any], camera_root: str
    ) -> ValidationResult:
        result = ValidationResult(total_items=1)
        try:
            DrivingSample.from_dict(item)
        except (TypeError, ValueError) as exc:
            result.add_error(index, "schema", str(exc))
            return result

        if item["scene"] not in SCENE_CONFIG["scene_categories"]:
            result.add_warning(index, "scene", f"unknown scene: {item['scene']}")
        if not isinstance(item["timestamp"], (int, float)):
            result.add_error(index, "timestamp", "must be numeric")
        if not isinstance(item["calibration"], dict) or not item["calibration"]:
            result.add_error(index, "calibration", "must be a non-empty object")
        if not isinstance(item["ego_state"], dict) or not item["ego_state"]:
            result.add_error(index, "ego_state", "must be a non-empty object")
        if item["label_source"] not in DATA_FORMAT_CONFIG["control_format"]["label_sources"]:
            result.add_error(
                index, "label_source",
                f"unsupported source: {item['label_source']}",
            )

        images = item["images"]
        cameras = CAMERA_NAMES if self.require_all_cameras else tuple(images)
        for camera in cameras:
            references = images.get(camera) or []
            if len(references) < self.num_frames:
                result.add_error(
                    index, f"images.{camera}",
                    f"expected at least {self.num_frames} frames, got {len(references)}",
                )
            for reference in references[-self.num_frames:]:
                path = self._resolve_path(camera_root, camera, reference)
                if path is None:
                    result.add_error(
                        index, f"images.{camera}", f"image not found: {reference}"
                    )
                    continue
                try:
                    with Image.open(path) as image:
                        image.verify()
                    with Image.open(path) as image:
                        width, height = image.size
                    if (
                        width < self.min_image_size[0]
                        or height < self.min_image_size[1]
                    ):
                        result.add_warning(
                            index, f"images.{camera}",
                            f"image too small: {width}x{height}",
                        )
                    if (
                        width > self.max_image_size[0]
                        or height > self.max_image_size[1]
                    ):
                        result.add_warning(
                            index, f"images.{camera}",
                            f"image too large: {width}x{height}",
                        )
                except OSError as exc:
                    result.add_error(
                        index, f"images.{camera}",
                        f"cannot decode {reference}: {exc}",
                    )

        controls = item.get("controls")
        if controls is not None:
            for key, (lower, upper) in CONTROL_RANGES.items():
                if key not in controls:
                    result.add_error(index, f"controls.{key}", "missing value")
                    continue
                value = controls[key]
                if not isinstance(value, (int, float)) or not np.isfinite(value):
                    result.add_error(
                        index, f"controls.{key}", "must be a finite number"
                    )
                elif not lower <= value <= upper:
                    result.add_error(
                        index, f"controls.{key}",
                        f"{value} outside [{lower}, {upper}]",
                    )
        if item.get("action") is not None and item["action"] not in DISCRETE_ACTIONS:
            result.add_error(index, "action", f"unknown action: {item['action']}")
        if controls is None and item.get("action") is None:
            result.add_warning(index, "labels", "sample has no control/action label")

        if self.strict_mode:
            if len(item.get("prompt", "")) < 10:
                result.add_warning(index, "prompt", "prompt is very short")
            if len(item.get("response", "")) < 10:
                result.add_warning(index, "response", "response is very short")
        return result

    @staticmethod
    def _resolve_path(root: str, camera: str, reference: Any):
        if not isinstance(reference, str):
            return None
        path = Path(reference)
        candidates = [path] if path.is_absolute() else [
            Path(root) / path,
            Path(root) / camera / path,
        ]
        return next((candidate for candidate in candidates if candidate.is_file()), None)

    def validate_control_consistency(self, data_path: str) -> Dict[str, float]:
        data = load_driving_records(data_path)
        violations = 0
        labelled = 0
        for item in data:
            controls = item.get("controls")
            if not controls:
                continue
            labelled += 1
            throttle = controls.get("throttle", 0.0)
            brake = controls.get("brake", 0.0)
            steering = controls.get("steering", 0.0)
            speed = (item.get("ego_state") or {}).get(
                "speed_kmh", item.get("speed", 0.0)
            )
            if throttle > 0.1 and brake > 0.1:
                violations += 1
            if speed > 100 and abs(steering) > 0.8:
                violations += 1
        return {
            "total": labelled,
            "violations": violations,
            "violation_rate": violations / labelled if labelled else 0.0,
        }

    def get_scene_coverage(self, data_path: str) -> Dict[str, int]:
        counts = defaultdict(int)
        for item in load_driving_records(data_path):
            counts[item.get("scene", "unknown")] += 1
        return dict(counts)

    def get_data_summary(
        self,
        data_path: str,
        camera_root: str = "./dataset/driving/raw/camera",
    ) -> str:
        return self.validate_dataset(data_path, camera_root).summary()
