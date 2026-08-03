"""Schema, loaders, and collator for four-camera driving data."""

import glob
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from config.driving_config import (
    ACTION_TO_ID,
    CAMERA_NAMES,
    CONTINUOUS_CONTROL_KEYS,
    DATA_FORMAT_CONFIG,
    DrivingConfig,
)


@dataclass
class SensorReferences:
    """Optional sensor references or inline values."""

    lidar: List[Any] = field(default_factory=list)
    radar: Dict[str, List[Any]] = field(default_factory=dict)
    gps_imu: Optional[Any] = None


@dataclass
class DrivingSample:
    """Versioned JSON-serializable sample contract."""

    scene: str
    prompt: str
    response: str
    images: Dict[str, List[str]]
    timestamp: int
    calibration: Dict[str, Any]
    ego_state: Dict[str, Any]
    controls: Optional[Dict[str, float]]
    label_source: str
    sensors: SensorReferences = field(default_factory=SensorReferences)
    action: Optional[str] = None
    weather: Optional[str] = None
    time_of_day: Optional[str] = None
    speed: Optional[float] = None
    schema_version: str = DATA_FORMAT_CONFIG["schema_version"]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DrivingSample":
        missing = [
            key for key in DATA_FORMAT_CONFIG["sft_fields"]["required"]
            if key not in value
        ]
        if missing:
            raise ValueError(f"missing required fields: {missing}")
        images = value["images"]
        if not isinstance(images, dict):
            raise TypeError("images must be a camera-to-frame-list mapping")
        absent_cameras = [name for name in CAMERA_NAMES if name not in images]
        if absent_cameras:
            raise ValueError(f"missing camera streams: {absent_cameras}")
        for camera in CAMERA_NAMES:
            if not isinstance(images[camera], list) or not images[camera]:
                raise ValueError(f"images.{camera} must be a non-empty list")
        action = value.get("action")
        if action is not None and action not in ACTION_TO_ID:
            raise ValueError(f"unknown action: {action}")
        controls = value.get("controls")
        if controls is not None and not isinstance(controls, dict):
            raise TypeError("controls must be an object or null")
        sensor_value = value.get("sensors") or {}
        sensors = (
            sensor_value if isinstance(sensor_value, SensorReferences)
            else SensorReferences(
                lidar=list(sensor_value.get("lidar") or []),
                radar=dict(sensor_value.get("radar") or {}),
                gps_imu=sensor_value.get("gps_imu"),
            )
        )
        known = {item.name for item in cls.__dataclass_fields__.values()}
        kwargs = {key: value[key] for key in known if key in value}
        kwargs["sensors"] = sensors
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_driving_records(data_path: str) -> List[Dict[str, Any]]:
    path = Path(data_path)
    if path.is_dir():
        records: List[Dict[str, Any]] = []
        for filename in sorted(path.glob("*.jsonl")):
            records.extend(load_driving_records(str(filename)))
        return records
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if isinstance(value, dict) and "data" in value:
            value = value["data"]
        return value if isinstance(value, list) else [value]
    raise ValueError(f"Unsupported data format: {data_path}")


class _ImageLoader:
    def _init_image_loader(
        self,
        config: DrivingConfig,
        image_root: str,
        num_frames: Optional[int],
        transform,
        strict_images: bool,
    ) -> None:
        self.config = config
        self.image_root = Path(image_root)
        self.num_frames = num_frames or config.num_history_frames
        self.transform = transform
        self.strict_images = strict_images

    def _resolve_image(self, camera: str, reference: str) -> Path:
        path = Path(reference)
        candidates = [path] if path.is_absolute() else [
            self.image_root / path,
            self.image_root / camera / path,
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return candidates[0]

    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        if self.transform is not None:
            value = self.transform(image)
            if isinstance(value, dict):
                value = value.get("pixel_values")
            if not torch.is_tensor(value):
                raise TypeError("image transform must return a tensor")
            return value.squeeze(0) if value.ndim == 4 else value
        height, width = self.config.camera_input_size
        image = image.resize((width, height), Image.Resampling.BILINEAR)
        tensor = torch.tensor(
            bytearray(image.tobytes()), dtype=torch.uint8
        ).reshape(height, width, 3).permute(2, 0, 1).float() / 255.0
        mean = tensor.new_tensor([0.485, 0.456, 0.406])[:, None, None]
        std = tensor.new_tensor([0.229, 0.224, 0.225])[:, None, None]
        return (tensor - mean) / std

    def _load_images(
        self, item: Mapping[str, Any]
    ) -> Tuple[torch.Tensor, torch.BoolTensor]:
        image_streams = item.get("images") or {}
        loaded_cameras, camera_masks = [], []
        height, width = self.config.camera_input_size
        for camera in self.config.camera_names:
            refs = list(image_streams.get(camera) or [])
            if len(refs) < self.num_frames and self.strict_images:
                raise FileNotFoundError(
                    f"{camera} has {len(refs)} frames; expected {self.num_frames}"
                )
            refs = refs[-self.num_frames:]
            frames: List[torch.Tensor] = []
            mask: List[bool] = []
            for reference in refs:
                try:
                    path = self._resolve_image(camera, str(reference))
                    with Image.open(path) as image:
                        frames.append(self._image_to_tensor(image.convert("RGB")))
                    mask.append(True)
                except (OSError, ValueError, UnidentifiedImageError) as exc:
                    if self.strict_images:
                        raise RuntimeError(
                            f"Cannot load {camera} image {reference!r}"
                        ) from exc
                    frames.append(torch.zeros(3, height, width))
                    mask.append(False)
            while len(frames) < self.num_frames:
                if frames:
                    frames.insert(0, frames[0].clone())
                else:
                    frames.append(torch.zeros(3, height, width))
                mask.insert(0, False)
            loaded_cameras.append(torch.stack(frames))
            camera_masks.append(torch.tensor(mask, dtype=torch.bool))
        return torch.stack(loaded_cameras), torch.stack(camera_masks)


class DrivingSFTDataset(_ImageLoader, Dataset):
    """Validated SFT dataset returning images shaped ``[4,T,3,H,W]``."""

    def __init__(
        self,
        data_path: str,
        config: Optional[DrivingConfig] = None,
        image_root: str = "./dataset/driving/raw/camera",
        tokenizer=None,
        max_seq_len: Optional[int] = None,
        num_frames: Optional[int] = None,
        transform=None,
        strict_images: bool = True,
        validate_schema: bool = True,
    ):
        self.config = config or DrivingConfig()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len or self.config.max_seq_len
        self._init_image_loader(
            self.config, image_root, num_frames, transform, strict_images
        )
        self.data = load_driving_records(data_path)
        if validate_schema:
            for index, item in enumerate(self.data):
                try:
                    DrivingSample.from_dict(item)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid driving sample {index}: {exc}") from exc

    def __len__(self) -> int:
        return len(self.data)

    def _tokenize(self, prompt: str, response: str) -> Dict[str, torch.Tensor]:
        if self.tokenizer is None:
            return {
                "input_ids": torch.tensor([1], dtype=torch.long),
                "attention_mask": torch.tensor([1], dtype=torch.long),
            }
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        encoded = self.tokenizer(
            text, truncation=True, max_length=self.max_seq_len,
            return_tensors="pt",
        )
        return {key: value.squeeze(0) for key, value in encoded.items()}

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.data[index]
        text = self._tokenize(item.get("prompt", ""), item.get("response", ""))
        pixels, image_mask = self._load_images(item)
        controls = item.get("controls")
        control_labels = None
        if controls is not None:
            if not all(key in controls for key in CONTINUOUS_CONTROL_KEYS):
                missing = [
                    key for key in CONTINUOUS_CONTROL_KEYS if key not in controls
                ]
                raise ValueError(f"Sample {index} controls missing {missing}")
            control_labels = torch.tensor(
                [controls[key] for key in CONTINUOUS_CONTROL_KEYS],
                dtype=torch.float32,
            )
        action = item.get("action")
        action_labels = (
            torch.tensor(ACTION_TO_ID[action], dtype=torch.long)
            if action is not None else None
        )
        sensors = self._load_sensors(item.get("sensors") or {})
        return {
            **text,
            "pixel_values": pixels,
            "image_mask": image_mask,
            **sensors,
            "control_labels": control_labels,
            "control_label_mask": torch.tensor(
                control_labels is not None, dtype=torch.bool
            ),
            "action_labels": action_labels,
            "action_label_mask": torch.tensor(
                action_labels is not None, dtype=torch.bool
            ),
            "scene": item.get("scene", "unknown"),
            "metadata": {
                "timestamp": item.get("timestamp"),
                "calibration": item.get("calibration"),
                "ego_state": item.get("ego_state"),
                "label_source": item.get("label_source"),
                "target_response": item.get("response", ""),
                **(item.get("metadata") or {}),
            },
        }

    def _load_sensors(self, sensors: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        lidar, lidar_mask = self._load_point_sensor(
            sensors.get("lidar"), self.config.lidar_num_points,
            self.config.lidar_point_dims, "lidar",
        )
        radar_value = sensors.get("radar")
        radar_refs = []
        if isinstance(radar_value, dict):
            for refs in radar_value.values():
                radar_refs.extend(refs or [])
        else:
            radar_refs = radar_value or []
        radar, radar_mask = self._load_point_sensor(
            radar_refs, self.config.radar_num_detections,
            self.config.radar_point_dims, "radar",
        )
        gps_value = sensors.get("gps_imu")
        gps = torch.zeros(self.config.gps_imu_dims, dtype=torch.float32)
        gps_mask = torch.tensor(False)
        if isinstance(gps_value, dict):
            gps_value = gps_value.get("values")
        if gps_value is not None:
            values = torch.as_tensor(gps_value, dtype=torch.float32).flatten()
            count = min(values.numel(), gps.numel())
            gps[:count] = values[:count]
            gps_mask = torch.tensor(count == gps.numel())
        return {
            "lidar_pointcloud": lidar,
            "lidar_mask": lidar_mask,
            "radar_data": radar,
            "radar_mask": radar_mask,
            "gps_imu": gps,
            "gps_imu_mask": gps_mask,
        }

    def _load_point_sensor(
        self, references: Any, max_points: int, dimensions: int, kind: str
    ) -> Tuple[torch.Tensor, torch.BoolTensor]:
        output = torch.zeros(max_points, dimensions, dtype=torch.float32)
        mask = torch.zeros(max_points, dtype=torch.bool)
        refs = references if isinstance(references, list) else []
        chunks: List[np.ndarray] = []
        for reference in refs:
            reference = reference.get("path") if isinstance(reference, dict) else reference
            if not reference:
                continue
            path = Path(reference)
            if not path.is_absolute():
                path = self.image_root / path
            try:
                if path.suffix == ".npy":
                    array = np.load(path)
                elif path.suffix == ".pcd" and kind == "radar":
                    from nuscenes.utils.data_classes import RadarPointCloud
                    array = RadarPointCloud.from_file(str(path)).points.T
                else:
                    array = np.fromfile(path, dtype=np.float32)
                    array = array.reshape(-1, dimensions)
                chunks.append(np.asarray(array, dtype=np.float32))
            except (OSError, ValueError, ImportError):
                continue
        if chunks:
            values = np.concatenate(chunks, axis=0)
            count = min(len(values), max_points)
            width = min(values.shape[1], dimensions)
            output[:count, :width] = torch.from_numpy(values[:count, :width])
            mask[:count] = True
        return output, mask


class DrivingDataCollator:
    """Pad text and preserve missing labels through explicit masks."""

    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        batch: Dict[str, Any] = {
            "input_ids": pad_sequence(
                [item["input_ids"] for item in features], batch_first=True,
                padding_value=self.pad_token_id,
            ),
            "attention_mask": pad_sequence(
                [item["attention_mask"] for item in features], batch_first=True,
                padding_value=0,
            ),
        }
        tensor_keys = [
            "pixel_values", "image_mask", "lidar_pointcloud", "lidar_mask",
            "radar_data", "radar_mask", "gps_imu", "gps_imu_mask",
            "control_label_mask", "action_label_mask",
        ]
        for key in tensor_keys:
            batch[key] = torch.stack([item[key] for item in features])
        control_mask = batch["control_label_mask"]
        action_mask = batch["action_label_mask"]
        batch["control_labels"] = (
            torch.stack([
                item["control_labels"]
                if item["control_labels"] is not None else torch.zeros(4)
                for item in features
            ]) if bool(control_mask.any()) else None
        )
        batch["action_labels"] = (
            torch.stack([
                item["action_labels"]
                if item["action_labels"] is not None
                else torch.tensor(-100, dtype=torch.long)
                for item in features
            ]) if bool(action_mask.any()) else None
        )
        batch["scene"] = [item["scene"] for item in features]
        batch["metadata"] = [item["metadata"] for item in features]
        return batch


def driving_collate_fn(features: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return DrivingDataCollator()(features)


class _PreferenceImageDataset(_ImageLoader, Dataset):
    def _common_init(
        self, data_path, config, image_root, tokenizer, max_seq_len, num_frames
    ):
        self.config = config or DrivingConfig()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.data = load_driving_records(data_path)
        self._init_image_loader(
            self.config, image_root, num_frames, None, strict_images=False
        )

    def __len__(self):
        return len(self.data)

    def _conversation(self, messages):
        if self.tokenizer is None:
            return {
                "input_ids": torch.tensor([1]),
                "attention_mask": torch.tensor([1]),
            }
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        values = self.tokenizer(
            text, truncation=True, max_length=self.max_seq_len,
            return_tensors="pt",
        )
        return {key: value.squeeze(0) for key, value in values.items()}


class DrivingDPODataset(_PreferenceImageDataset):
    """Legacy-compatible preference dataset using the shared image loader."""

    def __init__(
        self, data_path, config=None,
        image_root="./dataset/driving/raw/camera", tokenizer=None,
        max_seq_len=2048, num_frames=3,
    ):
        self._common_init(
            data_path, config, image_root, tokenizer, max_seq_len, num_frames
        )

    def __getitem__(self, index):
        item = self.data[index]
        chosen = self._conversation([
            {"role": "user", "content": item.get("prompt", "")},
            {"role": "assistant", "content": item["chosen"]["response"]},
        ])
        rejected = self._conversation([
            {"role": "user", "content": item.get("prompt", "")},
            {"role": "assistant", "content": item["rejected"]["response"]},
        ])
        pixels, _ = self._load_images(item)
        return {
            "chosen_input_ids": chosen["input_ids"],
            "chosen_attention_mask": chosen["attention_mask"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_attention_mask": rejected["attention_mask"],
            "pixel_values": pixels,
            "scene": item.get("scene", "unknown"),
        }


class DrivingRLAIFDataset(_PreferenceImageDataset):
    """Legacy-compatible AI-feedback dataset."""

    def __init__(
        self, data_path, config=None,
        image_root="./dataset/driving/raw/camera", tokenizer=None,
        max_seq_len=2048, num_frames=3,
    ):
        self._common_init(
            data_path, config, image_root, tokenizer, max_seq_len, num_frames
        )

    def __getitem__(self, index):
        item = self.data[index]
        values = self._conversation(item.get("conversations", []))
        pixels, _ = self._load_images(item)
        return {
            **values,
            "pixel_values": pixels,
            "scene": item.get("scene", "unknown"),
            "safety_score": item.get("safety_score", 0.0),
            "control_quality": item.get("control_quality", 0.0),
        }


class NuScenesDataset(DrivingSFTDataset):
    """Loads JSONL emitted by ``scripts/prepare_driving_data.py``."""
