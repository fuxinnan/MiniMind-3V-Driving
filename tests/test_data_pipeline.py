import json

import pytest
import torch
from PIL import Image

from config.driving_config import DISCRETE_ACTIONS, DrivingConfig
from data.data_validator import DataValidator
from data.driving_dataset import (
    DrivingDataCollator,
    DrivingSFTDataset,
    DrivingSample,
)


def _record(tmp_path):
    images = {}
    for camera in ("front", "left", "right", "rear"):
        directory = tmp_path / camera
        directory.mkdir()
        images[camera] = []
        for frame in range(3):
            path = directory / f"{frame}.jpg"
            Image.new("RGB", (224, 224), (frame * 20, 30, 40)).save(path)
            images[camera].append(f"{camera}/{frame}.jpg")
    return {
        "schema_version": "1.0",
        "scene": "urban",
        "prompt": "分析场景",
        "response": "保持车道",
        "images": images,
        "timestamp": 1,
        "calibration": {"front": {"camera_intrinsic": []}},
        "ego_state": {"speed_kmh": 10},
        "sensors": {"lidar": [], "radar": {}, "gps_imu": None},
        "controls": {"steering": 0, "throttle": 0.2, "brake": 0, "gear": 2},
        "action": "keep_lane",
        "label_source": "synthetic",
    }


def test_config_and_sample_contract(tmp_path):
    config = DrivingConfig()
    assert tuple(config.discrete_actions) == DISCRETE_ACTIONS
    record = _record(tmp_path)
    assert DrivingSample.from_dict(record).action == "keep_lane"


def test_dataset_collator_and_validator(tmp_path):
    record = _record(tmp_path)
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    validation = DataValidator().validate_dataset(str(path), str(tmp_path))
    assert validation.invalid_items == 0
    dataset = DrivingSFTDataset(
        str(path), image_root=str(tmp_path), config=DrivingConfig()
    )
    item = dataset[0]
    assert item["pixel_values"].shape == (4, 3, 3, 224, 224)
    assert item["image_mask"].all()
    batch = DrivingDataCollator()([item, item])
    assert batch["pixel_values"].shape == (2, 4, 3, 3, 224, 224)
    assert batch["action_labels"].shape == (2,)


def test_schema_rejects_missing_camera(tmp_path):
    record = _record(tmp_path)
    del record["images"]["rear"]
    with pytest.raises(ValueError, match="missing camera"):
        DrivingSample.from_dict(record)
