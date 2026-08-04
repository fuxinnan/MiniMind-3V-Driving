import json

import pytest
import torch
from PIL import Image

from config.driving_config import DISCRETE_ACTIONS, DrivingConfig
from data.data_validator import DataValidator
from data.driving_dataset import (
    DrivingDataCollator,
    DrivingDPOCollator,
    DrivingDPODataset,
    DrivingRLAIFCollator,
    DrivingRLAIFDataset,
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


def _dpo_record(tmp_path):
    base = _record(tmp_path)
    return {
        "schema_version": "1.0",
        "scene": base["scene"],
        "prompt": base["prompt"],
        "images": base["images"],
        "timestamp": base["timestamp"],
        "calibration": base["calibration"],
        "ego_state": base["ego_state"],
        "sensors": base["sensors"],
        "label_source": "synthetic",
        "chosen": {
            "response": "保持车道",
            "controls": base["controls"],
            "action": "keep_lane",
        },
        "rejected": {
            "response": "加速冲过路口",
            "controls": {"steering": 0.0, "throttle": 0.9, "brake": 0.0, "gear": 3},
            "action": "accelerate",
        },
    }


def _rlaif_record(tmp_path):
    base = _record(tmp_path)
    base["safety_score"] = 0.8
    base["control_quality"] = 0.7
    base["reward"] = 0.76
    return base


def test_dpo_dataset_and_collator(tmp_path):
    record = _dpo_record(tmp_path)
    path = tmp_path / "dpo.jsonl"
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    dataset = DrivingDPODataset(
        str(path), image_root=str(tmp_path), config=DrivingConfig()
    )
    item = dataset[0]
    assert item["pixel_values"].shape == (4, 3, 3, 224, 224)
    assert item["chosen_control_mask"]
    assert item["rejected_action_mask"]
    batch = DrivingDPOCollator()([item, item])
    assert batch["chosen_input_ids"].shape[0] == 2
    assert batch["rejected_input_ids"].shape[0] == 2
    assert batch["chosen_action_labels"].shape == (2,)


def test_rlaif_dataset_and_collator(tmp_path):
    record = _rlaif_record(tmp_path)
    path = tmp_path / "rlaif.jsonl"
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    dataset = DrivingRLAIFDataset(
        str(path), image_root=str(tmp_path), config=DrivingConfig()
    )
    item = dataset[0]
    assert 0.0 <= float(item["reward"]) <= 1.0
    batch = DrivingRLAIFCollator()([item, item])
    assert batch["reward"].shape == (2,)
    assert batch["safety_score"].shape == (2,)


def test_dpo_schema_requires_pair_response(tmp_path):
    record = _dpo_record(tmp_path)
    del record["rejected"]["response"]
    path = tmp_path / "bad_dpo.jsonl"
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="rejected"):
        DrivingDPODataset(str(path), image_root=str(tmp_path), config=DrivingConfig())
