"""
驾驶数据准备脚本

将原始驾驶数据转换为训练可用的 JSONL 格式:
    - 从 nuScenes/Waymo 等格式转换
    - 图像预处理和resize
    - 控制标签归一化
    - 数据清洗和校验
    - 训练/验证/测试集划分
"""

import os
import json
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import numpy as np
from tqdm import tqdm


def prepare_nuscenes_data(
    nuscenes_root: str,
    output_dir: str,
    split_ratios: Dict[str, float] = None,
    num_frames: int = 3,
    camera_names: List[str] = None,
    version: Optional[str] = None,
):
    """
    从 nuScenes 数据集准备训练数据

    Args:
        nuscenes_root: nuScenes 根目录
        output_dir: 输出目录
        split_ratios: 训练/验证/测试比例
        num_frames: 每场景使用的帧数
        camera_names: 使用的相机列表
    """
    from data.nuscenes_adapter import NuScenesAdapter
    split_ratios = split_ratios or {"train": 0.8, "val": 0.1, "test": 0.1}
    adapter = NuScenesAdapter(
        nuscenes_root, version=version, num_frames=num_frames
    )
    records = adapter.records()
    # Split by scene, not individual frames, to prevent temporal leakage.
    scene_tokens = sorted({
        item["metadata"]["scene_token"] for item in records
    })
    rng = np.random.default_rng(42)
    rng.shuffle(scene_tokens)
    train_end = int(len(scene_tokens) * split_ratios["train"])
    val_end = train_end + int(len(scene_tokens) * split_ratios["val"])
    scene_split = {
        token: (
            "train" if index < train_end
            else "val" if index < val_end else "test"
        )
        for index, token in enumerate(scene_tokens)
    }
    split_data = {"train": [], "val": [], "test": []}
    for item in records:
        split_data[scene_split[item["metadata"]["scene_token"]]].append(item)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for split, data in split_data.items():
        split_path = output_path / f"{split}.jsonl"
        with open(split_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  {split}: {len(data)} samples -> {split_path}")

    print(f"nuScenes conversion complete ({adapter.version})")


def classify_nuscenes_scene(scene_name: str) -> str:
    """将 nuScenes 场景名称分类为标准场景"""
    name_lower = scene_name.lower()

    if "highway" in name_lower or "motorway" in name_lower:
        return "highway"
    elif "intersection" in name_lower:
        return "intersection"
    elif "parking" in name_lower:
        return "parking"
    elif "residential" in name_lower:
        return "residential"
    elif "urban" in name_lower or "city" in name_lower:
        return "urban"
    elif "tunnel" in name_lower:
        return "tunnel"
    elif "construction" in name_lower:
        return "construction"
    elif "roundabout" in name_lower:
        return "roundabout"
    else:
        return "suburban"


def prepare_custom_data(
    raw_data_dir: str,
    output_dir: str,
    scene_dir: str = None,
    control_labels_dir: str = None,
):
    """
    准备自定义驾驶数据

    Args:
        raw_data_dir: 原始数据目录
        output_dir: 输出目录
        scene_dir: 场景分类目录
        control_labels_dir: 控制标签目录
    """
    print(f"Preparing custom data from {raw_data_dir}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 扫描原始数据
    data_items = []
    for jsonl_file in Path(raw_data_dir).glob("*.jsonl"):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data_items.append(json.loads(line))

    print(f"Found {len(data_items)} raw data items")

    # 划分训练/验证/测试集
    np.random.seed(42)
    indices = np.random.permutation(len(data_items))
    n_train = int(len(indices) * 0.8)
    n_val = int(len(indices) * 0.1)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    for split, indices_list in [
        ("train", train_indices),
        ("val", val_indices),
        ("test", test_indices),
    ]:
        split_data = [data_items[i] for i in indices_list]
        split_path = output_path / f"{split}.jsonl"

        with open(split_path, "w", encoding="utf-8") as f:
            for item in split_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"  {split}: {len(split_data)} samples -> {split_path}")

    print(f"Preparation complete!")


def generate_synthetic_data(
    output_dir: str,
    n_samples: int = 1000,
    scene_distribution: Dict[str, float] = None,
):
    """
    生成合成驾驶数据 (用于开发和测试)

    Args:
        output_dir: 输出目录
        n_samples: 样本数
        scene_distribution: 场景分布比例
    """
    scene_distribution = scene_distribution or {
        "highway": 0.3,
        "urban": 0.3,
        "intersection": 0.15,
        "parking": 0.1,
        "emergency": 0.05,
        "tunnel": 0.05,
        "construction": 0.05,
    }

    action_map = {
        "highway": ["keep_lane", "accelerate", "change_lane_left", "change_lane_right"],
        "urban": ["keep_lane", "turn_left", "turn_right", "stop", "decelerate"],
        "intersection": ["keep_lane", "turn_left", "turn_right", "stop", "yield"],
        "parking": ["stop", "turn_left", "turn_right", "decelerate"],
        "emergency": ["emergency_brake", "stop", "turn_left", "turn_right"],
        "tunnel": ["keep_lane", "decelerate", "accelerate"],
        "construction": ["decelerate", "stop", "keep_lane", "change_lane_left"],
    }

    weather_options = ["sunny", "cloudy", "rainy", "foggy", "night"]
    time_options = ["day", "dusk", "dawn", "night"]

    print(f"Generating {n_samples} synthetic samples")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    image_root = output_path / "synthetic_images"
    camera_names = ["front", "left", "right", "rear"]
    from PIL import Image, ImageDraw
    for camera_index, camera in enumerate(camera_names):
        camera_dir = image_root / camera
        camera_dir.mkdir(parents=True, exist_ok=True)
        for frame in range(3):
            image_path = camera_dir / f"frame_{frame}.jpg"
            image = Image.new("RGB", (224, 224), (30 + 20 * camera_index, 40, 60))
            ImageDraw.Draw(image).text((8, 8), f"{camera} t-{2-frame}", fill="white")
            image.save(image_path)

    data = []
    for i in tqdm(range(n_samples), desc="Generating"):
        # 选择场景
        scenes = list(scene_distribution.keys())
        weights = list(scene_distribution.values())
        scene = np.random.choice(scenes, p=weights)

        # 选择动作
        actions = action_map.get(scene, ["keep_lane"])
        action = np.random.choice(actions)

        # 生成控制信号
        controls = generate_controls_for_action(action)

        # 生成场景描述
        prompt = generate_scene_prompt(scene, controls)

        # 生成响应
        response = generate_response(action, controls, scene)

        data.append({
            "schema_version": "1.0",
            "scene": scene,
            "prompt": prompt,
            "response": response,
            "controls": controls,
            "images": {
                camera: [
                    str(Path("synthetic_images") / camera / f"frame_{frame}.jpg")
                    for frame in range(3)
                ]
                for camera in camera_names
            },
            "calibration": {
                camera: {"camera_intrinsic": [], "translation": [0, 0, 0],
                         "rotation": [1, 0, 0, 0]}
                for camera in camera_names
            },
            "ego_state": {"speed_kmh": float(np.random.uniform(0, 120))},
            "sensors": {"lidar": [], "radar": {}, "gps_imu": None},
            "label_source": "synthetic",
            "action": action,
            "weather": np.random.choice(weather_options),
            "time_of_day": np.random.choice(time_options),
            "speed": float(np.random.uniform(0, 120)),
            "timestamp": int(datetime.now().timestamp()),
        })

    # 保存为互不重叠的确定性切分。
    train_end = int(n_samples * 0.8)
    val_end = train_end + int(n_samples * 0.1)
    for split_name, split_data in (
        ("train", data[:train_end]),
        ("val", data[train_end:val_end]),
        ("test", data[val_end:]),
    ):
        split_path = output_path / f"{split_name}.jsonl"
        with open(split_path, "w", encoding="utf-8") as f:
            for item in split_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"  {split_name}: {len(split_data)} samples -> {split_path}")

    print(f"Synthetic data generation complete!")
    generate_preference_splits(output_path, data)


def _unsafe_action(action: str, scene: str) -> str:
    """Pick a contrasting / less-safe action for preference pairs."""
    fallback = {
        "keep_lane": "emergency_brake",
        "accelerate": "emergency_brake",
        "decelerate": "accelerate",
        "stop": "accelerate",
        "yield": "accelerate",
        "turn_left": "turn_right",
        "turn_right": "turn_left",
        "change_lane_left": "change_lane_right",
        "change_lane_right": "change_lane_left",
        "overtake": "emergency_brake",
        "park": "accelerate",
        "emergency_brake": "accelerate",
        "follow_lane": "emergency_brake",
    }
    if scene == "emergency":
        return "accelerate"
    return fallback.get(action, "accelerate")


def generate_preference_splits(output_path: Path, sft_samples: List[Dict]) -> None:
    """Emit DPO preference pairs and RLAIF reward-labeled JSONL from SFT rows."""
    train_end = int(len(sft_samples) * 0.8)
    val_end = train_end + int(len(sft_samples) * 0.1)
    splits = {
        "train": sft_samples[:train_end],
        "val": sft_samples[train_end:val_end],
        "test": sft_samples[val_end:],
    }
    for split_name, rows in splits.items():
        dpo_rows = []
        rlaif_rows = []
        for item in rows:
            action = item.get("action") or "keep_lane"
            controls = item.get("controls") or generate_controls_for_action(action)
            rejected_action = _unsafe_action(action, item.get("scene", "urban"))
            rejected_controls = generate_controls_for_action(rejected_action)
            dpo_rows.append({
                "schema_version": "1.0",
                "scene": item["scene"],
                "prompt": item["prompt"],
                "images": item["images"],
                "timestamp": item["timestamp"],
                "calibration": item["calibration"],
                "ego_state": item["ego_state"],
                "sensors": item.get("sensors") or {
                    "lidar": [], "radar": {}, "gps_imu": None,
                },
                "label_source": item.get("label_source", "synthetic"),
                "chosen": {
                    "response": item["response"],
                    "controls": controls,
                    "action": action,
                },
                "rejected": {
                    "response": generate_response(
                        rejected_action, rejected_controls, item["scene"]
                    ),
                    "controls": rejected_controls,
                    "action": rejected_action,
                },
            })
            # Higher reward for safer / control-aligned synthetic labels.
            safety = float(np.clip(0.55 + 0.4 * (action != "emergency_brake"), 0, 1))
            if item.get("scene") == "emergency" and action == "emergency_brake":
                safety = 0.95
            if rejected_action == action:
                safety = 0.5
            control_quality = float(np.clip(
                1.0 - abs(controls.get("steering", 0.0)) * 0.3
                - max(0.0, controls.get("brake", 0.0) - 0.5) * 0.2,
                0.2, 1.0,
            ))
            rlaif_rows.append({
                "schema_version": "1.0",
                "scene": item["scene"],
                "prompt": item["prompt"],
                "response": item["response"],
                "images": item["images"],
                "timestamp": item["timestamp"],
                "calibration": item["calibration"],
                "ego_state": item["ego_state"],
                "sensors": item.get("sensors") or {
                    "lidar": [], "radar": {}, "gps_imu": None,
                },
                "controls": controls,
                "action": action,
                "label_source": item.get("label_source", "synthetic"),
                "safety_score": safety,
                "control_quality": control_quality,
                "reward": float(0.6 * safety + 0.4 * control_quality),
            })

        dpo_path = output_path / f"dpo_{split_name}.jsonl"
        with open(dpo_path, "w", encoding="utf-8") as handle:
            for row in dpo_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        rlaif_path = output_path / f"rlaif_{split_name}.jsonl"
        with open(rlaif_path, "w", encoding="utf-8") as handle:
            for row in rlaif_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  dpo_{split_name}: {len(dpo_rows)} -> {dpo_path}")
        print(f"  rlaif_{split_name}: {len(rlaif_rows)} -> {rlaif_path}")


def generate_controls_for_action(action: str) -> Dict[str, float]:
    """根据动作生成控制信号"""
    control_map = {
        "keep_lane": {"steering": 0.0, "throttle": 0.3, "brake": 0.0, "gear": 2},
        "turn_left": {"steering": -0.3, "throttle": 0.2, "brake": 0.0, "gear": 2},
        "turn_right": {"steering": 0.3, "throttle": 0.2, "brake": 0.0, "gear": 2},
        "stop": {"steering": 0.0, "throttle": 0.0, "brake": 0.8, "gear": 2},
        "accelerate": {"steering": 0.0, "throttle": 0.8, "brake": 0.0, "gear": 3},
        "decelerate": {"steering": 0.0, "throttle": 0.0, "brake": 0.3, "gear": 2},
        "yield": {"steering": 0.0, "throttle": 0.1, "brake": 0.2, "gear": 2},
        "overtake": {"steering": -0.4, "throttle": 0.7, "brake": 0.0, "gear": 3},
        "park": {"steering": 0.5, "throttle": 0.0, "brake": 0.5, "gear": 0},
        "emergency_brake": {"steering": 0.0, "throttle": 0.0, "brake": 1.0, "gear": 2},
        "follow_lane": {"steering": 0.05, "throttle": 0.4, "brake": 0.0, "gear": 2},
        "change_lane_left": {"steering": -0.2, "throttle": 0.3, "brake": 0.0, "gear": 2},
        "change_lane_right": {"steering": 0.2, "throttle": 0.3, "brake": 0.0, "gear": 2},
    }
    return control_map.get(action, {"steering": 0.0, "throttle": 0.3, "brake": 0.0, "gear": 2})


def generate_scene_prompt(scene: str, controls: Dict) -> str:
    """生成场景描述 prompt"""
    prompts = {
        "highway": "高速公路行驶中，车速{speed}km/h，前方道路畅通，天气{weather}",
        "urban": "城市道路行驶，车速{speed}km/h，前方有红绿灯，天气{weather}",
        "intersection": "接近十字路口，车速{speed}km/h，信号灯{light}，天气{weather}",
        "parking": "停车场内，车速{speed}km/h，需要寻找停车位，天气{weather}",
        "emergency": "紧急情况！前方突然有障碍物，车速{speed}km/h，天气{weather}",
        "tunnel": "隧道内行驶，车速{speed}km/h，隧道内照明正常，天气{weather}",
        "construction": "施工区域，车速{speed}km/h，前方道路变窄，天气{weather}",
    }
    prompt_template = prompts.get(scene, prompts["urban"])
    return prompt_template.format(
        speed=int(np.random.uniform(0, 80)),
        weather=np.random.choice(["sunny", "cloudy", "rainy", "foggy"]),
        light=np.random.choice(["green", "yellow", "red"]),
    )


def generate_response(action: str, controls: Dict, scene: str) -> str:
    """生成决策响应"""
    responses = {
        "keep_lane": "保持当前车道行驶，速度稳定",
        "turn_left": "向左转弯，注意左侧来车和行人",
        "turn_right": "向右转弯，注意右侧行人和自行车",
        "stop": "停车，等待前方路况变化",
        "accelerate": "加速行驶，注意前方路况",
        "decelerate": "减速行驶，注意前方障碍物",
        "yield": "减速让行，等待安全后通过",
        "overtake": "超车，注意对向来车",
        "park": "停车入位，注意周围车辆和行人",
        "emergency_brake": "紧急制动！避免碰撞",
        "follow_lane": "跟随车道行驶，保持安全距离",
        "change_lane_left": "向左变道，注意左侧后视镜",
        "change_lane_right": "向右变道，注意右侧后视镜",
    }
    return responses.get(action, "保持安全驾驶")


def main():
    parser = argparse.ArgumentParser(description="准备驾驶训练数据")
    parser.add_argument("--source", type=str, default="synthetic",
                        choices=["synthetic", "nuscenes", "custom"],
                        help="数据源")
    parser.add_argument("--output", type=str, default="./dataset/driving/processed",
                        help="输出目录")
    parser.add_argument("--n_samples", type=int, default=1000,
                        help="合成数据样本数")
    parser.add_argument("--nuscenes_root", type=str, default="./data/nuscenes",
                        help="nuScenes 数据目录")
    parser.add_argument("--nuscenes_version", type=str, default=None,
                        help="例如 v1.0-mini；默认根据目录自动检测")

    args = parser.parse_args()

    if args.source == "synthetic":
        generate_synthetic_data(args.output, n_samples=args.n_samples)
    elif args.source == "nuscenes":
        prepare_nuscenes_data(
            args.nuscenes_root, args.output, version=args.nuscenes_version
        )
    elif args.source == "custom":
        prepare_custom_data(args.output, args.output)


if __name__ == "__main__":
    main()
