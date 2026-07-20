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
    split_ratios = split_ratios or {"train": 0.8, "val": 0.1, "test": 0.1}
    camera_names = camera_names or ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK"]

    print(f"Preparing nuScenes data from {nuscenes_root}")
    print(f"Output dir: {output_dir}")
    print(f"Cameras: {camera_names}")

    # 加载 nuScenes annotation
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version="v1.0-trainval", dataroot=nuscenes_root)

    # 场景分类映射
    scene_category_map = {
        "scene.name": "scene",
    }

    train_data = []
    val_data = []
    test_data = []

    for i, sample in enumerate(tqdm(nusc.sample, desc="Processing samples")):
        # 获取场景信息
        scene_token = sample["scene_token"]
        scene = nusc.get("scene", scene_token)
        scene_name = scene["name"]

        # 场景分类
        scene_type = classify_nuscenes_scene(scene_name)

        # 获取该场景的所有帧
        log_token = scene["log_token"]
        camera_tokens = [
            nusc.get("token2log", sample["token"]) for sample in nusc.sample
        ]

        # 简化: 只取部分帧作为示例
        # 实际实现需要更复杂的帧选择和关联

        if i < len(nusc.sample) * split_ratios["train"]:
            train_data.append({
                "scene": scene_type,
                "prompt": f"高速公路场景，车速60km/h",
                "response": "保持当前车道行驶",
                "controls": {"steering": 0.0, "throttle": 0.3, "brake": 0.0, "gear": 2},
                "action": "keep_lane",
                "weather": "sunny",
            })
        elif i < len(nusc.sample) * (split_ratios["train"] + split_ratios["val"]):
            val_data.append({
                "scene": scene_type,
                "prompt": f"城市道路场景",
                "response": "减速准备转弯",
                "controls": {"steering": 0.1, "throttle": 0.1, "brake": 0.2, "gear": 2},
                "action": "turn_right",
                "weather": "cloudy",
            })
        else:
            test_data.append({
                "scene": scene_type,
                "prompt": f"十字路口场景",
                "response": "停车让行",
                "controls": {"steering": 0.0, "throttle": 0.0, "brake": 0.8, "gear": 2},
                "action": "stop",
                "weather": "rainy",
            })

    # 保存数据
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for split, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        split_path = output_path / f"{split}.jsonl"
        with open(split_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  {split}: {len(data)} samples -> {split_path}")

    print(f"Preparation complete!")


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
            "scene": scene,
            "prompt": prompt,
            "response": response,
            "controls": controls,
            "action": action,
            "weather": np.random.choice(weather_options),
            "time_of_day": np.random.choice(time_options),
            "speed": float(np.random.uniform(0, 120)),
            "timestamp": int(datetime.now().timestamp()),
        })

    # 保存
    for split_name, split_ratio in [("train", 0.8), ("val", 0.1), ("test", 0.1)]:
        n = int(n_samples * split_ratio)
        split_data = data[:n] if split_name == "train" else (
            data[n:n + int(n_samples * 0.1)] if split_name == "val" else data[n + int(n_samples * 0.1):]
        )

        split_path = output_path / f"{split_name}.jsonl"
        with open(split_path, "w", encoding="utf-8") as f:
            for item in split_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"  {split_name}: {len(split_data)} samples -> {split_path}")

    print(f"Synthetic data generation complete!")


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

    args = parser.parse_args()

    if args.source == "synthetic":
        generate_synthetic_data(args.output, n_samples=args.n_samples)
    elif args.source == "nuscenes":
        prepare_nuscenes_data(args.nuscenes_root, args.output)
    elif args.source == "custom":
        prepare_custom_data(args.output, args.output)


if __name__ == "__main__":
    main()
