"""
驾驶数据加载器

支持多种驾驶数据集格式:
    - nuScenes: 标准自动驾驶数据集
    - Waymo Open: Waymo 开放数据集
    - CARLA: 仿真器数据
    - 自定义 JSONL 格式

数据格式:
    {"scene": "highway",
     "prompt": "描述当前驾驶场景...",
     "response": "驾驶决策解释...",
     "images": {"front": ["img1.jpg"], "left": [...], ...},
     "controls": {"steering": 0.05, "throttle": 0.3, "brake": 0.0, "gear": 2},
     "action": "keep_lane",
     "timestamp": 1234567890,
     "speed": 60.0,
     "weather": "sunny",
     "time_of_day": "day"}
"""

import os
import json
import glob
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from datasets import Dataset, load_dataset

import torch
from torch.utils.data import Dataset
from PIL import Image

from config.driving_config import DrivingConfig, DATA_FORMAT_CONFIG


class DrivingSFTDataset(Dataset):
    """
    驾驶监督微调数据集

    加载 JSONL 格式的驾驶对话数据，包含图像和控制标签
    """

    def __init__(
        self,
        data_path: str,
        config: DrivingConfig,
        image_root: str = "./dataset/driving/raw/camera",
        tokenizer=None,
        max_seq_len: int = 2048,
        num_frames: int = 3,
        transform=None,
    ):
        self.config = config
        self.image_root = Path(image_root)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.num_frames = num_frames
        self.transform = transform

        self.data = self._load_data(data_path)
        self._validate_data()

    def _load_data(self, data_path: str) -> List[Dict]:
        """加载 JSONL 数据"""
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
            if isinstance(data, dict) and "data" in data:
                return data["data"]
            return data if isinstance(data, list) else [data]
        elif os.path.isdir(data_path):
            all_data = []
            for jsonl_file in glob.glob(os.path.join(data_path, "*.jsonl")):
                all_data.extend(self._load_data(jsonl_file))
            return all_data
        else:
            raise ValueError(f"Unsupported data format: {data_path}")

    def _validate_data(self):
        """验证数据格式"""
        required_fields = DATA_FORMAT_CONFIG["sft_fields"]["required"]
        for i, item in enumerate(self.data):
            for field in required_fields:
                if field not in item:
                    raise ValueError(
                        f"Data item {i} missing required field '{field}'. "
                        f"Available fields: {list(item.keys())}"
                    )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        # 加载图像
        pixel_values = self._load_images(item)

        # 构建提示和响应
        prompt = item.get("prompt", "")
        response = item.get("response", "")

        # 构建对话格式
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]

        # 分词
        if self.tokenizer:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            inputs = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_seq_len,
                padding="max_length" if len(text) < self.max_seq_len else "do_not_pad",
                return_tensors="pt",
            )
        else:
            inputs = {
                "input_ids": torch.tensor([[1]]),
                "attention_mask": torch.tensor([[1]]),
            }

        # 控制标签
        control_labels = self._parse_controls(item)
        action_labels = self._parse_action(item)

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": pixel_values,
            "control_labels": control_labels,
            "action_labels": action_labels,
            "scene": item.get("scene", "unknown"),
            "metadata": {
                "speed": item.get("speed", 0.0),
                "weather": item.get("weather", "unknown"),
                "time_of_day": item.get("time_of_day", "unknown"),
            },
        }

    def _load_images(self, item: Dict) -> torch.Tensor:
        """
        加载多相机图像

        Returns:
            [num_cameras, num_frames, C, H, W]
        """
        images = item.get("images", {})
        num_cameras = self.config.num_cameras
        cam_names = self.config.camera_names[:num_cameras]
        img_size = self.config.camera_input_size

        pixel_values = []
        for cam_name in cam_names:
            cam_images = images.get(cam_name, [])
            if not cam_images:
                # 创建空白图像作为占位符
                blank_img = Image.new("RGB", img_size, (0, 0, 0))
                cam_images = [blank_img] * self.num_frames

            # 取最近的 num_frames 帧
            frames = cam_images[-self.num_frames:]
            while len(frames) < self.num_frames:
                frames.append(frames[-1] if frames else Image.new("RGB", img_size, (0, 0, 0)))

            cam_tensors = []
            for img in frames:
                if isinstance(img, str):
                    img_path = os.path.join(self.image_root, cam_name, img)
                    if os.path.exists(img_path):
                        img = Image.open(img_path).convert("RGB")
                    else:
                        img = Image.new("RGB", img_size, (0, 0, 0))
                if self.transform:
                    img = self.transform(img)
                else:
                    from torchvision import transforms
                    transform = transforms.Compose([
                        transforms.Resize(img_size),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225]),
                    ])
                    img = transform(img)
                cam_tensors.append(img)

            pixel_values.append(torch.stack(cam_tensors))

        return torch.stack(pixel_values)

    def _parse_controls(self, item: Dict) -> Optional[torch.Tensor]:
        """解析连续控制标签"""
        controls = item.get("controls", {})
        if not controls:
            return None
        keys = DATA_FORMAT_CONFIG["control_format"]["continuous_keys"]
        values = [controls.get(k, 0.0) for k in keys]
        return torch.tensor(values, dtype=torch.float32)

    def _parse_action(self, item: Dict) -> Optional[torch.LongTensor]:
        """解析离散动作标签"""
        action = item.get("action", "")
        if not action:
            return None
        action_to_id = {name: i for i, name in enumerate(self.config.discrete_actions)}
        return torch.tensor(action_to_id.get(action, 0), dtype=torch.long)


class DrivingDPODataset(Dataset):
    """
    驾驶偏好优化数据集 (Direct Preference Optimization)

    数据格式:
    {
        "prompt": "描述场景...",
        "chosen": {"response": "好的决策", "controls": {...}, "action": "..."},
        "rejected": {"response": "差的决策", "controls": {...}, "action": "..."},
        "images": {...},
        "scene": "highway"
    }
    """

    def __init__(
        self,
        data_path: str,
        config: DrivingConfig,
        image_root: str = "./dataset/driving/raw/camera",
        tokenizer=None,
        max_seq_len: int = 2048,
        num_frames: int = 3,
    ):
        self.config = config
        self.image_root = Path(image_root)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.num_frames = num_frames

        self.data = self._load_data(data_path)

    def _load_data(self, data_path: str) -> List[Dict]:
        if data_path.endswith(".jsonl"):
            data = []
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            return data
        elif os.path.isdir(data_path):
            all_data = []
            for jsonl_file in glob.glob(os.path.join(data_path, "*.jsonl")):
                all_data.extend(self._load_data(jsonl_file))
            return all_data
        raise ValueError(f"Unsupported data format: {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        pixel_values = self._load_images(item)
        prompt = item.get("prompt", "")

        chosen_response = item["chosen"]["response"]
        rejected_response = item["rejected"]["response"]

        messages_chosen = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen_response},
        ]
        messages_rejected = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": rejected_response},
        ]

        if self.tokenizer:
            chosen_text = self.tokenizer.apply_chat_template(
                messages_chosen, tokenize=False, add_generation_prompt=False
            )
            rejected_text = self.tokenizer.apply_chat_template(
                messages_rejected, tokenize=False, add_generation_prompt=False
            )
            chosen_inputs = self.tokenizer(
                chosen_text, truncation=True, max_length=self.max_seq_len,
                return_tensors="pt",
            )
            rejected_inputs = self.tokenizer(
                rejected_text, truncation=True, max_length=self.max_seq_len,
                return_tensors="pt",
            )
        else:
            chosen_inputs = {"input_ids": torch.tensor([[1]])}
            rejected_inputs = {"input_ids": torch.tensor([[1]])}

        return {
            "chosen_input_ids": chosen_inputs["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_inputs.get("attention_mask", torch.tensor([[1]])).squeeze(0),
            "rejected_input_ids": rejected_inputs["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_inputs.get("attention_mask", torch.tensor([[1]])).squeeze(0),
            "pixel_values": pixel_values,
            "scene": item.get("scene", "unknown"),
        }

    def _load_images(self, item: Dict) -> torch.Tensor:
        images = item.get("images", {})
        num_cameras = self.config.num_cameras
        cam_names = self.config.camera_names[:num_cameras]
        img_size = self.config.camera_input_size

        pixel_values = []
        for cam_name in cam_names:
            cam_images = images.get(cam_name, [])
            if not cam_images:
                blank_img = Image.new("RGB", img_size, (0, 0, 0))
                cam_images = [blank_img] * self.num_frames
            frames = cam_images[-self.num_frames:]
            while len(frames) < self.num_frames:
                frames.append(frames[-1] if frames else Image.new("RGB", img_size, (0, 0, 0)))
            cam_tensors = []
            for img in frames:
                if isinstance(img, str):
                    img_path = os.path.join(self.image_root, cam_name, img)
                    if os.path.exists(img_path):
                        img = Image.open(img_path).convert("RGB")
                    else:
                        img = Image.new("RGB", img_size, (0, 0, 0))
                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.Resize(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225]),
                ])
                cam_tensors.append(transform(img))
            pixel_values.append(torch.stack(cam_tensors))
        return torch.stack(pixel_values)


class DrivingRLAIFDataset(Dataset):
    """
    驾驶 AI 反馈强化学习数据集

    数据格式:
    {
        "conversations": [
            {"role": "user", "content": "场景描述..."},
            {"role": "assistant", "content": "模型生成的决策"}
        ],
        "images": {...},
        "scene": "highway",
        "safety_score": 0.95,
        "control_quality": 0.88
    }
    """

    def __init__(
        self,
        data_path: str,
        config: DrivingConfig,
        image_root: str = "./dataset/driving/raw/camera",
        tokenizer=None,
        max_seq_len: int = 2048,
        num_frames: int = 3,
    ):
        self.config = config
        self.image_root = Path(image_root)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.num_frames = num_frames

        self.data = self._load_data(data_path)

    def _load_data(self, data_path: str) -> List[Dict]:
        if data_path.endswith(".jsonl"):
            data = []
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            return data
        elif os.path.isdir(data_path):
            all_data = []
            for jsonl_file in glob.glob(os.path.join(data_path, "*.jsonl")):
                all_data.extend(self._load_data(jsonl_file))
            return all_data
        raise ValueError(f"Unsupported data format: {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        conversations = item.get("conversations", [])

        pixel_values = self._load_images(item)

        if self.tokenizer:
            text = self.tokenizer.apply_chat_template(
                conversations, tokenize=False, add_generation_prompt=False
            )
            inputs = self.tokenizer(
                text, truncation=True, max_length=self.max_seq_len,
                return_tensors="pt",
            )
        else:
            inputs = {"input_ids": torch.tensor([[1]])}

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs.get("attention_mask", torch.tensor([[1]])).squeeze(0),
            "pixel_values": pixel_values,
            "scene": item.get("scene", "unknown"),
            "safety_score": item.get("safety_score", 0.0),
            "control_quality": item.get("control_quality", 0.0),
        }

    def _load_images(self, item: Dict) -> torch.Tensor:
        images = item.get("images", {})
        num_cameras = self.config.num_cameras
        cam_names = self.config.camera_names[:num_cameras]
        img_size = self.config.camera_input_size

        pixel_values = []
        for cam_name in cam_names:
            cam_images = images.get(cam_name, [])
            if not cam_images:
                blank_img = Image.new("RGB", img_size, (0, 0, 0))
                cam_images = [blank_img] * self.num_frames
            frames = cam_images[-self.num_frames:]
            while len(frames) < self.num_frames:
                frames.append(frames[-1] if frames else Image.new("RGB", img_size, (0, 0, 0)))
            cam_tensors = []
            for img in frames:
                if isinstance(img, str):
                    img_path = os.path.join(self.image_root, cam_name, img)
                    if os.path.exists(img_path):
                        img = Image.open(img_path).convert("RGB")
                    else:
                        img = Image.new("RGB", img_size, (0, 0, 0))
                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.Resize(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225]),
                ])
                cam_tensors.append(transform(img))
            pixel_values.append(torch.stack(cam_tensors))
        return torch.stack(pixel_values)


class NuScenesDataset(Dataset):
    """
    nuScenes 数据集加载器

    支持 nuScenes 标准格式，自动转换为 SFT 格式
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        config: DrivingConfig = None,
        tokenizer=None,
        max_seq_len: int = 2048,
        num_frames: int = 3,
    ):
        self.data_root = data_root
        self.split = split
        self.config = config or DrivingConfig()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.num_frames = num_frames

        self.annotations_path = os.path.join(data_root, f"{split}.json")
        self.sample_to_scene = {}
        self.scenes = {}

        self._load_annotations()

    def _load_annotations(self):
        """加载 nuScenes 标注文件"""
        import json
        with open(self.annotations_path, "r") as f:
            data = json.load(f)

        for sample in data.get("sample", []):
            self.sample_to_scene[sample["id"]] = sample["scene_token"]

        for scene in data.get("scene", []):
            self.scenes[scene["id"]] = scene

    def __len__(self):
        return len(self.sample_to_scene)

    def __getitem__(self, idx: int):
        sample_id = list(self.sample_to_scene.keys())[idx]
        scene_token = self.sample_to_scene[sample_id]
        scene = self.scenes.get(scene_token, {})

        scene_name = scene.get("name", "unknown")
        scene_type = self._classify_scene(scene_name)

        # 这里需要实际加载 nuScenes 数据
        # 简化版本: 返回占位符
        return {
            "input_ids": torch.tensor([[1]]),
            "attention_mask": torch.tensor([[1]]),
            "pixel_values": torch.zeros((4, self.num_frames, 3, 224, 224)),
            "scene": scene_type,
            "metadata": {"scene_token": scene_token, "sample_token": sample_id},
        }

    def _classify_scene(self, scene_name: str) -> str:
        """根据场景名称分类"""
        scene_lower = scene_name.lower()
        if "highway" in scene_lower or "motorway" in scene_lower:
            return "highway"
        elif "intersection" in scene_lower:
            return "intersection"
        elif "parking" in scene_lower:
            return "parking"
        elif "residential" in scene_lower:
            return "residential"
        elif "urban" in scene_lower:
            return "urban"
        return "unknown"
