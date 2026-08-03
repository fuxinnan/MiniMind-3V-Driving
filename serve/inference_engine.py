"""Single-pass inference and strict request preprocessing."""

import base64
import io
import os
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
from PIL import Image, UnidentifiedImageError

from config.driving_config import DrivingConfig
from model.driving.model_driving import MiniMindDriving


class InferenceInputError(ValueError):
    """Invalid client input."""


class DrivingInferenceEngine:
    def __init__(
        self,
        checkpoint: Optional[str] = None,
        device: str = "cpu",
        vision_encoder_path: str = "./model/vision_model/clip-vit-base-patch16",
        model=None,
        tokenizer=None,
        require_checkpoint: bool = True,
    ):
        self.device = torch.device(device)
        self.model = model or self._load_model(
            checkpoint, vision_encoder_path, require_checkpoint
        )
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("./model")
        self.tokenizer = tokenizer
        self.model.to(self.device).eval()
        self.model_version = (
            Path(checkpoint).stem if checkpoint else "injected-test-model"
        )

    def _load_model(self, checkpoint, vision_path, require_checkpoint):
        if require_checkpoint and (not checkpoint or not Path(checkpoint).is_file()):
            raise FileNotFoundError(
                f"Driving checkpoint not found: {checkpoint!r}"
            )
        config_path = (
            Path(checkpoint).with_suffix(".config.json")
            if checkpoint else None
        )
        config = (
            DrivingConfig.from_json_file(str(config_path))
            if config_path and config_path.is_file()
            else DrivingConfig(vision_encoder_path=vision_path)
        )
        model = MiniMindDriving(config, vision_encoder_path=vision_path)
        if checkpoint:
            payload = torch.load(checkpoint, map_location="cpu")
            state = payload.get("model", payload) if isinstance(payload, dict) else payload
            missing, unexpected = model.load_state_dict(state, strict=False)
            if unexpected:
                raise RuntimeError(f"Unexpected checkpoint keys: {unexpected[:5]}")
        return model

    @staticmethod
    def _decode_image(value: Any) -> Image.Image:
        if not isinstance(value, str) or not value:
            raise InferenceInputError("each image must be a path or base64 string")
        try:
            if value.startswith("data:"):
                if "," not in value:
                    raise InferenceInputError("malformed data URI")
                value = value.split(",", 1)[1]
                return Image.open(io.BytesIO(base64.b64decode(value, validate=True))).convert(
                    "RGB"
                )
            path = Path(value)
            if path.is_file():
                return Image.open(path).convert("RGB")
            return Image.open(
                io.BytesIO(base64.b64decode(value, validate=True))
            ).convert("RGB")
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            raise InferenceInputError("image is not a valid path or base64 payload") from exc

    def preprocess_images(self, images: Mapping[str, Any]) -> torch.Tensor:
        config = self.model.config
        missing = [camera for camera in config.camera_names if camera not in images]
        if missing:
            raise InferenceInputError(f"missing camera streams: {missing}")
        height, width = config.camera_input_size
        cameras = []
        for camera in config.camera_names:
            references = images[camera]
            if not isinstance(references, list) or len(references) < config.num_history_frames:
                raise InferenceInputError(
                    f"{camera} requires {config.num_history_frames} frames"
                )
            frames = []
            for reference in references[-config.num_history_frames:]:
                image = self._decode_image(reference).resize(
                    (width, height), Image.Resampling.BILINEAR
                )
                tensor = torch.tensor(
                    bytearray(image.tobytes()), dtype=torch.uint8
                ).reshape(height, width, 3).permute(2, 0, 1).float() / 255.0
                mean = tensor.new_tensor([0.485, 0.456, 0.406])[:, None, None]
                std = tensor.new_tensor([0.229, 0.224, 0.225])[:, None, None]
                frames.append((tensor - mean) / std)
            cameras.append(torch.stack(frames))
        return torch.stack(cameras).unsqueeze(0).to(self.device)

    def _tokenize(self, prompt: str):
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoded = self.tokenizer(text, return_tensors="pt")
        return {
            key: value.to(self.device)
            for key, value in encoded.items() if torch.is_tensor(value)
        }

    def preprocess_sensors(self, sensors: Optional[Mapping[str, Any]]):
        sensors = sensors or {}
        config = self.model.config
        output: Dict[str, torch.Tensor] = {}
        active = ["camera"]

        def points(name, max_points, dimensions, value_key, mask_key):
            value = sensors.get(name)
            if value is None:
                return
            tensor = torch.as_tensor(value, dtype=torch.float32)
            if tensor.ndim != 2:
                raise InferenceInputError(f"{name} must be a 2D numeric array")
            padded = torch.zeros(max_points, dimensions)
            count = min(max_points, tensor.shape[0])
            width = min(dimensions, tensor.shape[1])
            padded[:count, :width] = tensor[:count, :width]
            mask = torch.zeros(max_points, dtype=torch.bool)
            mask[:count] = True
            output[value_key] = padded.unsqueeze(0).to(self.device)
            output[mask_key] = mask.unsqueeze(0).to(self.device)
            active.append(name)

        if config.enable_lidar:
            points(
                "lidar", config.lidar_num_points, config.lidar_point_dims,
                "lidar_pointcloud", "lidar_mask",
            )
        if config.enable_radar:
            points(
                "radar", config.radar_num_detections, config.radar_point_dims,
                "radar_data", "radar_mask",
            )
        if config.enable_gps_imu and sensors.get("gps_imu") is not None:
            gps = torch.as_tensor(sensors["gps_imu"], dtype=torch.float32).flatten()
            if gps.numel() != config.gps_imu_dims:
                raise InferenceInputError(
                    f"gps_imu requires {config.gps_imu_dims} values"
                )
            output["gps_imu"] = gps.unsqueeze(0).to(self.device)
            output["gps_imu_mask"] = torch.ones(
                1, dtype=torch.bool, device=self.device
            )
            active.append("gps_imu")
        return output, active

    @staticmethod
    def _reason(action: str, control: Dict[str, Any]) -> str:
        return (
            f"建议执行 {action}：转向 {control['steering']:.3f}，"
            f"油门 {control['throttle']:.3f}，制动 {control['brake']:.3f}。"
        )

    @torch.inference_mode()
    def infer(
        self,
        images: Mapping[str, Any],
        prompt: str,
        sensors: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        pixels = self.preprocess_images(images)
        sensor_inputs, active_sensors = self.preprocess_sensors(sensors)
        encoded = self._tokenize(prompt)
        outputs = self.model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded.get("attention_mask"),
            pixel_values=pixels,
            **sensor_inputs,
        )
        decision = outputs.control_outputs
        continuous = decision["continuous"][0].detach().cpu()
        control = self.model.control_head.decode_continuous(continuous)
        action_index = int(decision["discrete_action"][0])
        action = self.model.control_head.decode_discrete(action_index)
        probabilities = decision["discrete_probs"][0].detach().cpu().tolist()
        return {
            "text_response": self._reason(action, control),
            "control": control,
            "action": action,
            "action_probs": probabilities,
            "active_sensors": active_sensors,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "model_version": self.model_version,
        }
