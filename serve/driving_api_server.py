"""
驾驶推理 API 服务

基于 Flask 提供 RESTful API:
    POST /api/drive        - 单帧驾驶决策
    POST /api/drive_batch  - 批量驾驶决策
    GET  /api/health       - 健康检查
    GET  /api/model_info   - 模型信息
"""

import os
import time
import json
import base64
import logging
from typing import Dict, List, Optional
from pathlib import Path

import torch
from flask import Flask, request, jsonify

from model.driving.model_driving import MiniMindDriving, DrivingConfig
from config.driving_config import DrivingConfig as AppConfig


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class DrivingAPIServer:
    """
    驾驶推理 API 服务

    提供 HTTP RESTful API，支持多相机图像输入和控制信号输出
    """

    def __init__(
        self,
        model_path: str = "./checkpoints",
        model_filename: str = "driving_sft_512.pth",
        device: str = "cuda",
        host: str = "0.0.0.0",
        port: int = 8080,
        max_batch_size: int = 8,
        vision_encoder_path: str = "./model/vision_model/clip-vit-base-patch16",
    ):
        self.device = device
        self.host = host
        self.port = port
        self.max_batch_size = max_batch_size
        self.vision_encoder_path = vision_encoder_path

        # 加载模型
        self.model = self._load_model(model_path, model_filename)
        self.model.eval()

        # 加载 tokenizer
        from transformers import AutoTokenizer
        tokenizer_path = os.path.join(os.path.dirname(__file__), "..", "model")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # 创建 Flask 应用
        self.app = Flask(__name__)
        self._register_routes()

    def _load_model(self, model_path: str, model_filename: str) -> MiniMindDriving:
        """加载驾驶模型"""
        ckp_path = os.path.join(model_path, model_filename)

        if not os.path.exists(ckp_path):
            logger.warning(f"Checkpoint not found: {ckp_path}, using random init")
            config = DrivingConfig()
            return MiniMindDriving(config, vision_encoder_path=self.vision_encoder_path)

        # 加载权重
        from model.model_minimind import MiniMindConfig
        config = DrivingConfig()
        model = MiniMindDriving(config, vision_encoder_path=self.vision_encoder_path)

        weights = torch.load(ckp_path, map_location=self.device)
        if isinstance(weights, dict) and "model" in weights:
            model.load_state_dict(weights["model"], strict=False)
        else:
            model.load_state_dict(weights, strict=False)

        logger.info(f"Model loaded from {ckp_path}")
        logger.info(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")

        return model.to(self.device)

    def _register_routes(self):
        """注册 API 路由"""

        @self.app.route("/api/health", methods=["GET"])
        def health():
            return jsonify({
                "status": "healthy",
                "model_loaded": self.model is not None,
                "device": self.device,
            })

        @self.app.route("/api/model_info", methods=["GET"])
        def model_info():
            total = sum(p.numel() for p in self.model.parameters())
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            return jsonify({
                "model_type": "MiniMindDriving",
                "total_parameters": total,
                "trainable_parameters": trainable,
                "num_cameras": self.model.config.num_cameras,
                "camera_names": self.model.config.camera_names,
                "num_history_frames": self.model.config.num_history_frames,
                "discrete_actions": self.model.config.discrete_actions,
            })

        @self.app.route("/api/drive", methods=["POST"])
        def drive():
            """单帧驾驶决策"""
            try:
                data = request.get_json()
                if not data:
                    return jsonify({"error": "No JSON data provided"}), 400

                pixel_values = self._parse_images(data.get("images", {}))
                if pixel_values is None:
                    return jsonify({"error": "Invalid image data"}), 400

                prompt = data.get("prompt", "请分析当前驾驶场景并给出决策")
                max_tokens = data.get("max_tokens", 128)
                temperature = data.get("temperature", 0.7)

                result = self._infer(
                    pixel_values, prompt, max_tokens, temperature
                )
                return jsonify(result)

            except Exception as e:
                logger.error(f"Inference error: {str(e)}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/drive_batch", methods=["POST"])
        def drive_batch():
            """批量驾驶决策"""
            try:
                data = request.get_json()
                if not data:
                    return jsonify({"error": "No JSON data provided"}), 400

                items = data.get("batch", [])
                if not items:
                    return jsonify({"error": "Empty batch"}), 400

                if len(items) > self.max_batch_size:
                    return jsonify({
                        "error": f"Batch size exceeds max ({self.max_batch_size})"
                    }), 400

                results = []
                for item in items:
                    pixel_values = self._parse_images(item.get("images", {}))
                    if pixel_values is None:
                        results.append({"error": "Invalid image data"})
                        continue

                    prompt = item.get("prompt", "请分析当前驾驶场景并给出决策")
                    result = self._infer(pixel_values, prompt)
                    results.append(result)

                return jsonify({"results": results})

            except Exception as e:
                logger.error(f"Batch inference error: {str(e)}")
                return jsonify({"error": str(e)}), 500

    def _parse_images(self, images_data: Dict) -> Optional[torch.Tensor]:
        """
        解析输入的图像数据

        支持格式:
            {"front": ["base64_string", ...], "left": [...], ...}
            {"front": ["path/to/img1.jpg", ...], ...}
        """
        from PIL import Image
        from torchvision import transforms
        import io

        num_cameras = self.model.config.num_cameras
        cam_names = self.model.config.camera_names[:num_cameras]
        img_size = self.model.config.camera_input_size
        num_frames = self.model.config.num_history_frames

        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])

        pixel_values = []
        for cam_name in cam_names:
            cam_images = images_data.get(cam_name, [])
            if not cam_images:
                blank = torch.zeros((3, *img_size))
                cam_images = [blank] * num_frames
            else:
                frames = []
                for img_data in cam_images[-num_frames:]:
                    if isinstance(img_data, str):
                        if img_data.startswith("data:"):
                            # base64
                            img_bytes = base64.b64decode(
                                img_data.split(",")[1]
                            )
                            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        elif os.path.exists(img_data):
                            img = Image.open(img_data).convert("RGB")
                        else:
                            img = Image.new("RGB", img_size)
                    else:
                        img = img_data

                    frames.append(transform(img))

                while len(frames) < num_frames:
                    frames.append(frames[-1] if frames else torch.zeros((3, *img_size)))

                cam_images = frames

            pixel_values.append(torch.stack(cam_images))

        return torch.stack(pixel_values).to(self.device)

    @torch.no_grad()
    def _infer(
        self,
        pixel_values: torch.Tensor,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
    ) -> Dict:
        """执行单次推理"""
        start_time = time.time()

        messages = [{"role": "user", "content": prompt}]
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

        # 生成文本
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            pixel_values=pixel_values,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        text_response = self.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        )

        # 获取控制输出
        control_output = {
            "steering": 0.0,
            "throttle": 0.0,
            "brake": 0.0,
            "gear": 2,
        }

        # 尝试从模型获取控制输出
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                pixel_values=pixel_values,
            )
            if outputs.control_outputs is not None:
                ctrl = outputs.control_outputs["continuous"][0].cpu().numpy()
                control_output = {
                    "steering": float(ctrl[0]),
                    "throttle": float(ctrl[1]),
                    "brake": float(ctrl[2]),
                    "gear": int(round((ctrl[3] + 1) / 2 * 4)),
                }

        latency = time.time() - start_time

        return {
            "text_response": text_response,
            "control": control_output,
            "latency_ms": latency * 1000,
        }

    def run(self):
        """启动 API 服务"""
        logger.info(f"Starting Driving API server on {self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=False)


def main():
    """启动 API 服务"""
    server = DrivingAPIServer(
        model_path="./checkpoints",
        model_filename="driving_sft_512.pth",
        device="cuda",
        host="0.0.0.0",
        port=8080,
    )
    server.run()


if __name__ == "__main__":
    main()
