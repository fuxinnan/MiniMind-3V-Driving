"""Thin Flask API for the driving inference engine."""

import argparse
import logging
from typing import Optional

import torch
from flask import Flask, jsonify, request

from serve.inference_engine import DrivingInferenceEngine, InferenceInputError

LOGGER = logging.getLogger(__name__)


def create_app(
    engine: DrivingInferenceEngine,
    max_batch_size: int = 8,
    enable_cors: bool = False,
) -> Flask:
    app = Flask(__name__)
    if enable_cors:
        from flask_cors import CORS
        CORS(app)

    @app.get("/api/health")
    def health():
        return jsonify({
            "status": "healthy",
            "model_loaded": engine.model is not None,
            "device": str(engine.device),
            "model_version": engine.model_version,
        })

    @app.get("/api/model_info")
    def model_info():
        config = engine.model.config
        return jsonify({
            "model_type": "MiniMindDriving",
            "model_version": engine.model_version,
            "num_cameras": config.num_cameras,
            "camera_names": config.camera_names,
            "num_history_frames": config.num_history_frames,
            "discrete_actions": config.discrete_actions,
        })

    @app.post("/api/drive")
    def drive():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "JSON object required"}), 400
        try:
            result = engine.infer(
                payload.get("images") or {},
                str(payload.get("prompt") or "分析当前驾驶场景并给出决策"),
                payload.get("sensors"),
            )
            return jsonify(result)
        except InferenceInputError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception:
            LOGGER.exception("Driving inference failed")
            return jsonify({"error": "inference failed"}), 500

    @app.post("/api/drive_batch")
    def drive_batch():
        payload = request.get_json(silent=True)
        items = payload.get("batch") if isinstance(payload, dict) else None
        if not isinstance(items, list) or not items:
            return jsonify({"error": "non-empty batch required"}), 400
        if len(items) > max_batch_size:
            return jsonify({"error": f"batch exceeds {max_batch_size}"}), 400
        results = []
        for item in items:
            try:
                results.append(engine.infer(
                    item.get("images") or {},
                    str(item.get("prompt") or "分析当前驾驶场景并给出决策"),
                    item.get("sensors"),
                ))
            except InferenceInputError as exc:
                results.append({"error": str(exc)})
        return jsonify({"results": results})

    return app


class DrivingAPIServer:
    """Compatibility wrapper around the new app factory."""

    def __init__(
        self,
        model_path: str = "./checkpoints/driving_sft_512.pth",
        device: str = "cpu",
        host: str = "0.0.0.0",
        port: int = 8080,
        max_batch_size: int = 8,
        vision_encoder_path: str = "./model/vision_model/clip-vit-base-patch16",
        engine: Optional[DrivingInferenceEngine] = None,
        enable_cors: bool = False,
        **_,
    ):
        self.host, self.port = host, port
        self.engine = engine or DrivingInferenceEngine(
            checkpoint=model_path,
            device=device,
            vision_encoder_path=vision_encoder_path,
            require_checkpoint=True,
        )
        self.model = self.engine.model
        self.app = create_app(self.engine, max_batch_size, enable_cors)

    def run(self):
        self.app.run(host=self.host, port=self.port, debug=False)


def main():
    parser = argparse.ArgumentParser(description="MiniMind driving REST API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--vision-encoder", default="./model/vision_model/clip-vit-base-patch16")
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--cors", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    DrivingAPIServer(
        model_path=args.model_path,
        device=args.device,
        host=args.host,
        port=args.port,
        max_batch_size=args.max_batch_size,
        vision_encoder_path=args.vision_encoder,
        enable_cors=args.cors,
    ).run()


if __name__ == "__main__":
    main()
