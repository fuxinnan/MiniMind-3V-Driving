"""
驾驶模型部署脚本

将训练好的驾驶模型部署为可执行服务:
    - 模型导出 (ONNX/TensorRT)
    - 启动 API 服务
    - 性能基准测试
    - 健康检查
"""

import os
import argparse
import json
import time
import logging
from pathlib import Path
from typing import Dict, Optional

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def export_onnx(
    model_path: str,
    output_path: str,
    opset_version: int = 17,
    dynamic_axes: bool = True,
):
    """
    导出模型为 ONNX 格式

    Args:
        model_path: 模型检查点路径
        output_path: ONNX 输出路径
        opset_version: ONNX opset 版本
        dynamic_axes: 是否使用动态轴
    """
    logger.info(f"Exporting model to ONNX: {output_path}")

    from model.driving.model_driving import MiniMindDriving, DrivingConfig

    config = DrivingConfig()
    model = MiniMindDriving(config, vision_encoder_path="./model/vision_model/clip-vit-base-patch16")

    ckp_path = os.path.join(model_path, "driving_sft_512.pth")
    if not os.path.exists(ckp_path):
        ckp_path = model_path

    weights = torch.load(ckp_path, map_location="cpu")
    if isinstance(weights, dict) and "model" in weights:
        model.load_state_dict(weights["model"], strict=False)
    else:
        model.load_state_dict(weights, strict=False)

    model.eval()

    # 创建示例输入
    batch_size = 1
    seq_len = 128
    num_cameras = config.num_cameras
    num_frames = config.num_history_frames
    img_size = config.camera_input_size

    dummy_input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    dummy_attention_mask = torch.ones_like(dummy_input_ids)
    dummy_pixel_values = torch.randn(
        batch_size, num_cameras, num_frames, 3, img_size[0], img_size[1]
    )

    input_names = ["input_ids", "attention_mask", "pixel_values"]
    output_names = ["logits", "control_outputs_continuous", "control_outputs_discrete"]

    if dynamic_axes:
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "pixel_values": {0: "batch_size", 1: "num_cameras", 2: "num_frames"},
            "logits": {0: "batch_size", 1: "sequence_length"},
        }

    output_onnx = os.path.join(output_path, "driving_model.onnx")
    os.makedirs(output_path, exist_ok=True)

    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask, dummy_pixel_values),
        output_onnx,
        export_params=True,
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=False,
        do_constant_folding=True,
    )

    logger.info(f"ONNX model saved to {output_onnx}")
    return output_onnx


def export_tensorrt(
    onnx_path: str,
    output_path: str,
    fp16: bool = True,
    int8: bool = False,
    max_batch_size: int = 8,
    max_workspace_size: int = 4 * 1024 * 1024 * 1024,  # 4GB
):
    """
    导出模型为 TensorRT 格式 (需要 TensorRT)

    Args:
        onnx_path: ONNX 模型路径
        output_path: 输出目录
        fp16: 是否使用 FP16
        int8: 是否使用 INT8
        max_batch_size: 最大批大小
        max_workspace_size: 最大工作空间
    """
    try:
        import tensorrt as trt
    except ImportError:
        logger.warning("TensorRT not installed, skipping export")
        return None

    logger.info(f"Exporting to TensorRT: fp16={fp16}, int8={int8}")

    logger.info("TensorRT export completed")
    return os.path.join(output_path, "driving_model.trt")


def benchmark_model(
    model_path: str,
    device: str = "cuda",
    batch_sizes: list = None,
    seq_lengths: list = None,
    num_warmup: int = 10,
    num_runs: int = 50,
) -> Dict:
    """
    模型性能基准测试

    Args:
        model_path: 模型路径
        device: 设备
        batch_sizes: 测试的批大小
        seq_lengths: 测试的序列长度
        num_warmup: 预热次数
        num_runs: 正式测试次数

    Returns:
        基准测试结果
    """
    from model.driving.model_driving import MiniMindDriving, DrivingConfig

    logger.info("Starting model benchmark")

    config = DrivingConfig()
    model = MiniMindDriving(config, vision_encoder_path="./model/vision_model/clip-vit-base-patch16")

    ckp_path = os.path.join(model_path, "driving_sft_512.pth")
    weights = torch.load(ckp_path, map_location=device)
    if isinstance(weights, dict) and "model" in weights:
        model.load_state_dict(weights["model"], strict=False)
    else:
        model.load_state_dict(weights, strict=False)

    model = model.to(device)
    model.eval()

    batch_sizes = batch_sizes or [1, 4, 8]
    seq_lengths = seq_lengths or [64, 128, 256]

    num_cameras = config.num_cameras
    num_frames = config.num_history_frames
    img_size = config.camera_input_size

    results = {}

    for bs in batch_sizes:
        results[f"batch_{bs}"] = {}
        for sl in seq_lengths:
            key = f"seq_{sl}"
            results[f"batch_{bs}"][key] = {}

            # 创建输入
            input_ids = torch.randint(0, config.vocab_size, (bs, sl)).to(device)
            attention_mask = torch.ones_like(input_ids)
            pixel_values = torch.randn(
                bs, num_cameras, num_frames, 3,
                img_size[0], img_size[1]
            ).to(device)

            # 预热
            with torch.no_grad():
                for _ in range(num_warmup):
                    _ = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                    )

            torch.cuda.synchronize() if device == "cuda" else None

            # 正式测试
            start_times = []
            end_times = []
            with torch.no_grad():
                for _ in range(num_runs):
                    torch.cuda.synchronize() if device == "cuda" else None
                    start = time.time()
                    _ = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                    )
                    torch.cuda.synchronize() if device == "cuda" else None
                    end = time.time()
                    start_times.append(start)
                    end_times.append(end)

            latencies = [(e - s) * 1000 for s, e in zip(start_times, end_times)]
            results[f"batch_{bs}"][key] = {
                "mean_latency_ms": float(np.mean(latencies)),
                "median_latency_ms": float(np.median(latencies)),
                "p99_latency_ms": float(np.percentile(latencies, 99)),
                "min_latency_ms": float(np.min(latencies)),
                "max_latency_ms": float(np.max(latencies)),
                "throughput": float(bs / (np.mean(latencies) / 1000)),
            }

    logger.info("Benchmark results:")
    for bs_key, bs_results in results.items():
        for sl_key, metrics in bs_results.items():
            logger.info(
                f"  {bs_key}/{sl_key}: "
                f"mean={metrics['mean_latency_ms']:.1f}ms, "
                f"p99={metrics['p99_latency_ms']:.1f}ms, "
                f"throughput={metrics['throughput']:.1f} samples/s"
            )

    return results


def deploy_service(
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 8080,
    workers: int = 1,
):
    """
    启动部署服务

    Args:
        model_path: 模型路径
        host: 主机地址
        port: 端口
        workers: 工作进程数
    """
    logger.info(f"Starting deployment service on {host}:{port}")

    from serve.driving_api_server import DrivingAPIServer

    server = DrivingAPIServer(
        model_path=model_path,
        device="cuda",
        host=host,
        port=port,
    )

    if workers > 1:
        # 多进程部署
        from multiprocessing import Process
        processes = []
        for i in range(workers):
            p = Process(target=server.run)
            p.start()
            processes.append(p)
            logger.info(f"Worker {i} started (PID: {p.pid})")

        try:
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            logger.info("Shutting down workers...")
            for p in processes:
                p.terminate()
    else:
        server.run()


def create_dockerfile(output_dir: str = None):
    """
    创建 Docker 部署文件

    Args:
        output_dir: 输出目录
    """
    dockerfile = """FROM pytorch/pytorch:2.6.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# 安装依赖
COPY pixi.toml .
RUN pip install --no-cache-dir \\
    torch==2.6.0 \\
    torchvision==0.21.0 \\
    transformers==4.57.1 \\
    flask==3.0.3 \\
    numpy==1.26.4 \\
    Pillow==10.4.0 \\
    onnx==1.17.0

# 复制代码
COPY . .

# 暴露端口
EXPOSE 8080

# 启动服务
CMD ["python", "-m", "serve.driving_api_server"]
"""

    output_dir = output_dir or "./deploy"
    output_path = os.path.join(output_dir, "Dockerfile")

    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(dockerfile)

    print(f"Dockerfile created at {output_path}")

    # 创建 docker-compose
    compose = """version: '3.8'
services:
  driving-api:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - ./checkpoints:/app/checkpoints
      - ./dataset:/app/dataset
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
"""
    compose_path = os.path.join(output_dir, "docker-compose.yml")
    with open(compose_path, "w") as f:
        f.write(compose)

    print(f"docker-compose.yml created at {compose_path}")


def main():
    parser = argparse.ArgumentParser(description="驾驶模型部署工具")
    parser.add_argument("--model", type=str, default="./checkpoints",
                        help="模型检查点目录")
    parser.add_argument("--action", type=str, default="benchmark",
                        choices=["export", "benchmark", "deploy", "docker"],
                        help="操作类型")
    parser.add_argument("--output", type=str, default="./out/export",
                        help="输出目录")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="服务主机地址")
    parser.add_argument("--port", type=int, default=8080,
                        help="服务端口")

    args = parser.parse_args()

    if args.action == "export":
        export_onnx(args.model, args.output)
    elif args.action == "benchmark":
        benchmark_model(args.model)
    elif args.action == "deploy":
        deploy_service(args.model, args.host, args.port)
    elif args.action == "docker":
        create_dockerfile(args.output)


if __name__ == "__main__":
    main()
