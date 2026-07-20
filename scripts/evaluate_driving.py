"""
驾驶模型评估脚本

对训练好的驾驶模型进行全面评估:
    - 控制精度评估
    - 安全指标评估
    - 场景覆盖评估
    - 综合评分
"""

import os
import argparse
import logging
from pathlib import Path

import torch
import numpy as np

from model.driving.model_driving import MiniMindDriving, DrivingConfig
from data.driving_dataset import DrivingSFTDataset
from evaluate.driving_evaluator import DrivingEvaluator
from config.driving_config import DrivingConfig as AppConfig


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def evaluate_model(
    model_path: str,
    data_path: str,
    image_root: str = "./dataset/driving/raw/camera",
    output_dir: str = "./out/evaluation",
    device: str = "cuda",
    batch_size: int = 4,
    control_thresholds: dict = None,
    safety_thresholds: dict = None,
):
    """
    评估驾驶模型

    Args:
        model_path: 模型检查点路径
        data_path: 评估数据路径
        image_root: 图像根目录
        output_dir: 输出目录
        device: 设备
        batch_size: 批大小
        control_thresholds: 控制阈值
        safety_thresholds: 安全阈值

    Returns:
        评估结果
    """
    logger.info(f"Loading model from {model_path}")

    # 加载模型
    config = DrivingConfig()
    model = MiniMindDriving(config, vision_encoder_path="./model/vision_model/clip-vit-base-patch16")

    ckp_path = os.path.join(model_path, "driving_sft_512.pth")
    if not os.path.exists(ckp_path):
        # 尝试直接路径
        ckp_path = model_path

    weights = torch.load(ckp_path, map_location=device)
    if isinstance(weights, dict) and "model" in weights:
        model.load_state_dict(weights["model"], strict=False)
    else:
        model.load_state_dict(weights, strict=False)

    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded. Parameters: "
                f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")

    # 加载 tokenizer
    from transformers import AutoTokenizer
    tokenizer_path = os.path.join(os.path.dirname(__file__), "..", "model")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # 创建数据集
    dataset = DrivingSFTDataset(
        data_path=data_path,
        config=config,
        image_root=image_root,
        tokenizer=tokenizer,
        max_seq_len=config.max_position_embeddings,
    )

    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    logger.info(f"Evaluating on {len(dataset)} samples")

    # 创建评估器
    evaluator = DrivingEvaluator(
        control_thresholds=control_thresholds,
        safety_thresholds=safety_thresholds,
        device=device,
    )

    # 执行评估
    result = evaluator.evaluate(
        model=model,
        dataloader=dataloader,
        tokenizer=tokenizer,
        save_dir=output_dir,
    )

    # 打印报告
    report = evaluator.generate_report(result)
    print("\n" + report)

    # 保存详细报告
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "detailed_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
        f.write("\n\n--- 详细控制精度 ---\n")
        if result.control_metrics:
            f.write(f"  Steering MAE: {result.control_metrics.steering_mae:.4f}\n")
            f.write(f"  Throttle MAE: {result.control_metrics.throttle_mae:.4f}\n")
            f.write(f"  Brake MAE: {result.control_metrics.brake_mae:.4f}\n")

        f.write("\n--- 详细安全评分 ---\n")
        if result.safety_metrics:
            f.write(f"  Collision rate: {result.safety_metrics.collision_rate:.4%}\n")
            f.write(f"  Hard braking rate: {result.safety_metrics.hard_braking_rate:.4%}\n")
            f.write(f"  Comfort score: {result.safety_metrics.comfort_score:.1f}/100\n")

        f.write("\n--- 场景覆盖 ---\n")
        if result.coverage_report:
            f.write(f"  Coverage: {result.coverage_report.coverage_rate:.2%}\n")
            f.write(f"  Balance index: {result.coverage_report.balance_index:.3f}\n")

    logger.info(f"Detailed report saved to {report_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="评估驾驶模型")
    parser.add_argument("--model", type=str, default="./checkpoints",
                        help="模型检查点目录")
    parser.add_argument("--data", type=str, required=True,
                        help="评估数据路径 (JSONL)")
    parser.add_argument("--image-root", type=str,
                        default="./dataset/driving/raw/camera",
                        help="图像根目录")
    parser.add_argument("--output", type=str, default="./out/evaluation",
                        help="输出目录")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备 (cuda/cpu)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="批大小")
    parser.add_argument("--steering-threshold", type=float, default=2.0,
                        help="转向角阈值 (度)")
    parser.add_argument("--throttle-threshold", type=float, default=0.1,
                        help="油门阈值")
    parser.add_argument("--brake-threshold", type=float, default=0.1,
                        help="刹车阈值")

    args = parser.parse_args()

    control_thresholds = {
        "steering_deg": args.steering_threshold,
        "throttle_diff": args.throttle_threshold,
        "brake_diff": args.brake_threshold,
    }

    result = evaluate_model(
        model_path=args.model,
        data_path=args.data,
        image_root=args.image_root,
        output_dir=args.output,
        device=args.device,
        batch_size=args.batch_size,
        control_thresholds=control_thresholds,
    )

    print(f"\n综合评分: {result.overall_score:.1f}/100")


if __name__ == "__main__":
    main()
