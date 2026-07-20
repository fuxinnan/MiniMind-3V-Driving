"""
批量推理工具

支持大规模驾驶场景的批量推理:
    - 从数据文件读取场景
    - 批量前向传播
    - 结果保存为 JSON/CSV
    - 进度显示
"""

import os
import json
import csv
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm

from model.driving.model_driving import MiniMindDriving, DrivingConfig
from data.driving_dataset import DrivingSFTDataset
from config.driving_config import DrivingConfig as AppConfig


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class BatchDrivingInference:
    """
    批量驾驶推理器

    对大规模数据集进行批量推理，输出控制信号和决策文本
    """

    def __init__(
        self,
        model_path: str = "./checkpoints",
        model_filename: str = "driving_sft_512.pth",
        device: str = "cuda",
        batch_size: int = 4,
        vision_encoder_path: str = "./model/vision_model/clip-vit-base-patch16",
    ):
        self.device = device
        self.batch_size = batch_size
        self.vision_encoder_path = vision_encoder_path

        # 加载模型
        self.model = self._load_model(model_path, model_filename)
        self.model.eval()

        # 加载 tokenizer
        from transformers import AutoTokenizer
        tokenizer_path = os.path.join(os.path.dirname(__file__), "..", "model")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def _load_model(self, model_path: str, model_filename: str) -> MiniMindDriving:
        """加载驾驶模型"""
        ckp_path = os.path.join(model_path, model_filename)

        if not os.path.exists(ckp_path):
            logger.warning(f"Checkpoint not found: {ckp_path}")
            config = DrivingConfig()
            return MiniMindDriving(config, vision_encoder_path=self.vision_encoder_path)

        config = DrivingConfig()
        model = MiniMindDriving(config, vision_encoder_path=self.vision_encoder_path)

        weights = torch.load(ckp_path, map_location=self.device)
        if isinstance(weights, dict) and "model" in weights:
            model.load_state_dict(weights["model"], strict=False)
        else:
            model.load_state_dict(weights, strict=False)

        logger.info(f"Model loaded: {ckp_path}")
        return model.to(self.device)

    def run(
        self,
        data_path: str,
        output_path: str,
        image_root: str = "./dataset/driving/raw/camera",
        prompt_template: str = "请分析当前驾驶场景并给出决策",
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        save_images: bool = False,
    ) -> Dict:
        """
        执行批量推理

        Args:
            data_path: 数据文件路径 (JSONL)
            output_path: 输出文件路径
            image_root: 图像根目录
            prompt_template: 提示模板
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            save_images: 是否保存图像引用

        Returns:
            推理统计信息
        """
        # 创建数据集
        dataset = DrivingSFTDataset(
            data_path=data_path,
            config=self.model.config,
            image_root=image_root,
            tokenizer=self.tokenizer,
            max_seq_len=self.model.config.max_position_embeddings,
        )

        from torch.utils.data import DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        results = []
        stats = {
            "total": len(dataset),
            "success": 0,
            "failed": 0,
            "total_latency_ms": 0.0,
        }

        logger.info(f"Starting batch inference: {len(dataset)} samples, "
                     f"batch_size={self.batch_size}")

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="推理")):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                pixel_values = batch["pixel_values"].to(self.device)
                scenes = batch.get("scene", [None] * input_ids.shape[0])

                batch_start = time.time()

                # 前向传播
                try:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                    )

                    # 生成文本
                    generated_ids = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                    # 提取控制输出
                    control_outputs = {}
                    if outputs.control_outputs is not None:
                        ctrl = outputs.control_outputs["continuous"][0].cpu().numpy()
                        control_outputs = {
                            "steering": float(ctrl[0]),
                            "throttle": float(ctrl[1]),
                            "brake": float(ctrl[2]),
                            "gear": int(round((ctrl[3] + 1) / 2 * 4)),
                        }

                        # 离散动作
                        if "discrete_action" in outputs.control_outputs:
                            action_idx = outputs.control_outputs["discrete_action"][0].item()
                            control_outputs["action"] = \
                                self.model.config.discrete_actions[action_idx]
                            control_outputs["action_probs"] = \
                                outputs.control_outputs["discrete_probs"][0].cpu().numpy().tolist()

                    # 解码文本
                    text_responses = []
                    for i in range(input_ids.shape[0]):
                        text = self.tokenizer.decode(
                            generated_ids[i], skip_special_tokens=True
                        )
                        text_responses.append(text)

                    batch_latency = (time.time() - batch_start) * 1000

                    # 收集结果
                    for i in range(input_ids.shape[0]):
                        results.append({
                            "scene": scenes[i] if isinstance(scenes, list) else scenes,
                            "text_response": text_responses[i],
                            "control": control_outputs if i == 0 else {
                                k: (v[i] if isinstance(v, list) else v)
                                for k, v in control_outputs.items()
                                if k != "action_probs"
                            },
                            "latency_ms": batch_latency / input_ids.shape[0],
                        })

                    stats["success"] += input_ids.shape[0]
                    stats["total_latency_ms"] += batch_latency

                except Exception as e:
                    logger.error(f"Batch {batch_idx} failed: {str(e)}")
                    stats["failed"] += input_ids.shape[0]

        # 保存结果
        self._save_results(results, output_path)

        # 计算统计
        if stats["success"] > 0:
            stats["avg_latency_ms"] = (
                stats["total_latency_ms"] / stats["success"]
            )
            stats["throughput"] = stats["success"] / (
                stats["total_latency_ms"] / 1000
            )

        logger.info(f"Batch inference completed:")
        logger.info(f"  Total: {stats['total']}")
        logger.info(f"  Success: {stats['success']}")
        logger.info(f"  Failed: {stats['failed']}")
        logger.info(f"  Avg latency: {stats.get('avg_latency_ms', 0):.1f} ms/sample")
        logger.info(f"  Throughput: {stats.get('throughput', 0):.1f} samples/s")

        return stats

    def _save_results(self, results: List[Dict], output_path: str):
        """保存推理结果"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存 JSON
        json_path = output_path.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        # 保存 CSV (控制信号)
        csv_path = output_path.with_suffix(".csv")
        if results:
            # 展开控制信号
            flat_results = []
            for r in results:
                flat = {
                    "scene": r.get("scene", ""),
                    "text_response": r.get("text_response", ""),
                    "latency_ms": r.get("latency_ms", 0),
                }
                ctrl = r.get("control", {})
                flat.update(ctrl)
                flat_results.append(flat)

            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=flat_results[0].keys())
                writer.writeheader()
                writer.writerows(flat_results)

        logger.info(f"Results saved to {json_path} and {csv_path}")


def main():
    """批量推理入口"""
    import argparse

    parser = argparse.ArgumentParser(description="驾驶模型批量推理")
    parser.add_argument("--data", type=str, required=True,
                        help="数据文件路径 (JSONL)")
    parser.add_argument("--output", type=str, default="./out/driving_inference",
                        help="输出路径")
    parser.add_argument("--model", type=str, default="./checkpoints",
                        help="模型检查点目录")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="批大小")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备 (cuda/cpu)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="采样温度")
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="最大生成 token 数")

    args = parser.parse_args()

    inferencer = BatchDrivingInference(
        model_path=args.model,
        device=args.device,
        batch_size=args.batch_size,
    )

    stats = inferencer.run(
        data_path=args.data,
        output_path=args.output,
        temperature=args.temperature,
        max_new_tokens=args.max_tokens,
    )

    print(f"\n推理完成: {stats['success']}/{stats['total']} 成功")


if __name__ == "__main__":
    main()
