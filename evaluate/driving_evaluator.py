"""
驾驶评估器

综合评估自动驾驶模型的全面性能:
    - 控制精度
    - 决策准确性
    - 安全性
    - 场景覆盖
    - 生成文本质量
"""

import os
import json
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from evaluate.control_accuracy import ControlAccuracyEvaluator, ControlErrorMetrics
from evaluate.safety_metrics import SafetyEvaluator, SafetyMetrics
from evaluate.scene_coverage import SceneCoverageAnalyzer, SceneCoverageReport


@dataclass
class DrivingEvaluationResult:
    """驾驶评估综合结果"""
    # 控制精度
    control_metrics: Optional[ControlErrorMetrics] = None

    # 决策准确性
    action_accuracy: float = 0.0
    action_precision: float = 0.0
    action_recall: float = 0.0
    action_f1: float = 0.0

    # 安全性
    safety_metrics: Optional[SafetyMetrics] = None

    # 场景覆盖
    coverage_report: Optional[SceneCoverageReport] = None

    # 文本生成质量
    text_bleu: float = 0.0
    text_rouge: float = 0.0
    text_token_accuracy: float = 0.0

    # 综合评分
    overall_score: float = 0.0

    # 各场景详细结果
    per_scene_results: Dict[str, Dict] = field(default_factory=dict)

    # 样本数
    total_samples: int = 0
    valid_samples: int = 0


class DrivingEvaluator:
    """
    驾驶评估器

    综合评估自动驾驶端到端模型的性能
    """

    def __init__(
        self,
        control_thresholds: Optional[Dict[str, float]] = None,
        safety_thresholds: Optional[Dict[str, float]] = None,
        scene_categories: Optional[List[str]] = None,
        device: str = "cuda",
    ):
        self.control_evaluator = ControlAccuracyEvaluator(
            thresholds=control_thresholds
        )
        self.safety_evaluator = SafetyEvaluator(
            thresholds=safety_thresholds
        )
        self.scene_analyzer = SceneCoverageAnalyzer(
            standard_scenes=scene_categories
        )
        self.device = device

    def evaluate(
        self,
        model,
        dataloader,
        tokenizer=None,
        save_dir: Optional[str] = None,
    ) -> DrivingEvaluationResult:
        """
        全面评估模型

        Args:
            model: 评估的模型
            dataloader: 数据加载器
            tokenizer: 分词器
            save_dir: 结果保存目录

        Returns:
            DrivingEvaluationResult
        """
        model.eval()
        result = DrivingEvaluationResult()

        all_predicted_controls = []
        all_ground_truth_controls = []
        all_predicted_actions = []
        all_ground_truth_actions = []
        all_scenes = []
        all_trajectories = []
        all_speeds = []
        all_text_responses = []
        all_text_targets = []
        text_correct = 0
        text_total = 0

        with torch.no_grad():
            for batch in dataloader:
                # 提取数据
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                pixel_values = batch["pixel_values"].to(self.device)
                gt_controls = batch.get("control_labels")
                gt_actions = batch.get("action_labels")
                scenes = batch.get("scene", [])
                metadata = batch.get("metadata", {})

                # 前向传播
                model_inputs = {
                    key: batch[key].to(self.device)
                    for key in (
                        "lidar_pointcloud", "radar_data", "gps_imu",
                        "lidar_mask", "radar_mask", "gps_imu_mask",
                    ) if key in batch
                }
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    **model_inputs,
                )
                if outputs.logits.shape[1] > 1:
                    predicted_tokens = outputs.logits[:, :-1].argmax(dim=-1)
                    target_tokens = input_ids[:, 1:]
                    valid_tokens = attention_mask[:, 1:].bool()
                    text_correct += int(
                        (predicted_tokens[valid_tokens] == target_tokens[valid_tokens])
                        .sum().item()
                    )
                    text_total += int(valid_tokens.sum().item())

                # 提取预测控制
                if outputs.control_outputs is not None:
                    pred_controls = outputs.control_outputs["continuous"]
                    all_predicted_controls.append(pred_controls.cpu())

                    if gt_controls is not None:
                        control_mask = batch.get("control_label_mask")
                        if control_mask is not None:
                            pred_controls = pred_controls[control_mask.bool()]
                            all_predicted_controls[-1] = pred_controls.cpu()
                            gt_controls = gt_controls[control_mask.bool()]
                        gt_ctrl = gt_controls.to(self.device)
                        if gt_ctrl.dim() == 0:
                            gt_ctrl = gt_ctrl.unsqueeze(0)
                        all_ground_truth_controls.append(gt_ctrl.cpu())

                # 提取预测动作
                if outputs.control_outputs is not None and "discrete_action" in outputs.control_outputs:
                    pred_actions = outputs.control_outputs["discrete_action"]
                    all_predicted_actions.append(pred_actions.cpu())

                    if gt_actions is not None:
                        action_mask = batch.get("action_label_mask")
                        if action_mask is not None:
                            pred_actions = pred_actions[action_mask.bool()]
                            all_predicted_actions[-1] = pred_actions.cpu()
                            gt_actions = gt_actions[action_mask.bool()]
                        all_ground_truth_actions.append(gt_actions.cpu())

                # 收集场景标签
                if scenes:
                    all_scenes.extend(scenes)

                # 收集速度
                if isinstance(metadata, list):
                    for item in metadata:
                        ego = item.get("ego_state") or {}
                        all_speeds.append(float(ego.get("speed_kmh", 0.0)))

                result.total_samples += input_ids.shape[0]
                result.valid_samples += input_ids.shape[0]
        result.text_token_accuracy = (
            text_correct / text_total if text_total else 0.0
        )

        # 计算控制精度
        if all_predicted_controls and all_ground_truth_controls:
            pred_controls = torch.cat(all_predicted_controls, dim=0)
            gt_controls = torch.cat(all_ground_truth_controls, dim=0)
            result.control_metrics = self.control_evaluator.evaluate(
                pred_controls, gt_controls,
                torch.tensor(all_speeds) if all_speeds else None,
            )

        # 计算决策准确性
        if all_predicted_actions and all_ground_truth_actions:
            pred_actions = torch.cat(all_predicted_actions, dim=0)
            gt_actions = torch.cat(all_ground_truth_actions, dim=0)

            correct = (pred_actions == gt_actions).float().mean().item()
            result.action_accuracy = correct

            # 按类别计算 Precision/Recall/F1
            unique_actions = torch.unique(gt_actions)
            precisions = []
            recalls = []
            for action in unique_actions:
                pred_pos = (pred_actions == action)
                gt_pos = (gt_actions == action)
                tp = (pred_pos & gt_pos).float().sum().item()
                fp = (pred_pos & ~gt_pos).float().sum().item()
                fn = (~pred_pos & gt_pos).float().sum().item()

                precision = tp / (tp + fp + 1e-6)
                recall = tp / (tp + fn + 1e-6)
                precisions.append(precision)
                recalls.append(recall)

            result.action_precision = float(np.mean(precisions))
            result.action_recall = float(np.mean(recalls))
            f1 = 2 * result.action_precision * result.action_recall / \
                 (result.action_precision + result.action_recall + 1e-6)
            result.action_f1 = float(f1)

        # 计算安全评分
        if all_predicted_controls:
            pred_controls = torch.cat(all_predicted_controls, dim=0)
            speeds_tensor = torch.tensor(all_speeds) if all_speeds else None
            result.safety_metrics = self.safety_evaluator.evaluate(
                pred_controls, speeds=speeds_tensor,
                scene_types=all_scenes,
            )

        # 场景覆盖分析
        if all_scenes:
            result.coverage_report = self.scene_analyzer.analyze(all_scenes)

            # 按场景统计
            scene_groups = defaultdict(list)
            for i, scene in enumerate(all_scenes):
                scene_groups[scene].append(i)

            for scene, indices in scene_groups.items():
                if all_predicted_controls and all_ground_truth_controls:
                    pred = torch.cat(all_predicted_controls, dim=0)[indices]
                    gt = torch.cat(all_ground_truth_controls, dim=0)[indices]
                    result.per_scene_results[scene] = {
                        "count": len(indices),
                        "control_mae": float(torch.abs(pred - gt).mean().item()),
                    }

        # 综合评分
        result.overall_score = self._compute_overall_score(result)

        # 保存结果
        if save_dir:
            self._save_results(result, save_dir)

        return result

    def _compute_overall_score(self, result: DrivingEvaluationResult) -> float:
        """计算综合评分 (0-100)"""
        scores = []
        weights = []

        # 控制精度 (30%)
        if result.control_metrics:
            control_score = result.control_metrics.within_threshold_rate * 100
            scores.append(control_score)
            weights.append(0.30)

        # 决策准确性 (25%)
        if result.action_accuracy > 0:
            scores.append(result.action_accuracy * 100)
            weights.append(0.25)

        # 安全性 (25%)
        if result.safety_metrics:
            safety_score = result.safety_metrics.overall_safety_score
            scores.append(safety_score)
            weights.append(0.25)

        # 场景覆盖 (20%)
        if result.coverage_report:
            coverage_score = result.coverage_report.coverage_rate * 100
            scores.append(coverage_score)
            weights.append(0.20)

        if not scores:
            return 0.0

        return float(np.average(scores, weights=weights[:len(scores)]))

    def _save_results(self, result: DrivingEvaluationResult, save_dir: str):
        """保存评估结果"""
        os.makedirs(save_dir, exist_ok=True)

        report = self.generate_report(result)
        report_path = os.path.join(save_dir, "evaluation_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        # 保存 JSON 结果
        json_data = {
            "overall_score": result.overall_score,
            "total_samples": result.total_samples,
            "valid_samples": result.valid_samples,
            "action_accuracy": result.action_accuracy,
            "action_precision": result.action_precision,
            "action_recall": result.action_recall,
            "action_f1": result.action_f1,
            "text_token_accuracy": result.text_token_accuracy,
        }

        if result.control_metrics:
            json_data["control_metrics"] = {
                "overall_mae": result.control_metrics.overall_mae,
                "overall_rmse": result.control_metrics.overall_rmse,
                "within_threshold_rate": result.control_metrics.within_threshold_rate,
            }

        if result.safety_metrics:
            json_data["safety_metrics"] = {
                "overall_safety_score": result.safety_metrics.overall_safety_score,
                "collision_rate": result.safety_metrics.collision_rate,
                "hard_braking_rate": result.safety_metrics.hard_braking_rate,
                "comfort_score": result.safety_metrics.comfort_score,
            }

        if result.coverage_report:
            json_data["coverage"] = {
                "coverage_rate": result.coverage_report.coverage_rate,
                "balance_index": result.coverage_report.balance_index,
                "rare_scenes": result.coverage_report.rare_scenes,
            }

        json_path = os.path.join(save_dir, "evaluation_results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)

    def generate_report(self, result: DrivingEvaluationResult) -> str:
        """生成综合评估报告"""
        lines = [
            "=" * 50,
            "    自动驾驶端到端模型综合评估报告",
            "=" * 50,
            f"总样本数: {result.total_samples}",
            f"有效样本: {result.valid_samples}",
            f"综合评分: {result.overall_score:.1f}/100",
            "",
            "--- 决策准确性 ---",
            f"  Accuracy:  {result.action_accuracy:.2%}",
            f"  Precision: {result.action_precision:.2%}",
            f"  Recall:    {result.action_recall:.2%}",
            f"  F1:        {result.action_f1:.2%}",
        ]

        if result.control_metrics:
            lines.extend([
                "",
                "--- 控制精度 ---",
                f"  Overall MAE:     {result.control_metrics.overall_mae:.4f}",
                f"  Overall RMSE:    {result.control_metrics.overall_rmse:.4f}",
                f"  阈值内率:        {result.control_metrics.within_threshold_rate:.2%}",
            ])

        if result.safety_metrics:
            lines.extend([
                "",
                "--- 安全性 ---",
                f"  安全评分:    {result.safety_metrics.overall_safety_score:.1f}/100",
                f"  碰撞率:      {result.safety_metrics.collision_rate:.4%}",
                f"  急刹率:      {result.safety_metrics.hard_braking_rate:.4%}",
                f"  舒适性:      {result.safety_metrics.comfort_score:.1f}/100",
            ])

        if result.coverage_report:
            lines.extend([
                "",
                "--- 场景覆盖 ---",
                f"  覆盖率:      {result.coverage_report.coverage_rate:.2%}",
                f"  平衡指数:    {result.coverage_report.balance_index:.3f}",
            ])

        lines.extend(["", "=" * 50])
        return "\n".join(lines)
