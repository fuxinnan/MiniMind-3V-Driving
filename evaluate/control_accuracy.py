"""
控制精度评估器

评估模型输出的控制信号与真实控制信号的偏差:
    - 转向角误差 (Steering Angle Error)
    - 油门误差 (Throttle Error)
    - 刹车误差 (Brake Error)
    - 综合控制误差 (Combined Control Error)
    - 逐场景控制精度
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ControlErrorMetrics:
    """控制误差指标"""
    # 转向角误差
    steering_mae: float = 0.0       # 平均绝对误差 (度)
    steering_mse: float = 0.0       # 均方误差
    steering_rmse: float = 0.0      # 均方根误差
    steering_max_error: float = 0.0 # 最大误差

    # 油门误差
    throttle_mae: float = 0.0
    throttle_mse: float = 0.0
    throttle_rmse: float = 0.0
    throttle_max_error: float = 0.0

    # 刹车误差
    brake_mae: float = 0.0
    brake_mse: float = 0.0
    brake_rmse: float = 0.0
    brake_max_error: float = 0.0

    # 综合
    overall_mae: float = 0.0
    overall_rmse: float = 0.0
    within_threshold_rate: float = 0.0  # 在阈值内的比例

    # 样本数
    sample_count: int = 0


class ControlAccuracyEvaluator:
    """
    控制精度评估器

    评估模型预测的控制信号相对于真实控制信号的精度
    """

    # 控制误差阈值 (来自配置)
    DEFAULT_THRESHOLDS = {
        "steering_deg": 2.0,       # 转向角误差 < 2°
        "throttle_diff": 0.1,      # 油门误差 < 0.1
        "brake_diff": 0.1,         # 刹车误差 < 0.1
        "speed_kmh": 3.0,          # 速度误差 < 3 km/h
        "lat_offset_m": 0.3,       # 横向偏移 < 0.3m
        "long_offset_m": 0.5,      # 纵向偏移 < 0.5m
    }

    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None,
        steering_unit: str = "degrees",  # "degrees" / "normalized"
    ):
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS
        self.steering_unit = steering_unit

    def evaluate(
        self,
        predicted_controls: torch.Tensor,
        ground_truth_controls: torch.Tensor,
        speeds: Optional[torch.Tensor] = None,
    ) -> ControlErrorMetrics:
        """
        评估控制精度

        Args:
            predicted_controls: [N, 4] 预测的控制信号
                                [steering, throttle, brake, gear]
            ground_truth_controls: [N, 4] 真实控制信号
            speeds: [N] 当前车速 (km/h)

        Returns:
            ControlErrorMetrics
        """
        metrics = ControlErrorMetrics()
        metrics.sample_count = predicted_controls.shape[0]

        pred_steering = predicted_controls[:, 0]
        gt_steering = ground_truth_controls[:, 0]
        pred_throttle = predicted_controls[:, 1]
        gt_throttle = ground_truth_controls[:, 1]
        pred_brake = predicted_controls[:, 2]
        gt_brake = ground_truth_controls[:, 2]

        # 转向角误差
        steering_errors = torch.abs(pred_steering - gt_steering)
        if self.steering_unit == "degrees":
            steering_errors = steering_errors * 180.0 / np.pi

        metrics.steering_mae = float(steering_errors.mean())
        metrics.steering_mse = float(steering_errors.pow(2).mean())
        metrics.steering_rmse = float(torch.sqrt(metrics.steering_mse))
        metrics.steering_max_error = float(steering_errors.max())

        # 油门误差
        throttle_errors = torch.abs(pred_throttle - gt_throttle)
        metrics.throttle_mae = float(throttle_errors.mean())
        metrics.throttle_mse = float(throttle_errors.pow(2).mean())
        metrics.throttle_rmse = float(torch.sqrt(metrics.throttle_mse))
        metrics.throttle_max_error = float(throttle_errors.max())

        # 刹车误差
        brake_errors = torch.abs(pred_brake - gt_brake)
        metrics.brake_mae = float(brake_errors.mean())
        metrics.brake_mse = float(brake_errors.pow(2).mean())
        metrics.brake_rmse = float(torch.sqrt(metrics.brake_mse))
        metrics.brake_max_error = float(brake_errors.max())

        # 综合误差
        all_errors = torch.cat([
            steering_errors.unsqueeze(1),
            throttle_errors.unsqueeze(1),
            brake_errors.unsqueeze(1),
        ], dim=1)
        metrics.overall_mae = float(all_errors.mean())
        metrics.overall_rmse = float(torch.sqrt(
            (all_errors.pow(2).mean(dim=1)).mean()
        ))

        # 在阈值内的比例
        within_thresh = (
            (steering_errors < self.thresholds["steering_deg"]) &
            (throttle_errors < self.thresholds["throttle_diff"]) &
            (brake_errors < self.thresholds["brake_diff"])
        )
        metrics.within_threshold_rate = float(within_thresh.float().mean())

        return metrics

    def evaluate_per_scene(
        self,
        predicted_controls: torch.Tensor,
        ground_truth_controls: torch.Tensor,
        scene_labels: List[str],
    ) -> Dict[str, ControlErrorMetrics]:
        """
        按场景分组评估控制精度

        Args:
            predicted_controls: [N, 4]
            ground_truth_controls: [N, 4]
            scene_labels: [N] 场景标签列表

        Returns:
            {scene_name: ControlErrorMetrics}
        """
        scene_groups = {}
        for i, scene in enumerate(scene_labels):
            if scene not in scene_groups:
                scene_groups[scene] = {
                    "pred": [],
                    "gt": [],
                }
            scene_groups[scene]["pred"].append(predicted_controls[i])
            scene_groups[scene]["gt"].append(ground_truth_controls[i])

        results = {}
        for scene, groups in scene_groups.items():
            pred = torch.stack(groups["pred"])
            gt = torch.stack(groups["gt"])
            results[scene] = self.evaluate(pred, gt)

        return results

    def evaluate_control_consistency(
        self,
        predicted_controls: torch.Tensor,
    ) -> Dict[str, float]:
        """
        评估控制信号的一致性 (物理约束检查)

        Returns:
            {
                "throttle_brake_conflict_rate": float,
                "out_of_range_rate": float,
                "abrupt_change_rate": float,
            }
        """
        n = predicted_controls.shape[0]
        if n < 2:
            return {"throttle_brake_conflict_rate": 0.0, "out_of_range_rate": 0.0, "abrupt_change_rate": 0.0}

        # 油门和刹车冲突
        throttle = predicted_controls[:, 1]
        brake = predicted_controls[:, 2]
        conflicts = (throttle > 0.1) & (brake > 0.1)
        conflict_rate = float(conflicts.float().mean())

        # 超出范围
        steering = predicted_controls[:, 0]
        out_of_range = (
            torch.abs(steering) > 1.01 |
            throttle > 1.01 |
            brake > 1.01
        )
        oor_rate = float(out_of_range.float().mean())

        # 突变检测
        diffs = torch.diff(predicted_controls, dim=0)
        abrupt = torch.any(torch.abs(diffs) > 0.5, dim=1)
        abrupt_rate = float(abrupt.float().mean()) if n > 1 else 0.0

        return {
            "throttle_brake_conflict_rate": conflict_rate,
            "out_of_range_rate": oor_rate,
            "abrupt_change_rate": abrupt_rate,
        }

    def generate_report(
        self,
        metrics: ControlErrorMetrics,
        per_scene: Optional[Dict[str, ControlErrorMetrics]] = None,
    ) -> str:
        """生成控制精度评估报告"""
        report = (
            f"=== 控制精度评估报告 ===\n"
            f"样本数: {metrics.sample_count}\n\n"

            f"--- 转向角误差 ---\n"
            f"  MAE:  {metrics.steering_mae:.4f}\n"
            f"  RMSE: {metrics.steering_rmse:.4f}\n"
            f"  Max:  {metrics.steering_max_error:.4f}\n\n"

            f"--- 油门误差 ---\n"
            f"  MAE:  {metrics.throttle_mae:.4f}\n"
            f"  RMSE: {metrics.throttle_rmse:.4f}\n"
            f"  Max:  {metrics.throttle_max_error:.4f}\n\n"

            f"--- 刹车误差 ---\n"
            f"  MAE:  {metrics.brake_mae:.4f}\n"
            f"  RMSE: {metrics.brake_rmse:.4f}\n"
            f"  Max:  {metrics.brake_max_error:.4f}\n\n"

            f"--- 综合指标 ---\n"
            f"  Overall MAE:     {metrics.overall_mae:.4f}\n"
            f"  Overall RMSE:    {metrics.overall_rmse:.4f}\n"
            f"  阈值内率:         {metrics.within_threshold_rate:.2%}\n"
        )

        if per_scene:
            report += "\n--- 按场景统计 ---\n"
            for scene, scene_metrics in sorted(per_scene.items()):
                report += (
                    f"  [{scene}] MAE={scene_metrics.overall_mae:.4f}, "
                    f"RMSE={scene_metrics.overall_rmse:.4f}, "
                    f"阈值内={scene_metrics.within_threshold_rate:.2%}\n"
                )

        report += "\n========================"
        return report
