"""
安全指标评估器

评估驾驶决策的安全性:
    - 碰撞率 (Collision Rate)
    - 车道偏离 (Lane Deviation)
    - 交通规则违反 (Traffic Rule Violation)
    - 舒适性指标 (Comfort Metrics)
    - 急刹率 (Hard Braking Rate)
    - 加加速度 (Jerk)
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class SafetyMetrics:
    """安全指标集合"""
    # 碰撞相关
    collision_rate: float = 0.0             # 碰撞率
    near_miss_rate: float = 0.0             # 接近碰撞率
    min_ttc: float = float("inf")           # 最小碰撞时间 (Time to Collision)

    # 车道偏离
    lane_deviation_mae: float = 0.0         # 车道偏离平均距离 (m)
    lane_deviation_max: float = 0.0         # 最大偏离距离 (m)
    deviation_rate: float = 0.0             # 偏离率 (>0.5m 的比例)

    # 交通规则
    traffic_violation_count: int = 0        # 违规次数
    red_light_rate: float = 0.0             # 闯红灯率
    speed_limit_violation_rate: float = 0.0 # 超速率
    stop_sign_violation_rate: float = 0.0   # 停车标志违规率

    # 舒适性
    comfort_score: float = 0.0              # 综合舒适性评分 (0-100)
    avg_jerk: float = 0.0                   # 平均加加速度 (m/s^3)
    max_jerk: float = 0.0                   # 最大加加速度
    lateral_acceleration_mae: float = 0.0   # 横向加速度 MAE

    # 制动
    hard_braking_rate: float = 0.0          # 急刹率
    avg_brake_intensity: float = 0.0        # 平均刹车强度

    # 总体安全评分
    overall_safety_score: float = 0.0       # 综合安全评分 (0-100)
    sample_count: int = 0
    available_metrics: List[str] = field(default_factory=list)


class SafetyEvaluator:
    """
    安全指标评估器

    评估驾驶决策的安全性，提供多维度的安全评分
    """

    # 安全阈值
    THRESHOLDS = {
        "min_ttc_collision": 1.0,        # TTC < 1s 视为碰撞
        "min_ttc_near_miss": 3.0,        # TTC < 3s 视为接近碰撞
        "lane_deviation_safe": 0.3,      # 偏离 < 0.3m 安全
        "lane_deviation_warning": 0.5,   # 偏离 > 0.5m 警告
        "hard_brake_threshold": 0.8,     # 刹车 > 0.8 视为急刹
        "max_jerk_comfort": 2.0,         # 加加速度 > 2 m/s^3 不舒适
        "max_lateral_accel": 3.0,        # 横向加速度 > 3 m/s^2 不舒适
        "speed_limit_buffer": 5.0,       # 超速 > 5 km/h 违规
    }

    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        self.thresholds = thresholds or self.THRESHOLDS

    def evaluate(
        self,
        predicted_controls: torch.Tensor,
        ground_truth_controls: Optional[torch.Tensor] = None,
        trajectories: Optional[torch.Tensor] = None,
        speeds: Optional[torch.Tensor] = None,
        scene_types: Optional[List[str]] = None,
    ) -> SafetyMetrics:
        """
        综合安全评估

        Args:
            predicted_controls: [N, 4] 预测控制信号
            ground_truth_controls: [N, 4] 真实控制信号
            trajectories: [N, 3] 轨迹 (x, y, heading)
            speeds: [N] 速度序列 (km/h)
            scene_types: [N] 场景类型

        Returns:
            SafetyMetrics
        """
        metrics = SafetyMetrics()
        metrics.sample_count = predicted_controls.shape[0]

        # 1. 碰撞率评估
        if trajectories is not None:
            self._evaluate_collision_safety(trajectories, speeds, metrics)
            metrics.available_metrics.extend(["collision", "lane_deviation"])

        # 2. 车道偏离评估
        if trajectories is not None:
            self._evaluate_lane_deviation(trajectories, metrics)

        # 3. 交通规则评估
        self._evaluate_traffic_rules(predicted_controls, speeds, metrics)
        metrics.available_metrics.append("control_consistency")

        # 4. 舒适性评估
        if speeds is not None:
            self._evaluate_comfort(speeds, predicted_controls, metrics)
            metrics.available_metrics.append("comfort")

        # 5. 制动评估
        self._evaluate_braking(predicted_controls, metrics)
        metrics.available_metrics.append("braking")

        # 6. 综合安全评分
        metrics.overall_safety_score = self._compute_overall_score(metrics)

        return metrics

    def _evaluate_collision_safety(
        self,
        trajectories: torch.Tensor,
        speeds: Optional[torch.Tensor],
        metrics: SafetyMetrics,
    ):
        """评估碰撞安全"""
        if trajectories is None or trajectories.shape[0] < 2:
            return

        # 简化 TTC 计算 (基于位置变化)
        positions = trajectories[:, :2]  # [x, y]
        dt = 0.1  # 假设时间间隔 0.1s

        # 计算相邻点的距离变化
        diffs = torch.diff(positions, dim=0)
        distances = torch.norm(diffs, dim=1)

        # 简化: 如果距离突然减小，可能接近碰撞
        speed_changes = torch.diff(distances) / dt if speeds is not None else None

        # 最小 TTC (简化估计)
        if speeds is not None:
            relative_speeds = torch.abs(speed_changes) if speed_changes is not None else torch.zeros_like(distances)
            ttc = distances / (relative_speeds + 1e-6)
            metrics.min_ttc = float(torch.min(ttc).item())

            # 碰撞率 (TTC < 阈值)
            collision_mask = ttc < self.thresholds["min_ttc_collision"]
            metrics.collision_rate = float(collision_mask.float().mean())

            # 接近碰撞率
            near_miss_mask = (ttc < self.thresholds["min_ttc_near_miss"]) & \
                             (ttc >= self.thresholds["min_ttc_collision"])
            metrics.near_miss_rate = float(near_miss_mask.float().mean())

    def _evaluate_lane_deviation(
        self,
        trajectories: torch.Tensor,
        metrics: SafetyMetrics,
    ):
        """评估车道偏离"""
        if trajectories is None or trajectories.shape[0] < 2:
            return

        # 简化: 基于横向偏移计算
        positions = trajectories[:, :2]
        diffs = torch.diff(positions, dim=0)
        lateral_offsets = torch.abs(diffs[:, 1])  # 假设 y 方向为横向

        if lateral_offsets.numel() > 0:
            metrics.lane_deviation_mae = float(lateral_offsets.mean().item())
            metrics.lane_deviation_max = float(lateral_offsets.max().item())

            dev_mask = lateral_offsets > self.thresholds["lane_deviation_warning"]
            metrics.deviation_rate = float(dev_mask.float().mean())

    def _evaluate_traffic_rules(
        self,
        controls: torch.Tensor,
        speeds: Optional[torch.Tensor],
        metrics: SafetyMetrics,
    ):
        """评估交通规则遵守情况"""
        # 简化: 基于控制信号推断规则遵守
        n = controls.shape[0]

        # 急加速/急减速可能违反平稳驾驶规则
        if controls.shape[1] >= 2:
            throttle = controls[:, 1]
            brake = controls[:, 2]

            # 同时油门和刹车
            violations = (throttle > 0.1) & (brake > 0.1)
            metrics.traffic_violation_count = int(violations.sum().item())

        # 超速评估 (简化)
        if speeds is not None:
            # 假设限速 60 km/h
            speed_limit = 60.0
            speed_violations = speeds > (speed_limit + self.thresholds["speed_limit_buffer"])
            metrics.speed_limit_violation_rate = float(
                speed_violations.float().mean()
            )

    def _evaluate_comfort(
        self,
        speeds: torch.Tensor,
        controls: torch.Tensor,
        metrics: SafetyMetrics,
    ):
        """评估驾驶舒适性"""
        if speeds.numel() < 2:
            return

        # 将速度转换为 m/s
        speeds_ms = speeds / 3.6

        # 计算加速度
        accelerations = torch.diff(speeds_ms) / 0.1  # a = dv/dt

        # 计算加加速度 (Jerk)
        jerks = torch.diff(accelerations) / 0.1  # j = da/dt

        if jerks.numel() > 0:
            metrics.avg_jerk = float(torch.abs(jerks).mean().item())
            metrics.max_jerk = float(torch.abs(jerks).max().item())

        # 横向加速度 (基于转向和速度)
        if controls.shape[1] >= 1 and speeds.numel() > 0:
            steering = controls[:, 0]
            speed_squared = speeds_ms ** 2
            lateral_accel = torch.abs(steering) * speed_squared / 10.0  # 简化公式
            metrics.lateral_acceleration_mae = float(lateral_accel.mean().item())

        # 综合舒适性评分
        comfort_jerk = max(0, 100 - metrics.avg_jerk * 20)
        comfort_lateral = max(0, 100 - metrics.lateral_acceleration_mae * 15)
        metrics.comfort_score = (comfort_jerk * 0.6 + comfort_lateral * 0.4)

    def _evaluate_braking(
        self,
        controls: torch.Tensor,
        metrics: SafetyMetrics,
    ):
        """评估制动行为"""
        if controls.shape[1] < 3:
            return

        brake = controls[:, 2]
        metrics.avg_brake_intensity = float(brake.mean().item())

        # 急刹率
        hard_brake_mask = brake > self.thresholds["hard_brake_threshold"]
        metrics.hard_braking_rate = float(hard_brake_mask.float().mean())

    def _compute_overall_score(self, metrics: SafetyMetrics) -> float:
        """
        计算综合安全评分 (0-100)

        权重:
            - 碰撞安全: 40%
            - 车道偏离: 20%
            - 规则遵守: 20%
            - 舒适性: 20%
        """
        # 碰撞安全评分
        scores = []
        weights = []
        if "collision" in metrics.available_metrics:
            collision_score = max(0, 100 * (1 - metrics.collision_rate * 10))
            collision_score = max(
                0, collision_score - metrics.near_miss_rate * 20
            )
            scores.extend([
                collision_score,
                max(0, 100 * (1 - metrics.deviation_rate)),
            ])
            weights.extend([0.4, 0.2])
        violation_penalty = min(50, metrics.traffic_violation_count * 5)
        rule_score = max(0, 100 - violation_penalty)
        scores.append(rule_score)
        weights.append(0.2)
        if "comfort" in metrics.available_metrics:
            scores.append(metrics.comfort_score)
            weights.append(0.2)
        return float(np.average(scores, weights=weights)) if scores else 0.0

    def evaluate_per_scene(
        self,
        predicted_controls: torch.Tensor,
        trajectories: Optional[torch.Tensor] = None,
        speeds: Optional[torch.Tensor] = None,
        scene_types: Optional[List[str]] = None,
    ) -> Dict[str, SafetyMetrics]:
        """按场景分组评估安全指标"""
        if scene_types is None:
            return {"all": self.evaluate(predicted_controls, trajectories=trajectories, speeds=speeds)}

        scene_groups = defaultdict(lambda: {
            "pred": [], "traj": [], "speed": []
        })

        for i, scene in enumerate(scene_types):
            scene_groups[scene]["pred"].append(predicted_controls[i])
            if trajectories is not None:
                scene_groups[scene]["traj"].append(trajectories[i])
            if speeds is not None:
                scene_groups[scene]["speed"].append(speeds[i])

        results = {}
        for scene, groups in scene_groups.items():
            pred = torch.stack(groups["pred"])
            traj = torch.stack(groups["traj"]) if groups["traj"] else None
            spd = torch.stack(groups["speed"]) if groups["speed"] else None
            results[scene] = self.evaluate(pred, trajectories=traj, speeds=spd)

        return results

    def generate_report(self, metrics: SafetyMetrics) -> str:
        """生成安全评估报告"""
        return (
            f"=== 安全评分报告 ===\n"
            f"样本数: {metrics.sample_count}\n"
            f"综合安全评分: {metrics.overall_safety_score:.1f}/100\n\n"

            f"--- 碰撞安全 ---\n"
            f"  碰撞率:         {metrics.collision_rate:.4%}\n"
            f"  接近碰撞率:     {metrics.near_miss_rate:.4%}\n"
            f"  最小 TTC:       {metrics.min_ttc:.2f}s\n\n"

            f"--- 车道偏离 ---\n"
            f"  偏离 MAE:       {metrics.lane_deviation_mae:.3f}m\n"
            f"  偏离 Max:       {metrics.lane_deviation_max:.3f}m\n"
            f"  偏离率:         {metrics.deviation_rate:.4%}\n\n"

            f"--- 规则遵守 ---\n"
            f"  违规次数:       {metrics.traffic_violation_count}\n"
            f"  超速率:         {metrics.speed_limit_violation_rate:.4%}\n\n"

            f"--- 舒适性 ---\n"
            f"  综合评分:       {metrics.comfort_score:.1f}/100\n"
            f"  平均 Jerk:      {metrics.avg_jerk:.2f} m/s^3\n"
            f"  最大 Jerk:      {metrics.max_jerk:.2f} m/s^3\n"
            f"  横向加速度 MAE: {metrics.lateral_acceleration_mae:.2f} m/s^2\n\n"

            f"--- 制动 ---\n"
            f"  急刹率:         {metrics.hard_braking_rate:.4%}\n"
            f"  平均刹车强度:   {metrics.avg_brake_intensity:.3f}\n"
            f"===================="
        )
