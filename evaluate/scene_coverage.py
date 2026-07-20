"""
场景覆盖率分析器

分析训练数据中的场景分布和覆盖情况:
    - 场景类别分布
    - 场景覆盖率
    - 场景不平衡度
    - 场景采样策略建议
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict
from dataclasses import dataclass, field


@dataclass
class SceneDistribution:
    """场景分布"""
    scene_name: str
    count: int
    percentage: float
    is_balanced: bool = False
    sample_ratio: float = 1.0  # 建议的采样比例


@dataclass
class SceneCoverageReport:
    """场景覆盖率报告"""
    total_samples: int = 0
    unique_scenes: int = 0
    scene_distribution: List[SceneDistribution] = field(default_factory=list)
    coverage_rate: float = 0.0  # 覆盖的场景比例
    balance_index: float = 0.0  # 平衡指数 (0-1)
    rare_scenes: List[str] = field(default_factory=list)
    dominant_scenes: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


MIN_SAMPLES_PER_SCENE = 1000       # 每场景最少样本数
BALANCED_SAMPLES_PER_SCENE = 5000  # 平衡时的每场景样本数


class SceneCoverageAnalyzer:
    """
    场景覆盖率分析器

    分析训练数据的场景分布，识别不足和过采样场景
    """

    # 标准场景分类
    STANDARD_SCENES = [
        "highway", "urban", "suburban", "intersection", "roundabout",
        "parking", "tunnel", "construction", "emergency", "pedestrian_cross",
        "school_zone", "residential", "ringroad", "onramp_offramp",
    ]

    def __init__(
        self,
        standard_scenes: Optional[List[str]] = None,
        min_samples: int = MIN_SAMPLES_PER_SCENE,
        balanced_samples: int = BALANCED_SAMPLES_PER_SCENE,
    ):
        self.standard_scenes = standard_scenes or self.STANDARD_SCENES
        self.min_samples = min_samples
        self.balanced_samples = balanced_samples

    def analyze(
        self,
        scene_labels: List[str],
    ) -> SceneCoverageReport:
        """
        分析场景覆盖率

        Args:
            scene_labels: 场景标签列表

        Returns:
            SceneCoverageReport
        """
        report = SceneCoverageReport()
        report.total_samples = len(scene_labels)

        # 统计场景分布
        counter = Counter(scene_labels)
        report.unique_scenes = len(counter)

        # 计算每个场景的分布
        total = len(scene_labels)
        distribution = []
        for scene in self.standard_scenes:
            count = counter.get(scene, 0)
            percentage = count / total if total > 0 else 0.0
            is_balanced = count >= self.min_samples

            # 计算建议采样比例
            if count < self.min_samples:
                sample_ratio = self.min_samples / max(count, 1)
            elif count > self.balanced_samples:
                sample_ratio = self.balanced_samples / max(count, 1)
            else:
                sample_ratio = 1.0

            distribution.append(SceneDistribution(
                scene_name=scene,
                count=count,
                percentage=percentage,
                is_balanced=is_balanced,
                sample_ratio=sample_ratio,
            ))

        report.scene_distribution = distribution

        # 覆盖率
        covered_scenes = sum(1 for d in distribution if d.count > 0)
        report.coverage_rate = covered_scenes / len(self.standard_scenes)

        # 平衡指数 (基于熵)
        report.balance_index = self._compute_balance_index(counter, total)

        # 稀有场景 (< min_samples)
        report.rare_scenes = [
            d.scene_name for d in distribution
            if 0 < d.count < self.min_samples
        ]

        # 主导场景 (> balanced_samples)
        report.dominant_scenes = [
            d.scene_name for d in distribution
            if d.count > self.balanced_samples
        ]

        # 生成建议
        report.recommendations = self._generate_recommendations(
            distribution, counter, total
        )

        return report

    def _compute_balance_index(
        self,
        counter: Counter,
        total: int,
    ) -> float:
        """
        计算场景平衡指数 (基于归一化熵)

        Returns:
            0-1 之间的值，1 表示完全平衡
        """
        if total == 0 or len(counter) == 0:
            return 0.0

        # 计算香农熵
        probs = np.array([count / total for count in counter.values()])
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs))

        # 最大熵 (均匀分布)
        max_entropy = np.log(len(counter))

        return float(entropy / max_entropy) if max_entropy > 0 else 0.0

    def _generate_recommendations(
        self,
        distribution: List[SceneDistribution],
        counter: Counter,
        total: int,
    ) -> List[str]:
        """生成数据收集建议"""
        recommendations = []

        # 稀有场景建议
        rare = [d for d in distribution if 0 < d.count < self.min_samples]
        if rare:
            scene_names = ", ".join([d.scene_name for d in rare])
            recommendations.append(
                f"需要增加以下稀有场景的数据: {scene_names}"
            )

        # 缺失场景建议
        missing = [d for d in distribution if d.count == 0]
        if missing:
            scene_names = ", ".join([d.scene_name for d in missing])
            recommendations.append(
                f"以下场景完全缺失，需要收集数据: {scene_names}"
            )

        # 主导场景建议
        dominant = [d for d in distribution if d.count > self.balanced_samples]
        if dominant:
            scene_names = ", ".join([d.scene_name for d in dominant])
            recommendations.append(
                f"以下场景数据过量，建议过采样平衡: {scene_names}"
            )

        # 平衡建议
        balance_idx = self._compute_balance_index(counter, total)
        if balance_idx < 0.5:
            recommendations.append(
                "场景分布严重不平衡，建议采用分层采样或数据增强"
            )
        elif balance_idx < 0.8:
            recommendations.append(
                "场景分布有一定不平衡，建议适当调整采样比例"
            )

        return recommendations

    def get_sampling_weights(
        self,
        scene_labels: List[str],
        strategy: str = "inverse_frequency",  # "uniform" / "inverse_frequency" / "balanced"
    ) -> np.ndarray:
        """
        计算场景采样权重

        Args:
            scene_labels: 场景标签列表
            strategy: 采样策略

        Returns:
            采样权重数组 [N]
        """
        counter = Counter(scene_labels)
        total = len(scene_labels)
        n_scenes = len(counter)

        weights = np.ones(total, dtype=np.float32)

        if strategy == "uniform":
            weights[:] = 1.0 / total

        elif strategy == "inverse_frequency":
            for i, scene in enumerate(scene_labels):
                weights[i] = total / (n_scenes * counter[scene])

        elif strategy == "balanced":
            target_per_scene = self.balanced_samples
            for scene, count in counter.items():
                weight = target_per_scene / max(count, 1)
                weight = min(weight, 10.0)  # 限制最大权重
                mask = [i for i, s in enumerate(scene_labels) if s == scene]
                weights[mask] = weight

        # 归一化
        weights = weights / weights.sum()

        return weights

    def generate_report(self, report: SceneCoverageReport) -> str:
        """生成场景覆盖率报告"""
        lines = [
            "=== 场景覆盖率报告 ===",
            f"总样本数: {report.total_samples}",
            f"唯一场景数: {report.unique_scenes}/{len(self.standard_scenes)}",
            f"场景覆盖率: {report.coverage_rate:.2%}",
            f"平衡指数: {report.balance_index:.3f}",
            "",
            "--- 场景分布 ---",
        ]

        for dist in sorted(report.scene_distribution, key=lambda x: x.count, reverse=True):
            bar = "#" * int(dist.percentage * 50)
            status = "OK" if dist.is_balanced else ("MISSING" if dist.count == 0 else "LOW")
            lines.append(
                f"  [{status:5s}] {dist.scene_name:20s} {dist.count:6d} "
                f"({dist.percentage:.1%}) {bar}"
            )

        if report.rare_scenes:
            lines.append(f"\n--- 稀有场景 (<{self.min_samples} 样本) ---")
            for scene in report.rare_scenes:
                count = sum(1 for s in report.scene_distribution if s.scene_name == scene and s.count > 0)
                lines.append(f"  - {scene}")

        if report.recommendations:
            lines.append("\n--- 建议 ---")
            for rec in report.recommendations:
                lines.append(f"  * {rec}")

        lines.append("\n======================")
        return "\n".join(lines)
