"""
LoRA 配置
定义参数高效微调的 LoRA 参数
"""

from typing import List, Optional


class LoRAConfig:
    """
    LoRA 配置类
    定义 LoRA 微调的所有参数
    """

    # 默认 target modules (对应 MiniMind 模型中的层名)
    DEFAULT_TARGET_MODULES_ATTENTION = ["q_proj", "k_proj", "v_proj", "o_proj"]
    DEFAULT_TARGET_MODULES_MLP = ["gate_proj", "down_proj", "up_proj"]
    DEFAULT_TARGET_MODULES_ALL = DEFAULT_TARGET_MODULES_ATTENTION + DEFAULT_TARGET_MODULES_MLP

    def __init__(
        self,
        enable: bool = True,
        rank: int = 16,                  # LoRA 秩 (rank)
        alpha: int = 32,                 # LoRA alpha
        dropout: float = 0.05,           # LoRA dropout
        target_modules: str = "all_linear",  # "all_linear" / "attention" / "mlp" / ["q_proj", "k_proj"]
        layers_to_transform: Optional[List[int]] = None,  # 要应用 LoRA 的层索引 (None=全部)
        target_8bit: bool = False,       # 是否对 8bit 模型应用 LoRA
        target_4bit: bool = False,       # 是否对 4bit 模型应用 LoRA
    ):
        self.enable = enable
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules
        self.layers_to_transform = layers_to_transform

        if target_modules == "all_linear":
            self.resolved_modules = self.DEFAULT_TARGET_MODULES_ALL
        elif target_modules == "attention":
            self.resolved_modules = self.DEFAULT_TARGET_MODULES_ATTENTION
        elif target_modules == "mlp":
            self.resolved_modules = self.DEFAULT_TARGET_MODULES_MLP
        elif isinstance(target_modules, list):
            self.resolved_modules = target_modules
        else:
            raise ValueError(f"Unknown target_modules: {target_modules}")

        # 计算 LoRA 参数量比例 (相对于原 Linear 层)
        # LoRA 参数量 = 2 * in_features * rank (A 和 B 矩阵)
        # 原参数量 = 2 * in_features * out_features (weight 和 bias)
        # 比例 ≈ rank / out_features (假设 in_features ≈ out_features)
        self.parameter_ratio_estimate = rank / 512  # 假设 hidden_size=512

    def get_lora_params(self) -> dict:
        """获取 LoRA 参数字典 (用于传递给 nn.Linear)"""
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
        }

    def __repr__(self):
        return (f"LoRAConfig(enable={self.enable}, rank={self.rank}, alpha={self.alpha}, "
                f"dropout={self.dropout}, target={self.target_modules})")


def get_driving_lora_config() -> LoRAConfig:
    """获取自动驾驶领域推荐的 LoRA 配置"""
    return LoRAConfig(
        enable=True,
        rank=16,
        alpha=32,
        dropout=0.05,
        target_modules="attention",  # 只对注意力层应用 LoRA (推荐)
    )
