"""
基础配置 - 模型架构与训练超参数
不依赖任何特定领域，提供通用默认值
"""

from typing import Optional, List
import os


class BaseConfig:
    """
    基础配置类
    统一管理模型参数、训练超参数、硬件配置、日志配置
    """

    # ========== 模型架构参数 ==========
    # 这些参数决定了模型的大小和容量
    # 修改这些参数会改变模型结构，需要重新训练

    # 基础 Transformer 参数
    hidden_size: int = 512              # 隐藏层维度 (d_model)
    num_hidden_layers: int = 8          # Transformer 层数
    num_attention_heads: int = 8        # 注意力头数
    num_key_value_heads: Optional[int] = None  # GQA 的 KV 头数 (None = 不使用 GQA)
    intermediate_size: Optional[int] = None  # FFN 中间层维度 (None = hidden_size * 8/3)

    # 词表与序列
    vocab_size: int = 6400              # 词表大小
    max_position_embeddings: int = 32768  # 最大位置编码长度

    # 归一化与激活
    hidden_act: str = "silu"            # 激活函数 (silu / gelu / relu)
    rms_norm_eps: float = 1e-5          # RMSNorm epsilon
    dropout: float = 0.0                # Dropout 概率

    # 位置编码
    rope_theta: float = 1e6             # RoPE 基频
    inference_rope_scaling: bool = False  # 推理时是否开启 RoPE 外推 (4倍)

    # 硬件加速
    flash_attn: bool = True             # 是否使用 Flash Attention 2

    # MoE 配置
    use_moe: bool = False               # 是否启用 MoE
    num_experts_per_tok: int = 2        # 每个 token 选择的专家数 (Top-K)
    n_routed_experts: int = 4           # 专家总数
    n_shared_experts: int = 1           # 共享专家数
    aux_loss_alpha: float = 0.01        # MoE 辅助损失系数
    norm_topk_prob: bool = True         # 是否对 Top-K 概率归一化

    # ========== 训练超参数 ==========
    # 这些参数控制训练过程，可以在不同实验中调整

    # 优化器
    optimizer: str = "adamw"            # 优化器类型 (adamw / adam / sgd)
    weight_decay: float = 0.01          # 权重衰减
    adam_beta1: float = 0.9             # Adam beta1
    adam_beta2: float = 0.999           # Adam beta2
    adam_epsilon: float = 1e-8          # Adam epsilon

    # 学习率调度
    lr_scheduler: str = "cosine"        # 学习率调度 (cosine / linear / warmup_cosine)
    warmup_ratio: float = 0.05          # Warmup 比例 (占总步数的比例)
    warmup_steps: int = 0               # Warmup 步数 (与 warmup_ratio 二选一，优先使用 warmup_ratio)
    min_lr_ratio: float = 0.1            # 学习率最低值 / 初始学习率

    # 梯度控制
    grad_clip: float = 1.0              # 梯度裁剪阈值
    accumulation_steps: int = 1         # 梯度累积步数

    # 混合精度
    dtype: str = "bfloat16"             # 混合精度类型 (float32 / float16 / bfloat16)

    # ========== 数据配置 ==========
    batch_size: int = 16                # 每个 GPU 的 batch size
    num_workers: int = 4                # DataLoader 工作进程数
    max_seq_len: int = 512              # 最大序列长度
    pin_memory: bool = True             # DataLoader pin_memory

    # ========== 日志与检查点 ==========
    log_interval: int = 100             # 日志打印间隔 (步数)
    save_interval: int = 500            # 检查点保存间隔 (步数)
    eval_interval: int = 500            # 评估间隔 (步数)
    save_dir: str = "out"               # 模型权重保存目录
    checkpoint_dir: str = "checkpoints" # 训练检查点保存目录

    # 实验追踪
    use_wandb: bool = False             # 是否使用 Weights & Biases
    wandb_project: str = "MiniMind"     # wandb 项目名称
    wandb_run_name: Optional[str] = None  # wandb run 名称

    # ========== 分布式训练 ==========
    use_ddp: bool = False               # 是否使用 DDP 分布式训练
    world_size: int = 1                 # 总进程数 (GPU 数)
    rank: int = 0                       # 当前进程 rank
    local_rank: int = 0                 # 当前进程 local rank
    master_addr: str = "localhost"      # 主节点地址
    master_port: int = 29500            # 主节点端口

    # ========== 设备 ==========
    device: str = "cuda"                # 设备 (cuda / cpu)
    seed: int = 42                      # 随机种子

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config key: {key}")

    def to_dict(self):
        """将配置转换为字典，用于保存和序列化"""
        return {
            key: getattr(self, key)
            for key in dir(self)
            if not key.startswith("_") and not callable(getattr(self, key))
        }

    @classmethod
    def from_dict(cls, config_dict: dict):
        """从字典创建配置对象"""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})

    def __repr__(self):
        items = []
        for key in dir(self):
            if not key.startswith("_") and not callable(getattr(self, key)):
                items.append(f"{key}={getattr(self, key)}")
        return f"BaseConfig({', '.join(items)})"
