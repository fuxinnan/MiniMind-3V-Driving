"""
控制输出头

将 LLM 的 hidden state 映射到控制信号
支持:
    - 连续控制: 回归 [转向, 油门, 刹车, 挡位]
    - 离散控制: 分类 [保持车道, 左转, 右转, 停车, ...]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple


class ControlHead(nn.Module):
    """
    控制输出头

    架构:
        输入: LLM 最后一个 token 的 hidden state [B, hidden_size]
        输出:
            - continuous: 连续控制信号 [B, 4]
            - discrete_logits: 离散动作 logits [B, num_actions] (可选)
            - discrete_probs: 离散动作概率 [B, num_actions] (可选)

    设计考虑:
        - 连续控制使用 Tanh 激活，输出归一化到 [-1, 1]
        - 离散控制使用 Softmax，输出概率分布
        - 两个头共享部分 MLP 层，最后分叉
    """

    def __init__(
        self,
        hidden_size: int = 512,
        continuous_dims: int = 4,
        discrete_actions: Optional[List[str]] = None,
        control_hidden_size: int = 256,
    ):
        super().__init__()
        self.continuous_dims = continuous_dims
        self.discrete_actions = discrete_actions or []
        self.num_discrete = len(self.discrete_actions)
        self.control_hidden_size = control_hidden_size

        # 共享 MLP 层
        self.shared_mlp = nn.Sequential(
            nn.Linear(hidden_size, control_hidden_size),
            nn.GELU(),
            nn.LayerNorm(control_hidden_size),
            nn.Dropout(0.1),
            nn.Linear(control_hidden_size, control_hidden_size // 2),
            nn.GELU(),
            nn.LayerNorm(control_hidden_size // 2),
        )

        # 连续控制头
        self.continuous_proj = nn.Linear(control_hidden_size // 2, 3)
        self.gear_proj = nn.Linear(control_hidden_size // 2, 5)
        # 离散控制头
        if self.num_discrete > 0:
            self.discrete_proj = nn.Linear(control_hidden_size // 2, self.num_discrete)

        # 控制标签的归一化参数 (推理时使用)
        self.steering_range = (-1.0, 1.0)
        self.throttle_range = (0.0, 1.0)
        self.brake_range = (0.0, 1.0)

        self._init_parameters()

    def _init_parameters(self):
        """初始化参数"""
        for module in self.shared_mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.continuous_proj.weight)
        nn.init.xavier_uniform_(self.gear_proj.weight)
        if self.num_discrete > 0:
            nn.init.xavier_uniform_(self.discrete_proj.weight)

    def forward(
        self,
        hidden_state: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            hidden_state: [B, hidden_size] (最后一个 token 的 hidden state)

        Returns:
            {
                "continuous": torch.Tensor,       # [B, 4]
                "discrete_logits": torch.Tensor,  # [B, N] (可选)
                "discrete_probs": torch.Tensor,   # [B, N] (可选)
                "discrete_action": torch.Tensor,  # [B] (可选)
            }
        """
        # 共享 MLP
        shared = self.shared_mlp(hidden_state)  # [B, control_hidden_size//2]

        # 连续控制输出
        raw_continuous = self.continuous_proj(shared)
        steering = torch.tanh(raw_continuous[:, :1])
        pedals = torch.sigmoid(raw_continuous[:, 1:3])
        gear_logits = self.gear_proj(shared)
        gear = gear_logits.argmax(dim=-1, keepdim=True).to(steering.dtype)
        continuous = torch.cat([steering, pedals, gear], dim=-1)

        result = {
            "continuous": continuous,
            "continuous_regression": torch.cat([steering, pedals], dim=-1),
            "gear_logits": gear_logits,
        }

        # 离散控制输出
        if self.num_discrete > 0:
            discrete_logits = self.discrete_proj(shared)  # [B, N]
            discrete_probs = F.softmax(discrete_logits, dim=-1)  # [B, N]
            discrete_action = torch.argmax(discrete_probs, dim=-1)  # [B]

            result["discrete_logits"] = discrete_logits
            result["discrete_probs"] = discrete_probs
            result["discrete_action"] = discrete_action

        return result

    def decode_continuous(
        self,
        continuous_normalized: torch.Tensor,
        steering_range: Optional[Tuple[float, float]] = None,
        throttle_range: Optional[Tuple[float, float]] = None,
        brake_range: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, float]:
        """
        将归一化的连续控制信号解码为实际值

        Args:
            continuous_normalized: [4] 或 [B, 4]，[steering, throttle, brake, gear]
            steering_range: 转向角范围
            throttle_range: 油门范围
            brake_range: 刹车范围

        Returns:
            {"steering": float, "throttle": float, "brake": float, "gear": int}
        """
        if steering_range is None:
            steering_range = self.steering_range
        if throttle_range is None:
            throttle_range = self.throttle_range
        if brake_range is None:
            brake_range = self.brake_range

        # The head now emits physical normalized controls directly.
        steering = continuous_normalized[0].clamp(*steering_range)
        throttle = continuous_normalized[1].clamp(*throttle_range)
        brake = continuous_normalized[2].clamp(*brake_range)
        gear = int(round(float(continuous_normalized[3])))
        gear = max(0, min(4, gear))

        return {
            "steering": steering.item() if hasattr(steering, 'item') else steering,
            "throttle": throttle.item() if hasattr(throttle, 'item') else throttle,
            "brake": brake.item() if hasattr(brake, 'item') else brake,
            "gear": gear,
        }

    def decode_discrete(
        self,
        action_index: int,
    ) -> str:
        """
        将离散动作索引解码为动作名称

        Args:
            action_index: 动作索引

        Returns:
            动作名称字符串
        """
        if 0 <= action_index < self.num_discrete:
            return self.discrete_actions[action_index]
        return "unknown"

    def get_action_to_id(self) -> Dict[str, int]:
        """获取动作名称到 ID 的映射"""
        return {name: i for i, name in enumerate(self.discrete_actions)}

    def get_id_to_action(self) -> Dict[int, str]:
        """获取 ID 到动作名称的映射"""
        return {i: name for i, name in enumerate(self.discrete_actions)}
