"""
时序编码器

对多帧视觉特征进行时序建模，捕捉动态信息
支持两种模式:
    1. temporal_aggregate: 时序聚合 (在 CameraEncoder 中完成)
    2. temporal_enhance: 时序增强 (保留时序信息)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TemporalEncoder(nn.Module):
    """
    时序编码器

    架构:
        模式1 (聚合): 对多帧特征进行时序平均/注意力聚合
        模式2 (增强): 添加帧位置编码，保留时序信息

    设计考虑:
        - 自动驾驶是时序敏感任务，需要捕捉运动信息
        - 但视觉 token 数已经很大 (784)，不宜再增加
        - 因此默认采用聚合策略，将多帧压缩为单帧特征
    """

    def __init__(
        self,
        hidden_size: int = 512,
        num_history_frames: int = 3,
        mode: str = "aggregate",  # "aggregate" / "enhance"
        aggregation_method: str = "attention",  # "mean" / "attention" / "lstm"
    ):
        super().__init__()
        self.num_history_frames = num_history_frames
        self.mode = mode

        if mode == "aggregate":
            if aggregation_method == "attention":
                # 时序注意力聚合
                self.temporal_attn = nn.MultiheadAttention(
                    embed_dim=hidden_size,
                    num_heads=8,
                    dropout=0.1,
                    batch_first=True,
                )
                self.temporal_norm = nn.LayerNorm(hidden_size)
            elif aggregation_method == "mean":
                # 简单平均 (默认)
                pass
            elif aggregation_method == "lstm":
                self.temporal_lstm = nn.LSTM(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=False,
                )
            else:
                raise ValueError(f"Unknown aggregation_method: {aggregation_method}")

        elif mode == "enhance":
            # 帧位置编码
            self.frame_position_embedding = nn.Embedding(
                num_history_frames, hidden_size
            )
            self.frame_position_norm = nn.LayerNorm(hidden_size)

        else:
            raise ValueError(f"Unknown mode: {mode}")

    def forward(
        self,
        camera_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            camera_features: [B, num_cameras, num_frames, num_patches, hidden_size]

        Returns:
            mode="aggregate": [B, num_cameras, num_patches, hidden_size]
            mode="enhance":   [B, num_cameras, num_frames, num_patches, hidden_size]
        """
        B, NC, NF, NP, HS = camera_features.shape
        if not torch.jit.is_tracing() and NF != self.num_history_frames:
            raise ValueError(
                f"Expected {self.num_history_frames} frames, got {NF}"
            )

        if self.mode == "aggregate":
            return self._aggregate(camera_features)
        elif self.mode == "enhance":
            return self._enhance(camera_features)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _aggregate(self, camera_features: torch.Tensor) -> torch.Tensor:
        """时序聚合: 将多帧特征压缩为单帧"""
        B, NC, NF, NP, HS = camera_features.shape

        if hasattr(self, 'temporal_attn'):
            # 注意力聚合
            # 重排: [B, NC, NF, NP, HS] -> [B*NC*NP, NF, HS]
            flat = camera_features.permute(0, 1, 3, 2, 4).reshape(B * NC * NP, NF, HS)
            attn_output, _ = self.temporal_attn(flat, flat, flat)
            attn_output = self.temporal_norm(attn_output)
            # 取最后一个时间步
            aggregated = attn_output[:, -1, :]  # [B*NC*NP, HS]
            result = aggregated.view(B, NC, NP, HS)

        elif hasattr(self, 'temporal_lstm'):
            # LSTM 聚合
            flat = camera_features.permute(0, 1, 3, 2, 4).reshape(B * NC * NP, NF, HS)
            _, (hidden, _) = self.temporal_lstm(flat)
            # hidden: [num_layers, B*NC*NP, HS]
            aggregated = hidden[-1]  # [B*NC*NP, HS]
            result = aggregated.view(B, NC, NP, HS)

        else:
            # 简单平均
            result = camera_features.mean(dim=2)  # [B, NC, NP, HS]

        return result

    def _enhance(self, camera_features: torch.Tensor) -> torch.Tensor:
        """时序增强: 添加帧位置编码"""
        B, NC, NF, NP, HS = camera_features.shape

        # 帧位置编码
        frame_pos = torch.arange(NF, device=camera_features.device)
        pos_embed = self.frame_position_embedding(frame_pos)  # [NF, HS]
        pos_embed = pos_embed.view(1, 1, NF, 1, HS).expand(B, NC, NF, NP, HS)

        result = camera_features + pos_embed
        result = self.frame_position_norm(result)

        return result
