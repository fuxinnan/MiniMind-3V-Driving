"""
传感器融合模块

支持多种融合策略:
    - early_fusion: 在特征层面早期融合 (拼接后投影)
    - late_fusion: 在输出层面晚期融合 (分别处理再合并)
    - cross_attention: 交叉注意力融合
    - concat: 简单拼接
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SensorFusionModule(nn.Module):
    """
    传感器融合模块

    支持激光雷达、毫米波雷达、GPS/IMU 等可选传感器的融合
    默认只使用相机，其他传感器通过 enable 开关控制
    """

    def __init__(
        self,
        hidden_size: int = 512,
        fusion_method: str = "concat",  # "concat" / "cross_attention" / "early"
        enable_lidar: bool = False,
        enable_radar: bool = False,
        enable_gps_imu: bool = False,
        lidar_hidden_size: int = 512,
        radar_hidden_size: int = 64,
        gps_imu_dims: int = 6,
        lidar_point_dims: int = 5,
        radar_point_dims: int = 18,
    ):
        super().__init__()
        self.fusion_method = fusion_method
        self.enable_lidar = enable_lidar
        self.enable_radar = enable_radar
        self.enable_gps_imu = enable_gps_imu

        # 激光雷达编码器
        if enable_lidar:
            self.lidar_encoder = nn.Sequential(
                nn.Linear(lidar_point_dims, hidden_size),
                nn.GELU(),
                nn.LayerNorm(hidden_size),
            )

        # 雷达编码器
        if enable_radar:
            self.radar_encoder = nn.Sequential(
                nn.Linear(radar_point_dims, hidden_size),
                nn.GELU(),
                nn.LayerNorm(hidden_size),
            )

        # GPS/IMU 编码器
        if enable_gps_imu:
            self.gps_imu_encoder = nn.Sequential(
                nn.Linear(gps_imu_dims, hidden_size),
                nn.GELU(),
                nn.LayerNorm(hidden_size),
            )

        # 交叉注意力 (可选)
        if fusion_method == "cross_attention":
            self.fusion_attn = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=8,
                dropout=0.1,
                batch_first=True,
            )
            self.fusion_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        vision_sequence: torch.Tensor,       # [B, vision_tokens, hidden_size]
        lidar_feature: Optional[torch.Tensor] = None,  # [B, lidar_tokens, hidden_size]
        radar_feature: Optional[torch.Tensor] = None,  # [B, radar_tokens, hidden_size]
        gps_imu_feature: Optional[torch.Tensor] = None,  # [B, hidden_size] (标量特征)
        lidar_mask: Optional[torch.Tensor] = None,
        radar_mask: Optional[torch.Tensor] = None,
        gps_imu_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            vision_sequence:   [B, vision_tokens, hidden_size] (相机视觉特征)
            lidar_feature:     可选，激光雷达特征
            radar_feature:     可选，雷达特征
            gps_imu_feature:   可选，GPS/IMU 标量特征 [B, hidden_size]

        Returns:
            [B, total_tokens, hidden_size]
        """
        features = []

        def masked_pool(values, mask):
            encoded_mask = mask
            if encoded_mask is None:
                return values.mean(dim=1)
            encoded_mask = encoded_mask.to(values.dtype).unsqueeze(-1)
            return (values * encoded_mask).sum(dim=1) / encoded_mask.sum(
                dim=1
            ).clamp_min(1.0)

        # 融合激光雷达
        if self.enable_lidar and lidar_feature is not None:
            lidar_feat = self.lidar_encoder(lidar_feature)
            features.append(masked_pool(lidar_feat, lidar_mask).unsqueeze(1))

        # 融合雷达
        if self.enable_radar and radar_feature is not None:
            radar_feat = self.radar_encoder(radar_feature)
            features.append(masked_pool(radar_feat, radar_mask).unsqueeze(1))

        # 融合 GPS/IMU (广播到 vision_tokens 长度)
        if self.enable_gps_imu and gps_imu_feature is not None:
            gps_feat = self.gps_imu_encoder(gps_imu_feature)  # [B, hidden_size]
            if gps_imu_mask is not None:
                gps_feat = gps_feat * gps_imu_mask.to(gps_feat.dtype).unsqueeze(-1)
            features.append(gps_feat.unsqueeze(1))

        if not features:
            return vision_sequence

        if self.fusion_method == "concat":
            # Token concatenation preserves the camera sequence and adds one
            # masked summary token per enabled modality.
            return torch.cat([vision_sequence, *features], dim=1)

        elif self.fusion_method == "cross_attention":
            # 视觉作为 query，其他传感器作为 key/value
            query = vision_sequence
            key_value = torch.cat(features, dim=1)
            attn_output, _ = self.fusion_attn(query, key_value, key_value)
            output = self.fusion_norm(attn_output + query)
            return output

        elif self.fusion_method == "early":
            # 逐元素相加 (需要维度对齐)
            # 简化: 取平均
            summary = torch.cat(features, dim=1).mean(dim=1, keepdim=True)
            return vision_sequence + summary

        else:
            raise ValueError(f"Unknown fusion_method: {self.fusion_method}")
