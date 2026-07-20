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
    ):
        super().__init__()
        self.fusion_method = fusion_method
        self.enable_lidar = enable_lidar
        self.enable_radar = enable_radar
        self.enable_gps_imu = enable_gps_imu

        # 激光雷达编码器
        if enable_lidar:
            self.lidar_encoder = nn.Sequential(
                nn.Linear(lidar_hidden_size, hidden_size),
                nn.GELU(),
                nn.LayerNorm(hidden_size),
            )

        # 雷达编码器
        if enable_radar:
            self.radar_encoder = nn.Sequential(
                nn.Linear(radar_hidden_size, hidden_size),
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

        # 融合后投影
        if fusion_method == "concat":
            # 计算融合后的维度
            num_sensors = 1  # 相机 (1)
            if enable_lidar:
                num_sensors += 1
            if enable_radar:
                num_sensors += 1
            if enable_gps_imu:
                num_sensors += 1
            self.fusion_proj = nn.Sequential(
                nn.Linear(hidden_size * num_sensors, hidden_size),
                nn.GELU(),
                nn.LayerNorm(hidden_size),
            )

    def forward(
        self,
        vision_sequence: torch.Tensor,       # [B, vision_tokens, hidden_size]
        lidar_feature: Optional[torch.Tensor] = None,  # [B, lidar_tokens, hidden_size]
        radar_feature: Optional[torch.Tensor] = None,  # [B, radar_tokens, hidden_size]
        gps_imu_feature: Optional[torch.Tensor] = None,  # [B, hidden_size] (标量特征)
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
        features = [vision_sequence]
        B = vision_sequence.shape[0]

        # 融合激光雷达
        if self.enable_lidar and lidar_feature is not None:
            lidar_feat = self.lidar_encoder(lidar_feature)
            features.append(lidar_feat)

        # 融合雷达
        if self.enable_radar and radar_feature is not None:
            radar_feat = self.radar_encoder(radar_feature)
            features.append(radar_feat)

        # 融合 GPS/IMU (广播到 vision_tokens 长度)
        if self.enable_gps_imu and gps_imu_feature is not None:
            gps_feat = self.gps_imu_encoder(gps_imu_feature)  # [B, hidden_size]
            gps_feat = gps_feat.unsqueeze(1).expand(B, vision_sequence.shape[1], -1)
            features.append(gps_feat)

        if len(features) == 1:
            # 只有相机，直接返回
            return vision_sequence

        if self.fusion_method == "concat":
            # 拼接后投影
            combined = torch.cat(features, dim=-1)
            output = self.fusion_proj(combined)
            return output

        elif self.fusion_method == "cross_attention":
            # 视觉作为 query，其他传感器作为 key/value
            query = features[0]
            key_value = torch.cat(features[1:], dim=1)
            attn_output, _ = self.fusion_attn(query, key_value, key_value)
            output = self.fusion_norm(attn_output + query)
            return output

        elif self.fusion_method == "early":
            # 逐元素相加 (需要维度对齐)
            # 简化: 取平均
            output = torch.stack(features, dim=2).mean(dim=2)
            return output

        else:
            raise ValueError(f"Unknown fusion_method: {self.fusion_method}")
