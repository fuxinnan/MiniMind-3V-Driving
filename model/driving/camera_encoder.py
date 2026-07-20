"""
多相机编码器

将多个相机的 CLIP 视觉特征进行结构化解码和融合
输入: [B, num_cameras, num_patches, hidden_size]
输出: [B, total_vision_tokens, hidden_size]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CameraEncoder(nn.Module):
    """
    多相机编码器

    架构:
        1. 对每个相机的视觉特征进行投影到 LLM 维度
        2. 添加空间位置编码
        3. (可选) 跨相机交叉注意力融合

    设计考虑:
        - CLIP ViT-B/16 输出 14x14=196 个 patch + 1 个 [CLS]
        - 4 个相机 = 4 x 196 = 784 个视觉 token
        - 空间位置编码保持 patch 的 2D 结构信息
    """

    def __init__(
        self,
        ve_hidden_size: int = 768,         # CLIP ViT 输出维度
        hidden_size: int = 512,            # LLM 隐藏层维度
        num_cameras: int = 4,              # 相机数量
        image_tokens_per_camera: int = 196,  # 每相机 token 数
        use_cross_camera_attn: bool = False,  # 是否使用跨相机注意力
    ):
        super().__init__()
        self.num_cameras = num_cameras
        self.image_tokens_per_camera = image_tokens_per_camera

        # 投影层: CLIP 维度 -> LLM 维度
        self.projection = nn.Sequential(
            nn.Linear(ve_hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )

        # 空间位置编码 (保持 14x14 的 2D 结构)
        # 196 patches = 14x14 grid
        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, num_cameras, image_tokens_per_camera, hidden_size)
        )

        # 跨相机注意力 (可选，计算量大)
        self.use_cross_camera_attn = use_cross_camera_attn
        if use_cross_camera_attn:
            self.cross_camera_attn = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=8,
                dropout=0.1,
                batch_first=True,
            )
            self.cross_camera_norm = nn.LayerNorm(hidden_size)

        self._init_parameters()

    def _init_parameters(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.spatial_pos_embed)
        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        visual_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            visual_features: [B, num_cameras, num_patches, ve_hidden_size]
                             (已去掉 CLS token)
            attention_mask:  [B, num_cameras, num_patches] (可选)

        Returns:
            [B, num_cameras * num_patches, hidden_size]
        """
        B, NC, NP, VE_HS = visual_features.shape
        assert NC == self.num_cameras, \
            f"Expected {self.num_cameras} cameras, got {NC}"
        assert NP == self.image_tokens_per_camera, \
            f"Expected {self.image_tokens_per_camera} patches, got {NP}"

        # 1. 投影到 LLM 维度
        # [B, NC, NP, VE_HS] -> [B, NC, NP, HS]
        visual_features = self.projection(visual_features)

        # 2. 添加空间位置编码
        visual_features = visual_features + self.spatial_pos_embed

        # 3. (可选) 跨相机注意力
        if self.use_cross_camera_attn:
            # 重排: [B, NC, NP, HS] -> [B*NP, NC, HS]
            BNP = B * NP
            flat = visual_features.view(BNP, NC, -1)
            attn_output, _ = self.cross_camera_attn(flat, flat, flat)
            attn_output = self.cross_camera_norm(attn_output)
            visual_features = attn_output.view(B, NC, NP, -1)

        # 4. 展平为序列格式
        # [B, NC, NP, HS] -> [B, NC*NP, HS]
        output = visual_features.reshape(B, NC * NP, -1)

        return output

    def get_num_output_tokens(self) -> int:
        """获取输出 token 数"""
        return self.num_cameras * self.image_tokens_per_camera
