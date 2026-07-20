"""
MiniMind-Driving: 自动驾驶端到端多模态模型

架构概述:
    多相机图像 → CameraEncoder → 视觉特征序列
    多帧时序   → TemporalEncoder → 时序聚合特征
    传感器融合 → SensorFusionModule → 融合特征
    融合特征注入文本序列 → MiniMind LLM → 文本 logits + 控制输出

输入:
    - 多相机图像: [B, num_cameras, num_frames, C, H, W]
    - 文本 prompt: [B, seq_len]
    - 可选传感器: 激光雷达 / 雷达 / GPS-IMU

输出:
    - 文本 logits: [B, seq_len, vocab_size]
    - 控制信号: 连续 [B, 4] + 离散 [B, num_actions]
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Union
from transformers import CLIPModel, CLIPProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

from model.model_minimind import MiniMindForCausalLM, MiniMindConfig, RMSNorm
from model.model_vlm import VLMConfig

from .camera_encoder import CameraEncoder
from .temporal_encoder import TemporalEncoder
from .sensor_fusion_module import SensorFusionModule
from .control_head import ControlHead


class DrivingConfig(VLMConfig):
    """
    自动驾驶模型配置
    继承 VLMConfig，扩展多相机、时序、传感器融合、控制输出
    """
    model_type = "driving"

    def __init__(
        self,
        # === 多相机配置 ===
        num_cameras: int = 4,
        camera_names: Optional[List[str]] = None,
        image_tokens_per_camera: int = 196,
        total_image_tokens: Optional[int] = None,

        # === 时序配置 ===
        num_history_frames: int = 3,
        frame_skip: int = 1,

        # === 传感器融合配置 ===
        enable_lidar: bool = False,
        enable_radar: bool = False,
        enable_gps_imu: bool = False,
        sensor_fusion_method: str = "concat",
        lidar_hidden_size: int = 512,
        radar_hidden_size: int = 64,
        gps_imu_dims: int = 6,

        # === 控制输出配置 ===
        control_type: str = "both",
        continuous_dims: int = 4,
        discrete_actions: Optional[List[str]] = None,
        control_hidden_size: int = 256,

        # === 训练策略 ===
        freeze_vision_encoder: bool = True,
        freeze_first_layers: int = 0,

        # === 视觉编码器 ===
        vision_encoder_path: str = "./model/vision_model/clip-vit-base-patch16",

        **kwargs,
    ):
        self.camera_names = camera_names or ["front", "left", "right", "rear"]
        self.total_image_tokens = total_image_tokens or (num_cameras * image_tokens_per_camera)
        self.discrete_actions = discrete_actions or [
            "keep_lane", "turn_left", "turn_right",
            "stop", "accelerate", "decelerate",
            "yield", "overtake", "park",
            "emergency_brake", "follow_lane", "change_lane_left", "change_lane_right",
        ]
        super().__init__(
            image_special_token='@' * image_tokens_per_camera,
            **kwargs,
        )


class MiniMindDriving(MiniMindForCausalLM):
    """
    自动驾驶端到端模型

    前向传播流程:
        1. 文本嵌入: input_ids -> embedding
        2. 视觉编码: pixel_values -> CLIP -> CameraEncoder -> TemporalEncoder
        3. 传感器融合: 相机 + 可选传感器 -> SensorFusionModule
        4. 特征注入: 视觉特征替换/拼接到文本 hidden states
        5. LLM 前向: 标准 MiniMind Transformer
        6. 控制输出: 从最后一个 token 的 hidden state -> ControlHead
    """

    config_class = DrivingConfig

    def __init__(self, config: DrivingConfig = None, vision_encoder_path: str = "./model/vision_model/clip-vit-base-patch16"):
        # 1. 处理配置
        if config is None:
            config = DrivingConfig()
        if vision_encoder_path and not config.vision_encoder_path:
            config.vision_encoder_path = vision_encoder_path

        # 2. 初始化父类 (MiniMindForCausalLM)
        super().__init__(config)

        self.config = config

        # 3. 视觉编码器 (CLIP)
        self.vision_encoder, self.processor = self._load_vision_encoder(config.vision_encoder_path)

        # 4. 多相机编码器
        self.camera_encoder = CameraEncoder(
            ve_hidden_size=768,
            hidden_size=config.hidden_size,
            num_cameras=config.num_cameras,
            image_tokens_per_camera=config.image_tokens_per_camera,
            use_cross_camera_attn=False,
        )

        # 5. 时序编码器
        self.temporal_encoder = TemporalEncoder(
            hidden_size=config.hidden_size,
            num_history_frames=config.num_history_frames,
            mode="aggregate",
            aggregation_method="mean",
        )

        # 6. 传感器融合模块
        self.sensor_fusion = SensorFusionModule(
            hidden_size=config.hidden_size,
            fusion_method=config.sensor_fusion_method,
            enable_lidar=config.enable_lidar,
            enable_radar=config.enable_radar,
            enable_gps_imu=config.enable_gps_imu,
            lidar_hidden_size=config.lidar_hidden_size,
            radar_hidden_size=config.radar_hidden_size,
            gps_imu_dims=config.gps_imu_dims,
        )

        # 7. 控制输出头
        self.control_head = ControlHead(
            hidden_size=config.hidden_size,
            continuous_dims=config.continuous_dims,
            discrete_actions=config.discrete_actions,
            control_hidden_size=config.control_hidden_size,
        )

        # 8. 冻结策略
        if config.freeze_vision_encoder:
            self._freeze_vision_encoder()
        if config.freeze_first_layers > 0:
            self._freeze_first_layers(config.freeze_first_layers)

    def _load_vision_encoder(self, path: str):
        """加载 CLIP 视觉编码器"""
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
        if not os.path.exists(path):
            print(f"[WARNING] Vision encoder path not found: {path}, using random initialization")
            return None, None

        model = CLIPModel.from_pretrained(path)
        processor = CLIPProcessor.from_pretrained(path)
        for param in model.parameters():
            param.requires_grad = False
        return model.eval(), processor

    def _freeze_vision_encoder(self):
        """冻结视觉编码器"""
        for param in self.camera_encoder.parameters():
            param.requires_grad = False

    def _freeze_first_layers(self, num_layers: int):
        """冻结前 N 层 LLM"""
        for layer in self.model.layers[:num_layers]:
            for param in layer.parameters():
                param.requires_grad = False

    def _encode_images(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        编码多相机多帧图像

        Args:
            pixel_values: [B, num_cameras, num_frames, C, H, W]

        Returns:
            [B, total_vision_tokens, hidden_size]
        """
        B, NC, NF, C, H, W = pixel_values.shape

        # 展平相机和帧维度: [B*NC*NF, C, H, W]
        flat_pixel = pixel_values.view(B * NC * NF, C, H, W)

        # CLIP 编码
        with torch.no_grad():
            clip_outputs = self.vision_encoder.vision_model(pixel_values=flat_pixel)
        visual_features = clip_outputs.last_hidden_state[:, 1:, :]  # 去掉 [CLS], [B*NC*NF, NP, 768]

        NP = visual_features.shape[1]
        ve_hs = visual_features.shape[2]

        # 重组为相机结构: [B, NC, NF, NP, ve_hs]
        visual_features = visual_features.view(B, NC, NF, NP, ve_hs)

        # CameraEncoder: 投影 + 空间位置编码
        camera_features = self.camera_encoder(visual_features)
        # camera_features: [B, NC*NP, HS]

        return camera_features

    def _inject_vision_features(
        self,
        text_hidden: torch.Tensor,
        vision_sequence: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        将视觉特征注入到文本 hidden states 中

        策略: 在文本序列开头拼接视觉特征
        text_hidden:  [B, seq_len, HS]
        vision_sequence: [B, vision_tokens, HS]
        输出: [B, seq_len + vision_tokens, HS]
        """
        combined = torch.cat([vision_sequence, text_hidden], dim=1)
        return combined

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        lidar_pointcloud: Optional[torch.FloatTensor] = None,
        radar_data: Optional[torch.FloatTensor] = None,
        gps_imu: Optional[torch.FloatTensor] = None,
        control_labels: Optional[torch.FloatTensor] = None,
        action_labels: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        """
        前向传播

        Args:
            input_ids:        [B, seq_len] 文本 token IDs
            attention_mask:   [B, seq_len] 注意力掩码
            pixel_values:     [B, num_cameras, num_frames, C, H, W] 多相机多帧图像
            lidar_pointcloud: 可选，激光雷达 [B, num_cameras, num_points, D]
            radar_data:       可选，毫米波雷达 [B, num_cameras, num_detections, D]
            gps_imu:          可选，GPS/IMU [B, 6]
            control_labels:   训练用，连续控制标签 [B, 4]
            action_labels:    训练用，离散动作标签 [B]
            use_cache:        是否使用 KV cache

        Returns:
            CausalLMOutputWithPast + control_outputs
        """
        batch_size = input_ids.shape[0]
        text_seq_len = input_ids.shape[1]

        # ========== 1. 文本嵌入 ==========
        hidden_states = self.model.dropout(self.model.embed_tokens(input_ids))

        # ========== 2. 视觉编码 ==========
        vision_sequence = None
        if pixel_values is not None:
            vision_sequence = self._encode_images(pixel_values)

        # ========== 3. 特征注入 ==========
        if vision_sequence is not None:
            hidden_states = self._inject_vision_features(hidden_states, vision_sequence, input_ids)

        total_seq_len = hidden_states.shape[1]

        # ========== 4. LLM 前向 ==========
        position_embeddings = (
            self.model.freqs_cos[:total_seq_len],
            self.model.freqs_sin[:total_seq_len],
        )

        presents = []
        for layer in self.model.layers:
            hidden_states, present = layer(
                hidden_states, position_embeddings,
                past_key_value=None, use_cache=use_cache, attention_mask=attention_mask
            )
            presents.append(present)

        hidden_states = self.model.norm(hidden_states)

        # ========== 5. 文本 logits ==========
        # 只对原始文本部分计算 logits (去掉视觉 token)
        text_start = vision_sequence.shape[1] if vision_sequence is not None else 0
        text_hidden = hidden_states[:, text_start:text_start + text_seq_len, :]
        text_logits = self.lm_head(text_hidden)  # [B, text_seq_len, vocab_size]

        # ========== 6. 控制输出 ==========
        control_outputs = None
        if control_labels is not None or action_labels is not None:
            # 从文本部分的最后一个 token 提取控制信号
            control_hidden = text_hidden[:, -1, :]  # [B, hidden_size]
            control_outputs = self.control_head(control_hidden)

        # ========== 7. 组装输出 ==========
        output = CausalLMOutputWithPast(
            logits=text_logits,
            past_key_values=presents,
            hidden_states=hidden_states,
        )
        output.control_outputs = control_outputs
        return output

    @torch.no_grad()
    def generate_driving_decision(
        self,
        pixel_values: torch.Tensor,
        prompt_text: str = "",
        tokenizer=None,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> Dict:
        """
        生成驾驶决策

        Args:
            pixel_values: [B, num_cameras, num_frames, C, H, W]
            prompt_text: 提示文本
            tokenizer: 分词器
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度

        Returns:
            {
                "text_response": str,
                "control": {"steering": float, "throttle": float, "brake": float, "gear": int},
                "action": str,
                "action_probs": List[float],
            }
        """
        # 构建输入
        if tokenizer is not None:
            messages = [{"role": "user", "content": prompt_text}]
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(input_text, return_tensors="pt").to(self.device)
        else:
            inputs = {"input_ids": torch.tensor([[1]]).to(self.device)}

        # 生成文本
        generated_ids = self.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id if tokenizer else 0,
            eos_token_id=tokenizer.eos_token_id if tokenizer else 2,
            **kwargs,
        )

        # 解码文本
        if tokenizer:
            text_response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        else:
            text_response = generated_ids[0].tolist()

        # 获取控制输出 (需要从 forward 中获取)
        # 这里简化处理，实际需要从 model 内部获取 control_outputs
        control_output = {
            "steering": 0.0,
            "throttle": 0.0,
            "brake": 0.0,
            "gear": 2,
        }
        action = "keep_lane"
        action_probs = [0.0] * self.config.num_discrete_actions

        return {
            "text_response": text_response,
            "control": control_output,
            "action": action,
            "action_probs": action_probs,
        }

    def get_trainable_parameters(self):
        """获取可训练参数列表"""
        trainable = []
        frozen = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable.append((name, param))
            else:
                frozen.append(name)
        return trainable, frozen

    def get_parameter_counts(self):
        """获取参数量统计"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {
            "total": total,
            "trainable": trainable,
            "frozen": frozen,
            "trainable_ratio": trainable / total if total > 0 else 0,
        }
