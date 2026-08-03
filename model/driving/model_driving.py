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

from model.model_minimind import MiniMindForCausalLM
from config.driving_config import DrivingConfig

from .camera_encoder import CameraEncoder
from .temporal_encoder import TemporalEncoder
from .sensor_fusion_module import SensorFusionModule
from .control_head import ControlHead


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
        self.fallback_patch_projection = nn.Linear(3, 768)

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
            lidar_point_dims=config.lidar_point_dims,
            radar_point_dims=config.radar_point_dims,
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
        if self.vision_encoder is not None:
            for param in self.vision_encoder.parameters():
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

        # CLIP 编码；无本地权重时使用可训练的轻量 patch fallback，
        # 仅供 smoke test，生产训练必须加载真实 CLIP checkpoint。
        if self.vision_encoder is not None:
            grad_enabled = not self.config.freeze_vision_encoder
            with torch.set_grad_enabled(grad_enabled):
                clip_outputs = self.vision_encoder.vision_model(
                    pixel_values=flat_pixel
                )
            visual_features = clip_outputs.last_hidden_state[:, 1:, :]
        else:
            grid_size = int(self.config.image_tokens_per_camera ** 0.5)
            pooled = F.adaptive_avg_pool2d(flat_pixel, (grid_size, grid_size))
            visual_features = pooled.flatten(2).transpose(1, 2)
            visual_features = self.fallback_patch_projection(visual_features)

        NP = visual_features.shape[1]
        ve_hs = visual_features.shape[2]

        # 重组为相机结构: [B, NC, NF, NP, ve_hs]
        visual_features = visual_features.view(B, NC, NF, NP, ve_hs)

        # 先在每个相机/patch 内聚合时间，再做多相机投影。
        temporal_features = self.temporal_encoder(visual_features)
        camera_features = self.camera_encoder(temporal_features)
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
        lidar_mask: Optional[torch.BoolTensor] = None,
        radar_mask: Optional[torch.BoolTensor] = None,
        gps_imu_mask: Optional[torch.BoolTensor] = None,
        control_labels: Optional[torch.FloatTensor] = None,
        action_labels: Optional[torch.LongTensor] = None,
        control_label_mask: Optional[torch.BoolTensor] = None,
        action_label_mask: Optional[torch.BoolTensor] = None,
        labels: Optional[torch.LongTensor] = None,
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
            vision_sequence = self.sensor_fusion(
                vision_sequence,
                lidar_feature=lidar_pointcloud,
                radar_feature=radar_data,
                gps_imu_feature=gps_imu,
                lidar_mask=lidar_mask,
                radar_mask=radar_mask,
                gps_imu_mask=gps_imu_mask,
            )

        # ========== 3. 特征注入 ==========
        if vision_sequence is not None:
            hidden_states = self._inject_vision_features(hidden_states, vision_sequence, input_ids)
            visual_mask = torch.ones(
                batch_size, vision_sequence.shape[1],
                dtype=attention_mask.dtype if attention_mask is not None else torch.long,
                device=hidden_states.device,
            )
            text_mask = attention_mask if attention_mask is not None else torch.ones(
                batch_size, text_seq_len, dtype=visual_mask.dtype,
                device=hidden_states.device,
            )
            attention_mask = torch.cat([visual_mask, text_mask], dim=1)

        total_seq_len = hidden_states.shape[1]
        if total_seq_len > self.model.freqs_cos.shape[0]:
            raise ValueError(
                f"multimodal sequence length {total_seq_len} exceeds "
                f"max_position_embeddings={self.model.freqs_cos.shape[0]}"
            )

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
        if attention_mask is not None:
            text_attention = attention_mask[:, -text_seq_len:]
            last_index = text_attention.long().sum(dim=1).clamp_min(1) - 1
            control_hidden = text_hidden[
                torch.arange(batch_size, device=text_hidden.device), last_index
            ]
        else:
            control_hidden = text_hidden[:, -1, :]
        control_outputs = self.control_head(control_hidden)

        text_loss = None
        if labels is not None and text_logits.shape[1] > 1:
            text_loss = F.cross_entropy(
                text_logits[:, :-1].reshape(-1, text_logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        control_loss = None
        if control_labels is not None:
            target = control_labels.to(
                control_outputs["continuous_regression"].dtype
            )
            regression_loss = F.smooth_l1_loss(
                control_outputs["continuous_regression"], target[:, :3],
                reduction="none",
            ).mean(dim=-1)
            gear_loss = F.cross_entropy(
                control_outputs["gear_logits"],
                target[:, 3].long().clamp(0, 4),
                reduction="none",
            )
            per_sample = regression_loss + gear_loss
            mask = control_label_mask.bool() if control_label_mask is not None else (
                torch.ones_like(per_sample, dtype=torch.bool)
            )
            if mask.any():
                control_loss = per_sample[mask].mean()
        action_loss = None
        if action_labels is not None and "discrete_logits" in control_outputs:
            valid_actions = action_label_mask.bool() if action_label_mask is not None else (
                action_labels.ne(-100)
            )
            if valid_actions.any():
                action_loss = F.cross_entropy(
                    control_outputs["discrete_logits"][valid_actions],
                    action_labels[valid_actions],
                )
        losses = [value for value in (text_loss,) if value is not None]
        total_loss = sum(losses) if losses else None
        if control_loss is not None:
            weighted = self.config.loss_control_weight * control_loss
            total_loss = weighted if total_loss is None else total_loss + weighted
        if action_loss is not None:
            weighted = self.config.loss_action_weight * action_loss
            total_loss = weighted if total_loss is None else total_loss + weighted

        # ========== 7. 组装输出 ==========
        output = CausalLMOutputWithPast(
            loss=total_loss,
            logits=text_logits,
            past_key_values=presents,
            hidden_states=hidden_states,
        )
        output.control_outputs = control_outputs
        output.losses = {
            "text": text_loss,
            "control": control_loss,
            "action": action_loss,
        }
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

        decision_output = self(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            pixel_values=pixel_values,
        ).control_outputs
        continuous = decision_output["continuous"][0]
        control_output = self.control_head.decode_continuous(continuous)
        action_index = int(decision_output["discrete_action"][0])
        action = self.control_head.decode_discrete(action_index)
        action_probs = decision_output["discrete_probs"][0].detach().cpu().tolist()

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
