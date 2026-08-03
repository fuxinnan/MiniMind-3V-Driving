import pytest
import torch

pytest.importorskip("transformers")

from config.driving_config import DrivingConfig
from model.driving.model_driving import MiniMindDriving


def tiny_config(**kwargs):
    values = dict(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        max_seq_len=32,
        image_tokens_per_camera=4,
        camera_input_size=(16, 16),
        vision_encoder_path="",
        flash_attn=False,
    )
    values.update(kwargs)
    return DrivingConfig(**values)


def test_multitask_forward_and_backward():
    model = MiniMindDriving(tiny_config(), vision_encoder_path="")
    images = torch.randn(2, 4, 3, 3, 16, 16)
    tokens = torch.ones(2, 5, dtype=torch.long)
    outputs = model(
        input_ids=tokens,
        attention_mask=torch.ones_like(tokens),
        pixel_values=images,
        labels=tokens,
        control_labels=torch.tensor([[0, .2, 0, 2], [.1, .3, 0, 2.]]),
        action_labels=torch.tensor([0, 1]),
    )
    assert outputs.control_outputs["continuous"].shape == (2, 4)
    assert outputs.control_outputs["discrete_logits"].shape == (2, 13)
    assert outputs.loss is not None
    outputs.loss.backward()


def test_sensor_masks_add_optional_tokens():
    config = tiny_config(enable_gps_imu=True)
    model = MiniMindDriving(config, vision_encoder_path="")
    images = torch.randn(1, 4, 3, 3, 16, 16)
    tokens = torch.ones(1, 3, dtype=torch.long)
    outputs = model(
        input_ids=tokens,
        pixel_values=images,
        gps_imu=torch.zeros(1, 6),
        gps_imu_mask=torch.tensor([False]),
    )
    assert outputs.control_outputs["discrete_probs"].shape[-1] == 13
