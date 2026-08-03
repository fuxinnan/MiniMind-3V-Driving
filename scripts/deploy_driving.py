"""Verified component export for MiniMind-Driving.

The autoregressive LLM remains a PyTorch component in the MVP.  ONNX export is
limited to stable tensor-only components: camera-temporal encoding and the
control/action head.  TensorRT is intentionally not advertised here.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from config.driving_config import DrivingConfig
from model.driving.camera_encoder import CameraEncoder
from model.driving.control_head import ControlHead
from model.driving.temporal_encoder import TemporalEncoder


class VisionTemporalComponent(nn.Module):
    def __init__(self, config: DrivingConfig):
        super().__init__()
        self.temporal = TemporalEncoder(
            hidden_size=768,
            num_history_frames=config.num_history_frames,
            mode="aggregate",
            aggregation_method="mean",
        )
        self.camera = CameraEncoder(
            ve_hidden_size=768,
            hidden_size=config.hidden_size,
            num_cameras=config.num_cameras,
            image_tokens_per_camera=config.image_tokens_per_camera,
        )

    def forward(self, clip_patch_features):
        return self.camera(self.temporal(clip_patch_features))


class ControlExportComponent(nn.Module):
    def __init__(self, head: ControlHead):
        super().__init__()
        self.head = head

    def forward(self, hidden_state):
        output = self.head(hidden_state)
        return (
            output["continuous_regression"],
            output["gear_logits"],
            output["discrete_logits"],
        )


def _verify(path: Path, inputs: Dict[str, torch.Tensor], expected):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is required for export verification") from exc
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(
        None, {name: value.detach().cpu().numpy() for name, value in inputs.items()}
    )
    expected_values = expected if isinstance(expected, tuple) else (expected,)
    errors = []
    for onnx_value, torch_value in zip(actual, expected_values):
        np.testing.assert_allclose(
            onnx_value, torch_value.detach().cpu().numpy(), rtol=1e-4, atol=1e-5
        )
        errors.append(float(np.max(np.abs(onnx_value - torch_value.detach().cpu().numpy()))))
    return errors


def export_components(output_dir: str, opset: int = 17):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = DrivingConfig(flash_attn=False)

    vision = VisionTemporalComponent(config).eval()
    vision_input = torch.randn(
        1, config.num_cameras, config.num_history_frames,
        config.image_tokens_per_camera, 768,
    )
    vision_path = output / "vision_temporal.onnx"
    torch.onnx.export(
        vision, (vision_input,), str(vision_path),
        input_names=["clip_patch_features"],
        output_names=["vision_tokens"],
        dynamic_axes={"clip_patch_features": {0: "batch"}, "vision_tokens": {0: "batch"}},
        opset_version=opset,
    )
    with torch.no_grad():
        vision_expected = vision(vision_input)
    vision_error = _verify(
        vision_path, {"clip_patch_features": vision_input}, vision_expected
    )

    control = ControlExportComponent(ControlHead(
        hidden_size=config.hidden_size,
        discrete_actions=config.discrete_actions,
        control_hidden_size=config.control_hidden_size,
    )).eval()
    hidden = torch.randn(2, config.hidden_size)
    control_path = output / "control_action_head.onnx"
    torch.onnx.export(
        control, (hidden,), str(control_path),
        input_names=["hidden_state"],
        output_names=["continuous", "gear_logits", "action_logits"],
        dynamic_axes={
            "hidden_state": {0: "batch"}, "continuous": {0: "batch"},
            "gear_logits": {0: "batch"}, "action_logits": {0: "batch"},
        },
        opset_version=opset,
    )
    with torch.no_grad():
        control_expected = control(hidden)
    control_error = _verify(control_path, {"hidden_state": hidden}, control_expected)
    manifest = {
        "scope": "MVP component export; LLM stays in PyTorch",
        "opset": opset,
        "components": {
            "vision_temporal": {"path": str(vision_path), "max_abs_error": vision_error},
            "control_action_head": {"path": str(control_path), "max_abs_error": control_error},
        },
    }
    (output / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def benchmark_control(iterations: int = 100):
    config = DrivingConfig()
    component = ControlExportComponent(ControlHead(
        hidden_size=config.hidden_size,
        discrete_actions=config.discrete_actions,
    )).eval()
    value = torch.randn(1, config.hidden_size)
    with torch.no_grad():
        for _ in range(10):
            component(value)
        started = time.perf_counter()
        for _ in range(iterations):
            component(value)
    return {"control_head_latency_ms": (time.perf_counter() - started) * 1000 / iterations}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["export", "benchmark"], default="export")
    parser.add_argument("--output", default="./out/export")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    result = (
        export_components(args.output, args.opset)
        if args.action == "export" else benchmark_control()
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
