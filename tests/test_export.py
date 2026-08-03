import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("transformers")

from scripts.deploy_driving import export_components


def test_component_onnx_matches_pytorch(tmp_path):
    manifest = export_components(str(tmp_path))
    assert manifest["components"]["vision_temporal"]["max_abs_error"][0] < 1e-4
    assert max(
        manifest["components"]["control_action_head"]["max_abs_error"]
    ) < 1e-4
