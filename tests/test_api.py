from types import SimpleNamespace

import pytest

pytest.importorskip("flask")

from serve.driving_api_server import create_app
from serve.inference_engine import InferenceInputError


class FakeEngine:
    device = "cpu"
    model_version = "test"
    model = SimpleNamespace(config=SimpleNamespace(
        num_cameras=4,
        camera_names=["front", "left", "right", "rear"],
        num_history_frames=3,
        discrete_actions=["keep_lane"],
    ))

    def infer(self, images, prompt, sensors=None):
        if set(images) != {"front", "left", "right", "rear"}:
            raise InferenceInputError("missing camera streams")
        return {
            "text_response": "保持车道",
            "control": {"steering": 0, "throttle": .2, "brake": 0, "gear": 2},
            "action": "keep_lane",
            "action_probs": [1.0],
            "active_sensors": ["camera"],
            "latency_ms": 1,
            "model_version": "test",
        }


def test_drive_contract():
    client = create_app(FakeEngine()).test_client()
    response = client.post("/api/drive", json={
        "images": {name: ["x"] * 3 for name in ("front", "left", "right", "rear")}
    })
    assert response.status_code == 200
    body = response.get_json()
    assert body["action"] == "keep_lane"
    assert set(body["control"]) == {"steering", "throttle", "brake", "gear"}


def test_drive_rejects_missing_camera():
    client = create_app(FakeEngine()).test_client()
    response = client.post("/api/drive", json={"images": {"front": ["x"] * 3}})
    assert response.status_code == 400
