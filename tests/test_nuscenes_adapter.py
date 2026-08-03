from pathlib import Path

from data.nuscenes_adapter import NuScenesAdapter


class FakeNuScenes:
    def __init__(self):
        data = {
            "CAM_FRONT": "front2",
            "CAM_FRONT_LEFT": "left2",
            "CAM_FRONT_RIGHT": "right2",
            "CAM_BACK": "rear2",
        }
        self.sample = [{
            "token": "sample",
            "scene_token": "scene",
            "timestamp": 1_000_000,
            "prev": "",
            "data": data,
        }]
        self.tables = {
            ("scene", "scene"): {
                "name": "scene-0001", "description": "urban intersection"
            },
            ("ego_pose", "pose"): {
                "translation": [0, 0, 0], "rotation": [1, 0, 0, 0]
            },
            ("calibrated_sensor", "calib"): {
                "translation": [0, 0, 0],
                "rotation": [1, 0, 0, 0],
                "camera_intrinsic": [],
            },
        }
        for prefix in ("front", "left", "right", "rear"):
            for index in range(3):
                token = f"{prefix}{index}"
                self.tables[("sample_data", token)] = {
                    "filename": f"samples/{token}.jpg",
                    "prev": f"{prefix}{index - 1}" if index else "",
                    "calibrated_sensor_token": "calib",
                    "ego_pose_token": "pose",
                }

    def get(self, table, token):
        return self.tables[(table, token)]


def test_adapter_builds_four_camera_temporal_proxy_record():
    adapter = NuScenesAdapter.__new__(NuScenesAdapter)
    adapter.root = Path(".")
    adapter.version = "v1.0-mini"
    adapter.num_frames = 3
    adapter.nusc = FakeNuScenes()
    adapter.can_bus = None
    record = adapter.records()[0]
    assert record["scene"] == "intersection"
    assert record["label_source"] == "ego_motion_proxy"
    assert all(len(frames) == 3 for frames in record["images"].values())
    assert set(record["images"]) == {"front", "left", "right", "rear"}
