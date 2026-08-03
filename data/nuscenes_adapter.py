"""nuScenes to canonical DrivingSample conversion."""

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from config.driving_config import NUSCENES_CAMERA_MAP


def _yaw(quaternion: Iterable[float]) -> float:
    w, x, y, z = quaternion
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class NuScenesAdapter:
    """Build four-camera temporal records using official nuScenes tables."""

    def __init__(
        self,
        dataroot: str,
        version: Optional[str] = None,
        num_frames: int = 3,
        verbose: bool = False,
    ):
        try:
            from nuscenes.nuscenes import NuScenes
        except ImportError as exc:
            raise RuntimeError(
                "nuScenes conversion requires nuscenes-devkit"
            ) from exc
        root = Path(dataroot)
        if version is None:
            version = (
                "v1.0-mini" if (root / "v1.0-mini").exists()
                else "v1.0-trainval"
            )
        self.root = root
        self.version = version
        self.num_frames = num_frames
        self.nusc = NuScenes(version=version, dataroot=str(root), verbose=verbose)
        self.can_bus = None
        try:
            from nuscenes.can_bus.can_bus_api import NuScenesCanBus
            self.can_bus = NuScenesCanBus(dataroot=str(root))
        except (ImportError, AssertionError, FileNotFoundError):
            pass

    def _history(self, token: str) -> List[str]:
        values = []
        current = token
        while current and len(values) < self.num_frames:
            row = self.nusc.get("sample_data", current)
            values.append(row["filename"])
            current = row["prev"]
        values.reverse()
        while values and len(values) < self.num_frames:
            values.insert(0, values[0])
        return values

    def _calibration(self, token: str) -> Dict[str, Any]:
        sample_data = self.nusc.get("sample_data", token)
        calibrated = self.nusc.get(
            "calibrated_sensor", sample_data["calibrated_sensor_token"]
        )
        return {
            "translation": calibrated["translation"],
            "rotation": calibrated["rotation"],
            "camera_intrinsic": calibrated.get("camera_intrinsic", []),
        }

    def _ego(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        front = self.nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        pose = self.nusc.get("ego_pose", front["ego_pose_token"])
        return {
            "translation": pose["translation"],
            "rotation": pose["rotation"],
            "yaw": _yaw(pose["rotation"]),
        }

    def _proxy_controls(
        self, sample: Dict[str, Any], ego: Dict[str, Any]
    ) -> Dict[str, float]:
        previous_token = sample.get("prev")
        speed_mps = 0.0
        yaw_rate = 0.0
        acceleration = 0.0
        if previous_token:
            previous = self.nusc.get("sample", previous_token)
            previous_ego = self._ego(previous)
            dt = max((sample["timestamp"] - previous["timestamp"]) / 1e6, 1e-3)
            delta = np.asarray(ego["translation"]) - np.asarray(
                previous_ego["translation"]
            )
            speed_mps = float(np.linalg.norm(delta[:2]) / dt)
            yaw_delta = math.atan2(
                math.sin(ego["yaw"] - previous_ego["yaw"]),
                math.cos(ego["yaw"] - previous_ego["yaw"]),
            )
            yaw_rate = yaw_delta / dt
            older_token = previous.get("prev")
            if older_token:
                older = self.nusc.get("sample", older_token)
                older_ego = self._ego(older)
                older_dt = max(
                    (previous["timestamp"] - older["timestamp"]) / 1e6, 1e-3
                )
                older_speed = np.linalg.norm(
                    np.asarray(previous_ego["translation"])[:2]
                    - np.asarray(older_ego["translation"])[:2]
                ) / older_dt
                acceleration = (speed_mps - older_speed) / dt
        return {
            "steering": float(np.clip(yaw_rate / 0.7, -1.0, 1.0)),
            "throttle": float(np.clip(acceleration / 3.0, 0.0, 1.0)),
            "brake": float(np.clip(-acceleration / 5.0, 0.0, 1.0)),
            "gear": 2.0 if speed_mps >= 0 else 0.0,
        }

    def _can_controls(
        self, scene_name: str, timestamp: int
    ) -> Optional[Dict[str, float]]:
        if self.can_bus is None:
            return None

        def nearest(message_name):
            try:
                messages = self.can_bus.get_messages(scene_name, message_name)
            except Exception:
                return None
            if not messages:
                return None
            return min(messages, key=lambda row: abs(row["utime"] - timestamp))

        steering_row = nearest("steeranglefeedback")
        vehicle_row = nearest("vehicle_monitor")
        if steering_row is None or vehicle_row is None:
            return None
        steering_value = steering_row.get("value", steering_row.get("steering_angle"))
        throttle = vehicle_row.get("throttle")
        brake = vehicle_row.get("brake")
        if None in (steering_value, throttle, brake):
            return None
        # CAN steering feedback is in degrees on the released nuScenes bus.
        steering = float(np.clip(float(steering_value) / 540.0, -1.0, 1.0))
        return {
            "steering": steering,
            "throttle": float(np.clip(float(throttle), 0.0, 1.0)),
            "brake": float(np.clip(float(brake), 0.0, 1.0)),
            "gear": 2.0,
        }

    @staticmethod
    def _action(controls: Dict[str, float]) -> str:
        if controls["brake"] > 0.7:
            return "emergency_brake"
        if controls["brake"] > 0.25:
            return "decelerate"
        if controls["steering"] < -0.2:
            return "turn_left"
        if controls["steering"] > 0.2:
            return "turn_right"
        if controls["throttle"] > 0.5:
            return "accelerate"
        return "keep_lane"

    def records(self) -> List[Dict[str, Any]]:
        from data.driving_prompt_template import DrivingPromptTemplateEngine
        prompt_engine = DrivingPromptTemplateEngine()
        records = []
        for sample in self.nusc.sample:
            scene = self.nusc.get("scene", sample["scene_token"])
            ego = self._ego(sample)
            controls = self._can_controls(scene["name"], sample["timestamp"])
            label_source = "can_bus"
            if controls is None:
                controls = self._proxy_controls(sample, ego)
                label_source = "ego_motion_proxy"
            action = self._action(controls)
            images, calibration = {}, {}
            for canonical, channel in NUSCENES_CAMERA_MAP.items():
                token = sample["data"][channel]
                images[canonical] = self._history(token)
                calibration[canonical] = self._calibration(token)
            sensors: Dict[str, Any] = {"lidar": [], "radar": {}, "gps_imu": None}
            lidar_token = sample["data"].get("LIDAR_TOP")
            if lidar_token:
                lidar = self.nusc.get("sample_data", lidar_token)
                sensors["lidar"] = [lidar["filename"]]
            for channel, token in sample["data"].items():
                if channel.startswith("RADAR_"):
                    radar = self.nusc.get("sample_data", token)
                    sensors["radar"][channel] = [radar["filename"]]
            scene_text = scene.get("description") or scene.get("name", "nuScenes")
            scene_category = classify_scene(scene_text)
            conversation = prompt_engine.build_conversation_pair(
                scene_type=scene_category,
                action=action,
                controls=controls,
                reason=decision_reason(action),
                front_description=scene_text,
            )
            records.append({
                "schema_version": "1.0",
                "scene": scene_category,
                "prompt": conversation.user_content,
                "response": conversation.assistant_content,
                "images": images,
                "timestamp": int(sample["timestamp"]),
                "calibration": calibration,
                "ego_state": ego,
                "sensors": sensors,
                "controls": controls,
                "action": action,
                "label_source": label_source,
                "metadata": {
                    "sample_token": sample["token"],
                    "scene_token": sample["scene_token"],
                    "nuscenes_version": self.version,
                },
            })
        return records


def classify_scene(text: str) -> str:
    value = text.lower()
    for needle, scene in (
        ("highway", "highway"), ("intersection", "intersection"),
        ("roundabout", "roundabout"), ("parking", "parking"),
        ("residential", "residential"), ("construction", "construction"),
        ("tunnel", "tunnel"),
    ):
        if needle in value:
            return scene
    return "urban"


def decision_reason(action: str) -> str:
    reasons = {
        "turn_left": "车辆航向向左变化，保持低速并向左调整。",
        "turn_right": "车辆航向向右变化，保持低速并向右调整。",
        "decelerate": "车辆正在减速，保持制动并观察前方风险。",
        "emergency_brake": "检测到强减速需求，立即制动以降低碰撞风险。",
        "accelerate": "道路状态允许，平稳增加油门。",
        "keep_lane": "当前运动状态稳定，保持车道和安全车距。",
    }
    return reasons.get(action, "保持安全驾驶并持续观察周边环境。")
