"""Single source of truth for the driving domain.

The class deliberately contains the VLM/HuggingFace-facing attributes as well
as data parameters.  Model configs can inherit it or copy ``to_dict()`` without
redefining camera, sensor, or control semantics.
"""

from typing import Any, Dict, List, Optional, Tuple

try:
    from transformers import PretrainedConfig
except ImportError:  # Data preparation must not require transformers.
    class PretrainedConfig:  # type: ignore[no-redef]
        model_type = "driving"

        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        def to_dict(self):
            return dict(self.__dict__)


CAMERA_NAMES: Tuple[str, ...] = ("front", "left", "right", "rear")
NUSCENES_CAMERA_MAP: Dict[str, str] = {
    "front": "CAM_FRONT",
    "left": "CAM_FRONT_LEFT",
    "right": "CAM_FRONT_RIGHT",
    "rear": "CAM_BACK",
}
CONTINUOUS_CONTROL_KEYS: Tuple[str, ...] = (
    "steering", "throttle", "brake", "gear"
)
DISCRETE_ACTIONS: Tuple[str, ...] = (
    "keep_lane", "turn_left", "turn_right", "stop", "accelerate",
    "decelerate", "yield", "overtake", "park", "emergency_brake",
    "follow_lane", "change_lane_left", "change_lane_right",
)
ACTION_TO_ID: Dict[str, int] = {
    action: index for index, action in enumerate(DISCRETE_ACTIONS)
}
CONTROL_RANGES: Dict[str, Tuple[float, float]] = {
    "steering": (-1.0, 1.0),
    "throttle": (0.0, 1.0),
    "brake": (0.0, 1.0),
    "gear": (0.0, 4.0),
}

SENSOR_CONFIG: Dict[str, Any] = {
    "num_cameras": len(CAMERA_NAMES),
    "camera_names": list(CAMERA_NAMES),
    "camera_resolution": (1920, 1080),
    "camera_input_size": (224, 224),
    "image_tokens_per_camera": 196,
    "total_image_tokens": 784,
    "num_history_frames": 3,
    "frame_skip": 1,
    "frame_spacing": 1,
    "enable_lidar": False,
    "lidar_num_points": 16384,
    "lidar_point_dims": 5,
    "lidar_encoding": "point_cloud",
    "lidar_hidden_size": 512,
    "enable_radar": False,
    "radar_num_detections": 100,
    "radar_point_dims": 18,
    "radar_hidden_size": 64,
    "enable_gps_imu": False,
    "gps_imu_dims": 6,
}

CONTROL_CONFIG: Dict[str, Any] = {
    "control_type": "both",
    "continuous_dims": len(CONTINUOUS_CONTROL_KEYS),
    "continuous_labels": list(CONTINUOUS_CONTROL_KEYS),
    "control_ranges": CONTROL_RANGES,
    "discrete_actions": list(DISCRETE_ACTIONS),
    "num_discrete_actions": len(DISCRETE_ACTIONS),
}

TRAINING_CONFIG: Dict[str, Any] = {
    "sft_learning_rate": 1e-6,
    "sft_epochs": 3,
    "sft_batch_size": 4,
    "sft_max_seq_len": 2048,
    "freeze_vision_encoder": True,
    "freeze_camera_encoder": True,
    "freeze_first_layers": 0,
    "train_vision_proj": True,
    "train_llm": True,
    "train_control_head": True,
    "mixed_precision": "bfloat16",
    "gradient_accumulation": 4,
    "gradient_clip": 1.0,
    "lr_scheduler": "warmup_cosine",
    "warmup_ratio": 0.05,
    "min_lr_ratio": 0.1,
    "use_lora": False,
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": "all_linear",
}

SCENE_CONFIG: Dict[str, Any] = {
    "scene_categories": [
        "highway", "urban", "suburban", "intersection", "roundabout",
        "parking", "tunnel", "construction", "emergency",
        "pedestrian_cross", "school_zone", "residential", "ringroad",
        "onramp_offramp",
    ],
    "min_samples_per_scene": 1000,
    "min_total_samples": 50000,
}

EVALUATION_CONFIG: Dict[str, Any] = {
    "control_threshold": {
        "steering_deg": 2.0, "speed_kmh": 3.0, "lat_offset_m": 0.3,
        "long_offset_m": 0.5, "throttle_diff": 0.1, "brake_diff": 0.1,
    },
    "safety_metrics": [
        "collision_rate", "lane_deviation", "traffic_rule_violation",
        "comfort_metrics", "hard_braking_rate", "jerk",
    ],
}

DATA_FORMAT_CONFIG: Dict[str, Any] = {
    "schema_version": "1.0",
    "sft_fields": {
        "required": [
            "scene", "prompt", "response", "images", "timestamp",
            "calibration", "ego_state", "controls", "label_source",
        ],
        "optional": [
            "sensors", "action", "weather", "time_of_day", "speed",
        ],
    },
    "image_format": {
        "structure": "dict_of_lists",
        "camera_names": list(CAMERA_NAMES),
        "num_frames_per_camera": SENSOR_CONFIG["num_history_frames"],
        "frame_order": "oldest_to_newest",
    },
    "control_format": {
        "structure": "dict_or_null",
        "continuous_keys": list(CONTINUOUS_CONTROL_KEYS),
        "discrete_key": "action",
        "label_sources": ["human", "can_bus", "ego_motion_proxy", "synthetic"],
    },
}


class DrivingConfig(PretrainedConfig):
    """Complete serializable domain/VLM configuration."""

    model_type = "driving"

    def __init__(
        self,
        num_cameras: int = len(CAMERA_NAMES),
        camera_names: Optional[List[str]] = None,
        camera_input_size: Tuple[int, int] = (224, 224),
        image_tokens_per_camera: int = 196,
        num_history_frames: int = 3,
        frame_skip: int = 1,
        enable_lidar: bool = False,
        lidar_num_points: int = 16384,
        lidar_point_dims: int = 5,
        lidar_encoding: str = "point_cloud",
        lidar_hidden_size: int = 512,
        enable_radar: bool = False,
        radar_num_detections: int = 100,
        radar_point_dims: int = 18,
        radar_hidden_size: int = 64,
        enable_gps_imu: bool = False,
        gps_imu_dims: int = 6,
        sensor_fusion_method: str = "concat",
        control_type: str = "both",
        continuous_dims: int = len(CONTINUOUS_CONTROL_KEYS),
        discrete_actions: Optional[List[str]] = None,
        control_hidden_size: int = 256,
        freeze_vision_encoder: bool = True,
        freeze_first_layers: int = 0,
        vision_encoder_path: str = "./model/vision_model/clip-vit-base-patch16",
        hidden_size: int = 512,
        num_hidden_layers: int = 8,
        num_attention_heads: int = 8,
        num_key_value_heads: Optional[int] = 2,
        intermediate_size: Optional[int] = None,
        vocab_size: int = 6400,
        max_position_embeddings: int = 32768,
        max_seq_len: int = 2048,
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-5,
        dropout: float = 0.0,
        rope_theta: float = 1e6,
        inference_rope_scaling: bool = False,
        flash_attn: bool = True,
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        aux_loss_alpha: float = 0.01,
        scoring_func: str = "softmax",
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        image_special_token: Optional[str] = None,
        sft_batch_size: int = 4,
        sft_learning_rate: float = 1e-6,
        sft_epochs: int = 3,
        loss_control_weight: float = 0.3,
        loss_action_weight: float = 0.2,
        **kwargs,
    ):
        cameras = list(camera_names or CAMERA_NAMES)
        actions = list(discrete_actions or DISCRETE_ACTIONS)
        num_key_value_heads = num_key_value_heads or min(2, num_attention_heads)
        if num_cameras != len(cameras):
            raise ValueError("num_cameras must equal len(camera_names)")
        if tuple(actions) != DISCRETE_ACTIONS:
            raise ValueError(
                "The 13 driving actions are fixed domain semantics; "
                "use DISCRETE_ACTIONS instead of redefining them."
            )
        if continuous_dims != len(CONTINUOUS_CONTROL_KEYS):
            raise ValueError("continuous_dims must match CONTINUOUS_CONTROL_KEYS")

        super().__init__(**kwargs)
        values = locals().copy()
        values.pop("self")
        values.pop("kwargs")
        for key, value in values.items():
            setattr(self, key, value)
        self.camera_names = cameras
        self.discrete_actions = actions
        self.num_discrete_actions = len(actions)
        self.total_image_tokens = num_cameras * image_tokens_per_camera
        self.image_special_token = image_special_token or (
            "@" * image_tokens_per_camera
        )
        self.rope_scaling = {
            "beta_fast": 4,
            "beta_slow": 1,
            "factor": 4,
            "original_max_position_embeddings": max_seq_len,
            "attention_factor": 1.0,
            "type": "yarn",
        } if inference_rope_scaling else None
        # Legacy training names remain available.
        self.sft_max_seq_len = max_seq_len

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs):
        values = dict(config_dict)
        values.update(kwargs)
        return cls(**values)
