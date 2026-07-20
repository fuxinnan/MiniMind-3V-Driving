"""
自动驾驶领域配置
定义多相机、传感器、控制输出、场景覆盖、评估指标等自动驾驶专属配置
"""

from typing import List, Optional, Tuple, Dict


# ========== 传感器配置 ==========
# 定义输入端的多模态传感器参数

SENSOR_CONFIG: Dict = {
    # --- 相机配置 ---
    "num_cameras": 4,                    # 相机数量 (前/左/右/后)
    "camera_names": ["front", "left", "right", "rear"],
    "camera_resolution": (1920, 1080),   # 原始分辨率
    "camera_input_size": (224, 224),     # 输入到视觉编码器的尺寸 (CLIP 默认 224x224)
    "image_tokens_per_camera": 196,      # 每相机的 image token 数 (CLIP ViT-B/16: 14x14=196 patches)
    "total_image_tokens": 784,           # 4 相机 × 196

    # --- 时序配置 ---
    "num_history_frames": 3,             # 使用的历史帧数 (含当前帧)
    "frame_skip": 1,                     # 帧间隔 (每 N 帧采样一次)
    "frame_spacing": 1,                  # 帧之间的时间间隔 (帧数)

    # --- 可选传感器 ---
    "enable_lidar": False,               # 是否启用激光雷达
    "lidar_num_points": 16384,           # 激光雷达点数
    "lidar_encoding": "range_image",     # 编码方式: range_image / point_cloud / bev
    "lidar_hidden_size": 512,            # 激光雷达特征维度

    "enable_radar": False,               # 是否启用毫米波雷达
    "radar_num_detections": 100,         # 雷达检测点数
    "radar_hidden_size": 64,             # 雷达特征维度

    "enable_gps_imu": False,             # 是否启用 GPS/IMU
    "gps_imu_dims": 6,                   # [lat, lon, alt, roll, pitch, yaw]
}


# ========== 控制输出配置 ==========
# 定义输出端的控制信号格式

CONTROL_CONFIG: Dict = {
    # --- 输出类型 ---
    "control_type": "both",              # continuous / discrete / both
    # continuous: 输出连续控制信号 [转向角, 油门, 刹车, 挡位]
    # discrete:   输出离散动作类别 [保持车道, 左转, 右转, 停车, ...]
    # both:       同时输出两种

    # --- 连续控制 ---
    "continuous_dims": 4,                # 连续控制维度
    "continuous_labels": ["steering", "throttle", "brake", "gear"],
    "steering_range": (-1.0, 1.0),       # 转向角归一化范围 [-1, 1]
    "throttle_range": (0.0, 1.0),        # 油门归一化范围 [0, 1]
    "brake_range": (0.0, 1.0),           # 刹车归一化范围 [0, 1]
    "gear_range": (0, 4),                # 挡位 [0=倒, 1=N, 2=D1, 3=D2, 4=D3]

    # --- 离散控制 ---
    "discrete_actions": [
        "keep_lane", "turn_left", "turn_right",
        "stop", "accelerate", "decelerate",
        "yield", "overtake", "park",
        "emergency_brake", "follow_lane", "change_lane_left", "change_lane_right",
    ],
    "num_discrete_actions": 13,
}


# ========== 训练策略配置 ==========
# 定义自动驾驶微调的训练策略

TRAINING_CONFIG: Dict = {
    # --- SFT 训练 ---
    "sft_learning_rate": 1e-6,           # SFT 学习率 (较低以防遗忘)
    "sft_epochs": 3,                     # SFT 轮数
    "sft_batch_size": 4,                 # SFT batch size (多模态显存占用大)
    "sft_max_seq_len": 2048,             # SFT 最大序列长度

    # --- 冻结策略 ---
    "freeze_vision_encoder": True,       # 冻结 CLIP 视觉编码器
    "freeze_camera_encoder": True,       # 冻结相机编码器
    "freeze_first_layers": 0,            # 冻结前 N 层 LLM (0=不冻结)
    "train_vision_proj": True,           # 训练视觉投影层
    "train_llm": True,                   # 训练 LLM 主体
    "train_control_head": True,          # 训练控制输出头

    # --- 混合精度 ---
    "mixed_precision": "bfloat16",       # float32 / float16 / bfloat16

    # --- 梯度 ---
    "gradient_accumulation": 4,          # 梯度累积步数
    "gradient_clip": 1.0,                # 梯度裁剪阈值

    # --- 学习率调度 ---
    "lr_scheduler": "warmup_cosine",     # warmup_cosine / cosine / linear
    "warmup_ratio": 0.05,                # Warmup 比例
    "min_lr_ratio": 0.1,                 # 最低学习率比例

    # --- LoRA ---
    "use_lora": False,                   # 是否使用 LoRA
    "lora_rank": 16,                     # LoRA rank
    "lora_alpha": 32,                    # LoRA alpha
    "lora_dropout": 0.05,                # LoRA dropout
    "lora_target_modules": "all_linear", # 目标模块: all_linear / attention / mlp / qkv
}


# ========== 场景覆盖配置 ==========
# 定义自动驾驶场景分类和最低数据要求

SCENE_CONFIG: Dict = {
    "scene_categories": [
        "highway",               # 高速公路
        "urban",                 # 城市道路
        "suburban",              # 郊区道路
        "intersection",          # 十字路口
        "roundabout",            # 环岛
        "parking",               # 停车场
        "tunnel",                # 隧道
        "construction",          # 施工区域
        "emergency",             # 紧急情况
        "pedestrian_cross",      # 人行横道
        "school_zone",           # 学校区域
        "residential",           # 住宅区
        "ringroad",              # 环路
        "onramp_offramp",        # 匝道
    ],
    "min_samples_per_scene": 1000,     # 每场景最少样本数
    "min_total_samples": 50000,        # 总最少样本数
}


# ========== 评估配置 ==========
# 定义自动驾驶评估指标和阈值

EVALUATION_CONFIG: Dict = {
    # --- 控制误差阈值 ---
    "control_threshold": {
        "steering_deg": 2.0,           # 转向角误差 < 2°
        "speed_kmh": 3.0,              # 速度误差 < 3 km/h
        "lat_offset_m": 0.3,           # 横向偏移 < 0.3m
        "long_offset_m": 0.5,          # 纵向偏移 < 0.5m
        "throttle_diff": 0.1,          # 油门误差 < 0.1
        "brake_diff": 0.1,             # 刹车误差 < 0.1
    },

    # --- 安全指标 ---
    "safety_metrics": [
        "collision_rate",              # 碰撞率
        "ocd_lane_deviation",          # 车道偏离距离
        "traffic_rule_violation",      # 交通规则违反次数
        "comfort_metrics",             # 舒适性指标
        "hard_braking_rate",           # 急刹率
        " Jerk",                       # 加加速度 (舒适性)
    ],

    # --- 场景评估 ---
    "scene_evaluation": {
        "per_scene_metrics": True,     # 按场景统计指标
        "min_scene_samples": 100,      # 每场景最少样本用于评估
    },
}


# ========== 数据格式配置 ==========
# 定义训练数据的 JSONL 格式

DATA_FORMAT_CONFIG: Dict = {
    # SFT 数据格式
    "sft_fields": {
        "required": ["scene", "prompt", "response", "images", "controls"],
        "optional": ["timestamp", "speed", "action", "weather", "time_of_day"],
    },

    # 图像字段格式
    "image_format": {
        "structure": "dict_of_lists",  # {"front": ["img1.jpg", ...], "left": [...]}
        "num_frames_per_camera": 3,    # 每相机的帧数
        "frame_naming": "{timestamp}_{camera}.jpg",
    },

    # 控制标签格式
    "control_format": {
        "structure": "dict",           # {"steering": 0.05, "throttle": 0.3, ...}
        "continuous_keys": ["steering", "throttle", "brake", "gear"],
        "discrete_key": "action",      # 离散动作字段名
    },
}


class DrivingConfig:
    """
    自动驾驶领域配置管理类
    整合所有自动驾驶相关配置，提供统一的访问接口
    """

    def __init__(
        self,
        # 传感器
        num_cameras: int = 4,
        camera_names: Optional[List[str]] = None,
        camera_input_size: Tuple[int, int] = (224, 224),
        image_tokens_per_camera: int = 196,
        num_history_frames: int = 3,
        frame_skip: int = 1,
        enable_lidar: bool = False,
        enable_radar: bool = False,
        enable_gps_imu: bool = False,

        # 控制输出
        control_type: str = "both",
        continuous_dims: int = 4,
        discrete_actions: Optional[List[str]] = None,

        # 训练策略
        freeze_vision_encoder: bool = True,
        freeze_first_layers: int = 0,
        use_lora: bool = False,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: str = "all_linear",

        # 场景
        scene_categories: Optional[List[str]] = None,
        min_samples_per_scene: int = 1000,

        # 数据
        sft_max_seq_len: int = 2048,
        sft_batch_size: int = 4,
        sft_learning_rate: float = 1e-6,
        sft_epochs: int = 3,

        # 评估
        control_threshold: Optional[Dict] = None,
    ):
        self.camera_names = camera_names or SENSOR_CONFIG["camera_names"]
        self.num_cameras = num_cameras
        self.camera_input_size = camera_input_size
        self.image_tokens_per_camera = image_tokens_per_camera
        self.total_image_tokens = num_cameras * image_tokens_per_camera
        self.num_history_frames = num_history_frames
        self.frame_skip = frame_skip
        self.enable_lidar = enable_lidar
        self.enable_radar = enable_radar
        self.enable_gps_imu = enable_gps_imu

        self.control_type = control_type
        self.continuous_dims = continuous_dims
        self.discrete_actions = discrete_actions or CONTROL_CONFIG["discrete_actions"]
        self.num_discrete_actions = len(self.discrete_actions)

        self.freeze_vision_encoder = freeze_vision_encoder
        self.freeze_first_layers = freeze_first_layers
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules

        self.scene_categories = scene_categories or SCENE_CONFIG["scene_categories"]
        self.min_samples_per_scene = min_samples_per_scene

        self.sft_max_seq_len = sft_max_seq_len
        self.sft_batch_size = sft_batch_size
        self.sft_learning_rate = sft_learning_rate
        self.sft_epochs = sft_epochs

        self.control_threshold = control_threshold or EVALUATION_CONFIG["control_threshold"]

    def to_dict(self) -> Dict:
        return {
            "num_cameras": self.num_cameras,
            "camera_names": self.camera_names,
            "camera_input_size": self.camera_input_size,
            "image_tokens_per_camera": self.image_tokens_per_camera,
            "total_image_tokens": self.total_image_tokens,
            "num_history_frames": self.num_history_frames,
            "frame_skip": self.frame_skip,
            "enable_lidar": self.enable_lidar,
            "enable_radar": self.enable_radar,
            "enable_gps_imu": self.enable_gps_imu,
            "control_type": self.control_type,
            "continuous_dims": self.continuous_dims,
            "discrete_actions": self.discrete_actions,
            "num_discrete_actions": self.num_discrete_actions,
            "freeze_vision_encoder": self.freeze_vision_encoder,
            "freeze_first_layers": self.freeze_first_layers,
            "use_lora": self.use_lora,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_modules": self.lora_target_modules,
            "scene_categories": self.scene_categories,
            "min_samples_per_scene": self.min_samples_per_scene,
            "sft_max_seq_len": self.sft_max_seq_len,
            "sft_batch_size": self.sft_batch_size,
            "sft_learning_rate": self.sft_learning_rate,
            "sft_epochs": self.sft_epochs,
            "control_threshold": self.control_threshold,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict):
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__init__.__code__.co_varnames})

    def __repr__(self):
        return f"DrivingConfig(num_cameras={self.num_cameras}, " \
               f"num_history_frames={self.num_history_frames}, " \
               f"control_type={self.control_type})"
