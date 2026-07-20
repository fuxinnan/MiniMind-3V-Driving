"""
驾驶 Prompt 模板引擎

为自动驾驶场景提供标准化的 Prompt 模板:
    - 场景描述模板
    - 决策解释模板
    - 控制输出模板
    - 多轮对话模板
    - 安全评估模板
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class DrivingPrompt:
    """驾驶 Prompt 数据结构"""
    user_content: str
    assistant_content: str
    system_content: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def to_messages(self) -> List[Dict[str, str]]:
        """转换为对话消息格式"""
        messages = []
        if self.system_content:
            messages.append({"role": "system", "content": self.system_content})
        messages.append({"role": "user", "content": self.user_content})
        messages.append({"role": "assistant", "content": self.assistant_content})
        return messages


class DrivingPromptTemplateEngine:
    """
    驾驶 Prompt 模板引擎

    提供多种预定义模板，支持变量替换和组合
    """

    # 系统提示
    SYSTEM_PROMPT = (
        "你是一个专业的自动驾驶决策模型。你的任务是分析多相机传感器输入，"
        "理解当前驾驶场景，并输出安全的驾驶决策和控制信号。"
        "\n\n请遵循以下原则:\n"
        "1. 安全第一: 始终优先避免碰撞\n"
        "2. 遵守交通规则: 严格按照交通信号和标志行驶\n"
        "3. 礼让行人: 遇到行人必须减速或停车让行\n"
        "4. 平稳驾驶: 避免急加速、急刹车和急转弯\n"
        "5. 合理预判: 提前观察并预判其他交通参与者的行为"
    )

    # 场景描述模板
    SCENE_DESCRIPTION_TEMPLATES = {
        "highway": (
            "当前场景: 高速公路\n"
            "车速: {speed} km/h\n"
            "天气: {weather}\n"
            "时间: {time_of_day}\n"
            "前方: {front_description}\n"
            "左侧: {left_description}\n"
            "右侧: {right_description}\n"
            "后方: {rear_description}\n\n"
            "请分析当前场景并给出驾驶决策。"
        ),
        "urban": (
            "当前场景: 城市道路\n"
            "车速: {speed} km/h\n"
            "天气: {weather}\n"
            "时间: {time_of_day}\n"
            "前方: {front_description}\n"
            "左侧: {left_description}\n"
            "右侧: {right_description}\n"
            "后方: {rear_description}\n"
            "附近车道: {lane_info}\n\n"
            "请分析当前场景并给出驾驶决策。"
        ),
        "intersection": (
            "当前场景: 十字路口\n"
            "车速: {speed} km/h\n"
            "天气: {weather}\n"
            "时间: {time_of_day}\n"
            "信号灯状态: {traffic_light}\n"
            "前方: {front_description}\n"
            "左侧: {left_description}\n"
            "右侧: {right_description}\n\n"
            "请分析当前场景并给出驾驶决策。"
        ),
        "parking": (
            "当前场景: 停车场\n"
            "车速: {speed} km/h\n"
            "天气: {weather}\n"
            "前方: {front_description}\n"
            "左侧: {left_description}\n"
            "右侧: {right_description}\n"
            "后方: {rear_description}\n\n"
            "请分析当前场景并给出驾驶决策。"
        ),
        "emergency": (
            "当前场景: 紧急情况\n"
            "车速: {speed} km/h\n"
            "天气: {weather}\n"
            "前方: {front_description}\n\n"
            "请分析当前场景并立即给出紧急驾驶决策。"
        ),
    }

    # 决策输出模板
    DECISION_RESPONSE_TEMPLATES = {
        "keep_lane": (
            "决策: 保持当前车道行驶\n"
            "转向角: {steering:.2f}\n"
            "油门: {throttle:.2f}\n"
            "刹车: {brake:.2f}\n"
            "挡位: {gear}\n"
            "原因: {reason}"
        ),
        "turn_left": (
            "决策: 向左转弯\n"
            "转向角: {steering:.2f}\n"
            "油门: {throttle:.2f}\n"
            "刹车: {brake:.2f}\n"
            "挡位: {gear}\n"
            "原因: {reason}"
        ),
        "turn_right": (
            "决策: 向右转弯\n"
            "转向角: {steering:.2f}\n"
            "油门: {throttle:.2f}\n"
            "刹车: {brake:.2f}\n"
            "挡位: {gear}\n"
            "原因: {reason}"
        ),
        "stop": (
            "决策: 停车\n"
            "转向角: {steering:.2f}\n"
            "油门: {throttle:.2f}\n"
            "刹车: {brake:.2f}\n"
            "挡位: {gear}\n"
            "原因: {reason}"
        ),
        "accelerate": (
            "决策: 加速\n"
            "转向角: {steering:.2f}\n"
            "油门: {throttle:.2f}\n"
            "刹车: {brake:.2f}\n"
            "挡位: {gear}\n"
            "原因: {reason}"
        ),
        "emergency_brake": (
            "决策: 紧急制动!\n"
            "转向角: {steering:.2f}\n"
            "油门: {throttle:.2f}\n"
            "刹车: {brake:.2f}\n"
            "挡位: {gear}\n"
            "原因: {reason}"
        ),
    }

    def __init__(
        self,
        use_system_prompt: bool = True,
        include_control_values: bool = True,
        language: str = "zh",  # "zh" / "en"
    ):
        self.use_system_prompt = use_system_prompt
        self.include_control_values = include_control_values
        self.language = language

    def build_scene_prompt(
        self,
        scene_type: str,
        speed: float = 0.0,
        weather: str = "sunny",
        time_of_day: str = "day",
        front_description: str = "无异常",
        left_description: str = "无异常",
        right_description: str = "无异常",
        rear_description: str = "无异常",
        lane_info: str = "",
        traffic_light: str = "green",
    ) -> DrivingPrompt:
        """
        构建场景描述 Prompt

        Args:
            scene_type: 场景类型
            speed: 当前车速 (km/h)
            weather: 天气
            time_of_day: 时间段
            front_description: 前方描述
            left_description: 左侧描述
            right_description: 右侧描述
            rear_description: 后方描述
            lane_info: 车道信息
            traffic_light: 信号灯状态

        Returns:
            DrivingPrompt
        """
        template = self.SCENE_DESCRIPTION_TEMPLATES.get(scene_type, self.SCENE_DESCRIPTION_TEMPLATES["urban"])

        user_content = template.format(
            speed=speed,
            weather=weather,
            time_of_day=time_of_day,
            front_description=front_description,
            left_description=left_description,
            right_description=right_description,
            rear_description=rear_description,
            lane_info=lane_info,
            traffic_light=traffic_light,
        )

        system_content = self.SYSTEM_PROMPT if self.use_system_prompt else None

        return DrivingPrompt(
            user_content=user_content,
            assistant_content="",
            system_content=system_content,
            metadata={
                "scene_type": scene_type,
                "speed": speed,
                "weather": weather,
                "time_of_day": time_of_day,
            },
        )

    def build_decision_response(
        self,
        action: str,
        steering: float = 0.0,
        throttle: float = 0.0,
        brake: float = 0.0,
        gear: int = 2,
        reason: str = "",
    ) -> str:
        """
        构建决策响应

        Args:
            action: 动作类型
            steering: 转向角
            throttle: 油门
            brake: 刹车
            gear: 挡位
            reason: 决策原因

        Returns:
            决策响应文本
        """
        template = self.DECISION_RESPONSE_TEMPLATES.get(action, self.DECISION_RESPONSE_TEMPLATES["keep_lane"])

        return template.format(
            steering=steering,
            throttle=throttle,
            brake=brake,
            gear=gear,
            reason=reason,
        )

    def build_conversation_pair(
        self,
        scene_type: str,
        action: str,
        controls: Dict[str, float],
        reason: str,
        **scene_kwargs,
    ) -> DrivingPrompt:
        """
        构建完整的对话对 (Prompt + Response)

        Args:
            scene_type: 场景类型
            action: 动作类型
            controls: 控制信号字典
            reason: 决策原因
            **scene_kwargs: 场景描述参数

        Returns:
            DrivingPrompt
        """
        # 构建用户 Prompt
        prompt = self.build_scene_prompt(scene_type, **scene_kwargs)

        # 构建助手响应
        response = self.build_decision_response(
            action=action,
            steering=controls.get("steering", 0.0),
            throttle=controls.get("throttle", 0.0),
            brake=controls.get("brake", 0.0),
            gear=int(controls.get("gear", 2)),
            reason=reason,
        )

        return DrivingPrompt(
            user_content=prompt.user_content,
            assistant_content=response,
            system_content=prompt.system_content,
            metadata=prompt.metadata,
        )

    def build_safety_eval_prompt(
        self,
        scene_type: str,
        action: str,
        controls: Dict[str, float],
        description: str,
    ) -> DrivingPrompt:
        """
        构建安全评估 Prompt

        用于 DPO/RLHF 阶段，评估驾驶决策的安全性
        """
        user_content = (
            f"场景: {scene_type}\n"
            f"决策动作: {action}\n"
            f"控制信号: 转向={controls.get('steering', 0):.2f}, "
            f"油门={controls.get('throttle', 0):.2f}, "
            f"刹车={controls.get('brake', 0):.2f}, "
            f"挡位={int(controls.get('gear', 2))}\n"
            f"场景描述: {description}\n\n"
            f"请评估此驾驶决策的安全性 (1-5分)，并说明理由。"
        )

        return DrivingPrompt(
            user_content=user_content,
            assistant_content="",
            system_content=self.SYSTEM_PROMPT,
            metadata={"scene_type": scene_type, "action": action},
        )

    def build_multi_turn_prompt(
        self,
        history: List[Dict[str, str]],
        current_scene: str,
        **scene_kwargs,
    ) -> DrivingPrompt:
        """
        构建多轮对话 Prompt

        用于训练模型的长期决策能力
        """
        history_text = "\n".join(
            f"{'用户' if h['role'] == 'user' else '系统'}: {h['content']}"
            for h in history
        )

        current_prompt = self.build_scene_prompt(current_scene, **scene_kwargs)

        user_content = (
            f"历史对话:\n{history_text}\n\n"
            f"当前场景:\n{current_prompt.user_content}\n\n"
            f"请根据历史信息给出当前决策。"
        )

        return DrivingPrompt(
            user_content=user_content,
            assistant_content="",
            system_content=self.SYSTEM_PROMPT,
            metadata={"history_length": len(history), "current_scene": current_scene},
        )

    def get_template_names(self) -> Dict[str, List[str]]:
        """获取所有可用模板名称"""
        return {
            "scene_descriptions": list(self.SCENE_DESCRIPTION_TEMPLATES.keys()),
            "decision_responses": list(self.DECISION_RESPONSE_TEMPLATES.keys()),
        }
