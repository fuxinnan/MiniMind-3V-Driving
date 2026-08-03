# MiniMind-3V-Driving

基于 MiniMind-3V 的自动驾驶垂直微调工程。项目将通用视觉语言模型迁移到驾驶场景，当前 MVP 已打通：

- 四路环视相机（前、左、右、后）与多帧时序输入
- CLIP patch 特征、时序聚合、多相机编码和 MiniMind 文本建模
- 转向、油门、刹车回归，挡位分类，13 类驾驶动作分类
- 文字决策依据、离线评估和 REST 推理接口
- nuScenes mini/full 转换、统一数据校验、训练与测试入口
- 视觉时序组件及控制头的 ONNX 导出和数值一致性验证

本 README 只描述仓库中已经实现的能力。TensorRT、闭环仿真碰撞率和生产级点云编码仍属于后续工作。

## 架构

```text
nuScenes / canonical JSONL
  ├─ 4 cameras × T frames ─ CLIP ─ TemporalEncoder ─ CameraEncoder ─┐
  ├─ LiDAR (optional) ───────────────────────────────────────────────┤
  ├─ Radar (optional) ───────────────────────────────────────────────┤
  └─ GPS/IMU (optional) ─────────────────────────────────────────────┘
                                      │ modality tokens + masks
                                      ▼
                                MiniMind Transformer
                         ┌────────────┼──────────────┐
                         ▼            ▼              ▼
                    text logits   controls        13 actions
```

可选传感器在 MVP 中使用轻量点特征投影和 mask。它们可进入模型，但尚不等同于完整的 PointPillars、BEVFusion 或闭环感知栈。

## 当前能力矩阵

| 能力 | 状态 | 说明 |
|---|---|---|
| 四相机三帧训练 | 已实现 | tensor 为 `[B,4,T,3,H,W]` |
| 时序与多相机编码 | 已实现 | 时间聚合后保留每相机 patch token |
| 连续控制与 13 类动作 | 已实现 | steering/throttle/brake 回归，gear/action 分类 |
| nuScenes 转换 | 已实现 | 支持 mini/trainval 自动检测或显式版本 |
| LiDAR/Radar/GPS-IMU | 实验性 | 数据引用、mask 和轻量融合已接入 |
| 离线控制/动作评估 | 已实现 | MAE/RMSE、Accuracy、macro Precision/Recall/F1 |
| TTC/碰撞/车道偏离 | 条件可用 | 只有提供轨迹证据才计算，否则标记不可用 |
| REST API | 已实现 | 严格四路输入，不用黑图静默替代错误请求 |
| ONNX | 已实现（分组件） | 视觉时序与控制动作头，含 ORT 对齐 |
| TensorRT | 未实现 | 不提供空壳导出或虚假 engine 文件 |

## 环境

推荐使用 Pixi：

```bash
pixi install
pixi run check-gpu
pixi run install-clip-vit
```

`pixi.toml` 支持 `linux-64` 与 `win-64`，并锁定 NumPy 1.26 以避免部分 PyTorch 扩展与 NumPy 2 ABI 不兼容。

## 数据契约

所有数据源先转换成 canonical JSONL：

```json
{
  "schema_version": "1.0",
  "scene": "urban",
  "prompt": "分析驾驶场景并给出安全控制",
  "response": "当前运动稳定，保持车道和安全车距。",
  "images": {
    "front": ["samples/CAM_FRONT/a.jpg", "samples/CAM_FRONT/b.jpg", "samples/CAM_FRONT/c.jpg"],
    "left": ["..."],
    "right": ["..."],
    "rear": ["..."]
  },
  "timestamp": 1532402927612460,
  "calibration": {"front": {"translation": [], "rotation": [], "camera_intrinsic": []}},
  "ego_state": {"translation": [], "rotation": [], "yaw": 0.0},
  "sensors": {"lidar": [], "radar": {}, "gps_imu": null},
  "controls": {"steering": 0.0, "throttle": 0.2, "brake": 0.0, "gear": 2},
  "action": "keep_lane",
  "label_source": "ego_motion_proxy"
}
```

`label_source` 必须说明标签来源。nuScenes 不直接提供标准化的方向盘、油门和刹车真值；转换器当前由 ego motion 生成代理标签并明确写入 `ego_motion_proxy`，不能把它当作车辆 CAN 真值。后续接入 CAN bus 时应写为 `can_bus`。

13 类动作固定为：

```text
keep_lane, turn_left, turn_right, stop, accelerate, decelerate,
yield, overtake, park, emergency_brake, follow_lane,
change_lane_left, change_lane_right
```

## 数据准备

### nuScenes mini

先将官方数据放到 `./data/nuscenes`：

```bash
pixi run prepare-nuscenes-mini
pixi run validate-data
```

或手动执行：

```bash
python scripts/prepare_driving_data.py \
  --source nuscenes \
  --nuscenes_root ./data/nuscenes \
  --nuscenes_version v1.0-mini \
  --output ./dataset/driving/processed
```

转换按 scene 划分 train/val/test，避免相邻帧跨集合造成泄漏。

### 合成 smoke 数据

```bash
pixi run prepare-driving-data
pixi run validate-synthetic-data
```

合成数据会生成可实际加载的四路图片，仅用于验证工程链路，不用于报告模型效果。

## 训练

```bash
pixi run train-driving-sft
```

核心多任务目标：

```text
L = L_text + λ_control × (L_steer,pedal + L_gear) + λ_action × L_action
```

缺失控制或动作标签时，collator 使用 label mask 屏蔽相应损失。训练默认要求真实 CLIP checkpoint；只有 smoke test 可使用：

```bash
pixi run train-driving-smoke
```

`--allow-random-vision` 只允许轻量 fallback，不代表有效驾驶模型。

## 评估

```bash
pixi run eval-driving
```

离线评估包括控制误差、动作分类、文本 token 准确率、分场景统计及场景覆盖。归一化 steering 的角度换算由 `steering_max_degrees` 配置，不再错误地把 `[-1,1]` 当作弧度。

安全指标分两类：

- 可直接计算：油门/刹车冲突、急刹率、控制平滑性
- 需要轨迹证据：TTC、碰撞率、车道偏离、轨迹舒适性

没有轨迹时，后一类不会参与综合安全分。

## REST API

```bash
python -m serve.driving_api_server \
  --model-path ./checkpoints/driving_sft_512.pth \
  --host 0.0.0.0 --port 8080 --device cuda
```

请求必须包含每路至少 `T` 帧有效路径、裸 base64 或 data URI：

```json
POST /api/drive
{
  "images": {
    "front": ["data:image/jpeg;base64,...", "...", "..."],
    "left": ["...", "...", "..."],
    "right": ["...", "...", "..."],
    "rear": ["...", "...", "..."]
  },
  "prompt": "分析当前场景并给出安全决策"
}
```

响应：

```json
{
  "text_response": "建议执行 keep_lane：...",
  "control": {"steering": 0.01, "throttle": 0.3, "brake": 0.0, "gear": 2},
  "action": "keep_lane",
  "action_probs": [0.1],
  "active_sensors": ["camera"],
  "latency_ms": 25.1,
  "model_version": "driving_sft_512"
}
```

服务启动时 checkpoint 不存在会直接失败，不会返回随机模型结果。

## 测试与导出

```bash
pixi run test-driving
pixi run deploy-export
```

导出物：

```text
out/export/
├── vision_temporal.onnx
├── control_action_head.onnx
└── export_manifest.json
```

导出过程使用 ONNX Runtime 对 PyTorch 输出做容差校验。MiniMind 自回归 LLM 在 MVP 中仍由 PyTorch 执行。

## 主要目录

```text
config/                    统一驾驶配置与动作语义
data/                      schema、dataset、nuScenes adapter、增强与校验
model/driving/             时序、多相机、传感器融合和多任务输出
trainer/train_driving_sft.py
evaluate/                  控制、动作、安全与场景评估
serve/inference_engine.py  单次多模态前向
serve/driving_api_server.py
scripts/                   数据转换、评估与 ONNX 导出
tests/                     CPU、模型、API 与导出测试
```

## 后续工作

- 读取并对齐真实 nuScenes CAN bus 控制标签
- 使用 PointPillars/BEV 编码替换轻量点云投影
- CARLA 闭环评估与碰撞、车道偏离基准
- KV cache 下的多模态自回归文字生成
- TensorRT 分组件构建、校准与性能验证

## License

MIT
