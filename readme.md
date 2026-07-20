# MiniMind-Driving: 自动驾驶端到端多模态大模型

基于 MiniMind 架构的自动驾驶端到端模型，支持多相机视觉输入 + 控制信号输出 + 自然语言决策解释。

> 源项目: [minimind](https://github.com/jingyaogong/minimind) | [minimind-v](https://github.com/jingyaogong/minimind-v)

---

## 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                        输入端 (多模态传感器)                      │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  前视相机    │  │  左视相机    │  │  右视相机    │          │
│  │ 1920×1080    │  │ 1920×1080    │  │ 1920×1080    │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                  │                   │
│  ┌──────┴───────┐  ┌─────┴───────┐          │                   │
│  │  后视相机    │  │  激光雷达   │  │  GPS/IMU    │          │
│  │ 1920×1080    │  │ (可选)      │  │ (可选)      │          │
│  └──────┬───────┘  └─────────────┘  └─────────────┘          │
│         │                                                     │
└─────────┼─────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                        视觉编码 pipeline                         │
│                                                                 │
│  CLIP ViT-B/16 → CameraEncoder → TemporalEncoder                │
│  (冻结)        (可训练)        (可训练)                          │
│                                                                 │
│  4 相机 × 3 帧 × 196 patches → 784 视觉 token                   │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                        传感器融合                                │
│                                                                 │
│  视觉特征 + 激光雷达 + 毫米波雷达 + GPS/IMU                      │
│         → SensorFusionModule → 融合特征                         │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                        MiniMind LLM                              │
│                                                                 │
│  融合特征注入文本序列 → Transformer Layers (GQA + RMSNorm)      │
│  → 文本 logits + 控制信号                                       │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                        输出端                                    │
│                                                                 │
│  ┌──────────────────────────────────────────────────────┐      │
│  │ 连续控制: [转向角, 油门, 刹车, 挡位]                  │      │
│  │ 离散决策: [保持车道, 左转, 右转, 停车, 避让, ...]     │      │
│  │ 自然语言: "前方有行人，减速避让"                      │      │
│  └──────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

## 快速开始

### 环境安装

```bash
# 使用 pixi 管理依赖
pixi install

# 验证 GPU 环境
pixi run check-gpu

# 下载 CLIP 视觉编码器
pixi run install-clip-vit
```

### 数据准备

```bash
# 生成合成数据 (开发测试用)
pixi run prepare-driving-data

# 数据校验
pixi run validate-data

# 场景覆盖分析
pixi run scene-coverage
```

### 模型训练

```bash
# 驾驶模型 SFT 训练
pixi run train-driving-sft

# 或手动指定参数
python trainer/train_driving_sft.py \
    --data_path ./dataset/driving/processed/train.jsonl \
    --epochs 3 \
    --batch_size 4 \
    --learning_rate 1e-6 \
    --max_seq_len 2048 \
    --num_cameras 4 \
    --num_history_frames 3 \
    --accumulation_steps 4 \
    --freeze_vision \
    --use_augmentation
```

### 模型评估

```bash
# 全面评估
pixi run eval-driving

# 或手动指定
python scripts/evaluate_driving.py \
    --data ./dataset/driving/processed/test.jsonl \
    --model ./checkpoints
```

### 推理服务

```bash
# 批量推理
pixi run batch-infer

# 启动 API 服务
pixi run serve-api

# 查看模型信息
curl http://localhost:8080/api/model_info

# 单次推理
curl -X POST http://localhost:8080/api/drive \
    -H "Content-Type: application/json" \
    -d '{"images": {"front": ["img1.jpg"]}, "prompt": "分析当前场景"}'
```

### 模型部署

```bash
# 导出 ONNX 格式
pixi run deploy-export

# 性能基准测试
pixi run deploy-benchmark

# 生成 Docker 部署文件
pixi run deploy-docker
```

---

## 项目结构

```
minimind-learn/
├── config/                          # 配置系统
│   ├── base_config.py               #   基础模型配置
│   ├── driving_config.py            #   自动驾驶领域配置
│   ├── sensor_config.py             #   传感器配置
│   └── lora_config.py               #   LoRA 配置
│
├── model/                           # 模型定义
│   ├── model_minimind.py            #   MiniMind LLM 核心
│   ├── model_vlm.py                 #   多模态 VLM
│   ├── model_lora.py                #   LoRA 微调
│   └── driving/                     #   驾驶模型
│       ├── model_driving.py         #     MiniMindDriving 主模型
│       ├── camera_encoder.py        #     多相机编码器
│       ├── temporal_encoder.py      #     时序编码器
│       ├── sensor_fusion_module.py  #     传感器融合模块
│       └── control_head.py          #     控制输出头
│
├── data/                            # 数据管理
│   ├── driving_dataset.py           #   驾驶数据集加载器
│   ├── sensor_fusion.py             #   多传感器融合工具
│   ├── data_augmentation.py         #   数据增强
│   ├── data_validator.py            #   数据校验
│   └── driving_prompt_template.py   #   Prompt 模板引擎
│
├── trainer/                         # 训练脚本
│   ├── train_driving_sft.py         #   驾驶模型 SFT
│   ├── train_pretrain.py            #   预训练
│   ├── train_full_sft.py            #   全参 SFT
│   ├── train_lora.py                #   LoRA 微调
│   ├── train_dpo.py                 #   DPO
│   ├── train_ppo.py                 #   PPO
│   ├── train_grpo.py                #   GRPO
│   ├── train_distillation.py        #   知识蒸馏
│   ├── train_pretrain_vlm.py        #   VLM 预训练
│   ├── train_sft_vlm.py             #   VLM SFT
│   └── train_utils.py               #   训练工具函数
│
├── evaluate/                        # 评估系统
│   ├── driving_evaluator.py         #   综合评估器
│   ├── control_accuracy.py          #   控制精度评估
│   ├── scene_coverage.py            #   场景覆盖率分析
│   └── safety_metrics.py            #   安全指标评估
│
├── serve/                           # 推理服务
│   ├── driving_api_server.py        #   REST API 服务
│   └── batch_driving_infer.py       #   批量推理
│
├── scripts/                         # 工具脚本
│   ├── train_tokenizer.py           #   Tokenizer 训练
│   ├── prepare_driving_data.py      #   数据准备
│   ├── evaluate_driving.py          #   模型评估
│   └── deploy_driving.py            #   模型部署
│
├── dataset/                         # 数据集
│   └── driving/                     #   驾驶数据
│       ├── raw/camera/              #     原始相机图像
│       ├── processed/               #     处理后数据
│       ├── control_labels/          #     控制标签
│       └── scenes/                  #     场景分类
│
├── doc/                             # 项目文档
│   ├── 0.环境搭建.md
│   ├── 1.train_tokenizer.md
│   ├── 2.DataLoader.md
│   ├── 3.模型构建.md
│   ├── 4.Pretrain.md
│   ├── 5.SFT.md
│   ├── 6.LoRA.md
│   ├── 7.PPO.md
│   ├── 8.DPO.md
│   ├── 9.白盒蒸馏.md
│   ├── 10.MoE.md
│   ├── 11.GRPO.md
│   ├── 12.minimind-v.md
│   └── 13.强化学习刨根问底.md
│
├── pixi.toml                        # 环境配置
├── eval_llm.py                      # 推理与对话
└── readme.md                        # 本文件
```

---

## 模型配置

### 传感器配置

| 传感器 | 默认配置 | 说明 |
|--------|----------|------|
| 前视相机 | 1920×1080 → 224×224 | 主视角 |
| 左/右/后视相机 | 1920×1080 → 224×224 | 环视 |
| 历史帧 | 3 帧 | 当前帧 + 2 历史帧 |
| 视觉编码器 | CLIP ViT-B/16 | 冻结 |
| 激光雷达 | 可选 | 16384 点 |
| 毫米波雷达 | 可选 | 100 检测点 |
| GPS/IMU | 可选 | 6 维 [lat, lon, alt, roll, pitch, yaw] |

### 控制输出

| 类型 | 维度 | 范围 | 说明 |
|------|------|------|------|
| 转向角 | 1 | [-1, 1] | 归一化 |
| 油门 | 1 | [0, 1] | 归一化 |
| 刹车 | 1 | [0, 1] | 归一化 |
| 挡位 | 1 | [0, 4] | 倒/N/D1/D2/D3 |
| 离散动作 | 13 | - | 保持车道/左转/右转/停车/等 |

### 场景分类

高速公路、城市道路、郊区道路、十字路口、环岛、停车场、隧道、施工区域、紧急情况、人行横道、学校区域、住宅区、环路、匝道

---

## 训练流程

```
预训练 (Pretrain)
    │
    ▼
VLM 预训练 (冻结 LLM)
    │
    ▼
驾驶 SFT (可训练: 视觉投影 + LLM + 控制头)
    │
    ▼
DPO / GRPO (偏好优化)
    │
    ▼
评估 → 部署
```

### 损失函数

```
L = L_text + λ_ctrl × L_control + λ_action × L_action
```

- `L_text`: 文本交叉熵损失
- `L_control`: 控制 MSE 损失
- `L_action`: 离散动作交叉熵损失
- `λ_ctrl`: 控制损失权重 (默认 0.3)
- `λ_action`: 动作损失权重 (默认 0.2)

---

## 评估指标

### 控制精度

| 指标 | 说明 | 阈值 |
|------|------|------|
| Steering MAE | 转向角平均绝对误差 | < 2° |
| Throttle MAE | 油门平均误差 | < 0.1 |
| Brake MAE | 刹车平均误差 | < 0.1 |
| 阈值内率 | 所有指标同时达标比例 | > 80% |

### 安全指标

| 指标 | 说明 |
|------|------|
| 碰撞率 | TTC < 1s 的比例 |
| 急刹率 | 刹车 > 0.8 的比例 |
| 舒适性评分 | 基于 Jerk 和横向加速度 |
| 车道偏离 | 偏离 > 0.5m 的比例 |

### 综合评分

```
综合评分 = 控制精度 × 30% + 决策准确性 × 25% + 安全性 × 25% + 场景覆盖 × 20%
```

---

## API 接口

### 健康检查

```
GET /api/health
```

### 模型信息

```
GET /api/model_info
```

### 单帧推理

```
POST /api/drive
Content-Type: application/json

{
    "images": {
        "front": ["base64_string", ...],
        "left": [...],
        "right": [...],
        "rear": [...]
    },
    "prompt": "分析当前驾驶场景",
    "max_tokens": 128,
    "temperature": 0.7
}
```

### 批量推理

```
POST /api/drive_batch
Content-Type: application/json

{
    "batch": [
        {"images": {...}, "prompt": "..."},
        {"images": {...}, "prompt": "..."}
    ]
}
```

---

## 文档索引

| 序号 | 章节 | 文档 |
|------|------|------|
| 📚 0 | 环境搭建 | [→](doc/0.环境搭建.md) |
| 🗂️ 1 | Tokenizer 训练 | [→](doc/1.train_tokenizer.md) |
| 📦 2 | DataLoader | [→](doc/2.DataLoader.md) |
| 🏗️ 3 | 模型构建 | [→](doc/3.模型构建.md) |
| 🚀 4 | Pretrain | [→](doc/4.Pretrain.md) |
| 🧑‍🏫 5 | SFT | [→](doc/5.SFT.md) |
| ⚡ 6 | LoRA | [→](doc/6.LoRA.md) |
| 🍅 7 | 强化学习 | [→](doc/13.强化学习刨根问底.md) |
| 🏆 8 | PPO | [→](doc/7.PPO.md) |
| ✅ 9 | DPO | [→](doc/8.DPO.md) |
| 🔬 10 | 白盒蒸馏 | [→](doc/9.白盒蒸馏.md) |
| 🧩 11 | MoE | [→](doc/10.MoE.md) |
| 🔄 12 | GRPO | [→](doc/11.GRPO.md) |
| 👀 13 | MiniMind-V | [→](doc/12.minimind-v.md) |
| 🚗 **驾驶** | **MiniMind-Driving** | **本文档** |

---

## License

MIT
