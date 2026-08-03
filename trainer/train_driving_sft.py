"""
MiniMind-Driving SFT 训练脚本

监督微调自动驾驶端到端模型:
    - 多相机图像输入
    - 文本对话 + 控制信号输出
    - 支持冻结策略
    - 支持 LoRA 微调
    - 支持 DDP 分布式训练
"""

import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from model.driving.model_driving import MiniMindDriving, DrivingConfig
from data.driving_dataset import DrivingDataCollator, DrivingSFTDataset
from data.data_augmentation import DrivingDataAugmentation
from trainer.train_utils import (
    get_lr, Logger, is_main_process, init_distributed_mode,
    setup_seed, lm_checkpoint, SkipBatchSampler,
)

warnings.filterwarnings('ignore')


def train_epoch(
    epoch,
    loader,
    iters,
    start_step=0,
    wandb=None,
    use_augmentation=False,
    augmentation=None,
):
    """
    训练一个 epoch

    损失函数:
        L = L_text + lambda_ctrl * L_control + lambda_action * L_action
    """
    start_time = time.time()

    for step, batch in enumerate(loader, start=start_step + 1):
        input_ids = batch["input_ids"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        pixel_values = batch["pixel_values"].to(args.device)
        control_labels = batch.get("control_labels")
        action_labels = batch.get("action_labels")
        sensor_keys = (
            "lidar_pointcloud", "radar_data", "gps_imu", "lidar_mask",
            "radar_mask", "gps_imu_mask", "control_label_mask",
            "action_label_mask",
        )
        sensors = {
            key: batch[key].to(args.device)
            for key in sensor_keys if key in batch
        }

        if control_labels is not None:
            control_labels = control_labels.to(args.device)
        if action_labels is not None:
            action_labels = action_labels.to(args.device)

        # 数据增强
        if use_augmentation and augmentation and pixel_values is not None:
            pixel_values = augmentation.augment_images_batch(pixel_values)

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                control_labels=control_labels,
                action_labels=action_labels,
                labels=input_ids.masked_fill(attention_mask == 0, -100),
                **sensors,
            )
            if outputs.loss is None:
                raise RuntimeError("model returned no multitask loss")
            zero_loss = torch.zeros((), device=args.device)
            loss_text = outputs.losses.get("text")
            loss_ctrl = outputs.losses.get("control")
            loss_action = outputs.losses.get("action")
            loss_text = zero_loss if loss_text is None else loss_text
            loss_ctrl = zero_loss if loss_ctrl is None else loss_ctrl
            loss_action = zero_loss if loss_action is None else loss_action
            total_loss = outputs.loss
            total_loss = total_loss / args.accumulation_steps

        scaler.scale(total_loss).backward()

        if step % args.accumulation_steps == 0 or step == iters:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = total_loss.item() * args.accumulation_steps
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60

            Logger(
                f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}) '
                f'loss:{current_loss:.6f} text_loss:{loss_text.item():.4f} '
                f'ctrl_loss:{loss_ctrl.item():.4f} action_loss:{loss_action.item():.4f} '
                f'lr:{current_lr:.12f} time:{eta_min}min:'
            )

            if wandb:
                wandb.log({
                    "loss": current_loss,
                    "loss_text": loss_text.item(),
                    "loss_ctrl": loss_ctrl.item(),
                    "loss_action": loss_action.item(),
                    "lr": current_lr,
                    "epoch_Time": eta_min,
                })

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if model.config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{model.config.hidden_size}{moe_suffix}.pth'

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            clean_state_dict = {
                key: value for key, value in state_dict.items()
                if not key.startswith('vision_encoder.')
            }
            clean_state_dict = {k: v.half().cpu() for k, v in clean_state_dict.items()}
            torch.save(clean_state_dict, ckp)

            lm_checkpoint(
                model.config,
                weight=args.save_weight,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir=args.save_dir,
                scaler=scaler,
            )
            model.train()
            del state_dict, clean_state_dict

        del input_ids, attention_mask, pixel_values, outputs, total_loss
        torch.cuda.empty_cache()


@torch.no_grad()
def validate_epoch(loader):
    model.eval()
    totals = {"loss": 0.0, "text": 0.0, "control": 0.0, "action": 0.0}
    count = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        model_inputs = {
            key: batch[key].to(args.device)
            for key in (
                "pixel_values", "lidar_pointcloud", "radar_data", "gps_imu",
                "lidar_mask", "radar_mask", "gps_imu_mask",
                "control_label_mask", "action_label_mask",
            ) if key in batch
        }
        for key in ("control_labels", "action_labels"):
            if batch.get(key) is not None:
                model_inputs[key] = batch[key].to(args.device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids.masked_fill(attention_mask == 0, -100),
            **model_inputs,
        )
        totals["loss"] += float(outputs.loss)
        for key in ("text", "control", "action"):
            value = outputs.losses.get(key)
            totals[key] += float(value) if value is not None else 0.0
        count += 1
    model.train()
    metrics = {key: value / max(count, 1) for key, value in totals.items()}
    Logger(
        "Validation: " + " ".join(
            f"{key}={value:.4f}" for key, value in metrics.items()
        )
    )
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind-Driving SFT")
    parser.add_argument("--save_dir", type=str, default="./out/checkpoints", help="模型保存目录")
    parser.add_argument('--save_weight', default='driving_sft', type=str, help="保存权重前缀")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=50, help="日志间隔")
    parser.add_argument("--save_interval", type=int, default=200, help="保存间隔")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="最大序列长度")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数")
    parser.add_argument('--num_cameras', default=4, type=int, help="相机数量")
    parser.add_argument('--num_history_frames', default=3, type=int, help="历史帧数")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE")
    parser.add_argument("--data_path", type=str, default="../dataset/driving/processed/train.jsonl",
                        help="训练数据路径")
    parser.add_argument("--images_path", type=str, default="../dataset/driving/raw/camera",
                        help="图像根目录")
    parser.add_argument("--val_data_path", type=str, default=None,
                        help="可选验证集 JSONL")
    parser.add_argument('--from_weight', default='none', type=str,
                        help="基于哪个权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1],
                        help="是否续训")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用 wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Driving-SFT",
                        help="wandb 项目名")
    parser.add_argument("--control_loss_weight", type=float, default=0.3,
                        help="控制损失权重")
    parser.add_argument("--action_loss_weight", type=float, default=0.2,
                        help="动作损失权重")
    parser.add_argument("--freeze_vision", action="store_true",
                        help="冻结视觉编码器")
    parser.add_argument("--freeze_first_layers", type=int, default=0,
                        help="冻结前 N 层 LLM")
    parser.add_argument("--use_augmentation", action="store_true",
                        help="使用数据增强")
    parser.add_argument("--allow-random-vision", action="store_true",
                        help="仅 smoke test：CLIP 缺失时允许轻量 fallback")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置模型参数 ==========
    os.makedirs(args.save_dir, exist_ok=True)
    driving_config = DrivingConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_position_embeddings=(
            args.max_seq_len
            + args.num_cameras * 196
            + 3  # optional sensor summary tokens
        ),
        max_seq_len=args.max_seq_len,
        use_moe=bool(args.use_moe),
        num_cameras=args.num_cameras,
        num_history_frames=args.num_history_frames,
        freeze_vision_encoder=args.freeze_vision,
        freeze_first_layers=args.freeze_first_layers,
        loss_control_weight=args.control_loss_weight,
        loss_action_weight=args.action_loss_weight,
    )

    ckp_data = lm_checkpoint(
        driving_config, weight=args.save_weight, save_dir=args.save_dir
    ) if args.from_resume == 1 else None

    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. 配置 wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = (
            f"MiniMind-Driving-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}"
        )
        wandb.init(
            project=args.wandb_project,
            name=wandb_run_name,
            id=wandb_id,
            resume=resume,
        )

    # ========== 5. 初始化模型 ==========
    vision_encoder_path = os.path.join(
        os.path.dirname(__file__), "..", "model", "vision_model", "clip-vit-base-patch16"
    )
    model = MiniMindDriving(driving_config, vision_encoder_path=vision_encoder_path)
    if model.vision_encoder is None and not args.allow_random_vision:
        raise FileNotFoundError(
            f"CLIP checkpoint not found at {vision_encoder_path}; "
            "download it or pass --allow-random-vision for smoke tests"
        )

    # 加载预训练权重
    if args.from_weight != 'none':
        weight_path = os.path.join(args.save_dir, f"{args.from_weight}_{args.hidden_size}.pth")
        if os.path.exists(weight_path):
            weights = torch.load(weight_path, map_location=args.device)
            if isinstance(weights, dict) and "model" in weights:
                model.load_state_dict(weights["model"], strict=False)
            else:
                model.load_state_dict(weights, strict=False)
            Logger(f"Loaded weights from {weight_path}")
        else:
            raise FileNotFoundError(f"Requested checkpoint not found: {weight_path}")

    # 冻结策略
    if args.freeze_vision:
        for param in model.camera_encoder.parameters():
            param.requires_grad = False
        if model.vision_encoder:
            for param in model.vision_encoder.parameters():
                param.requires_grad = False

    if args.freeze_first_layers > 0:
        for layer in model.model.layers[:args.freeze_first_layers]:
            for param in layer.parameters():
                param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    Logger(f"Model: total={total / 1e6:.2f}M, trainable={trainable / 1e6:.2f}M "
           f"({trainable / total:.1%})")

    model = model.to(args.device)

    # 加载 tokenizer
    tokenizer_path = os.path.join(os.path.dirname(__file__), "..", "model")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # ========== 6. 初始化数据集 ==========
    train_ds = DrivingSFTDataset(
        data_path=args.data_path,
        config=driving_config,
        image_root=args.images_path,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
    )
    collator = DrivingDataCollator(
        pad_token_id=tokenizer.pad_token_id or 0
    )
    val_loader = None
    if args.val_data_path:
        val_ds = DrivingSFTDataset(
            data_path=args.val_data_path,
            config=driving_config,
            image_root=args.images_path,
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collator,
        )

    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
    )

    # ========== 7. 从 ckp 恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data.get('scaler'))
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
        Logger(f"Resumed from epoch={start_epoch}, step={start_step}")

    # ========== 8. DDP 包装 ==========
    if dist.is_initialized():
        model._ddc_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 9. 数据增强 ==========
    augmentation = None
    if args.use_augmentation:
        augmentation = DrivingDataAugmentation()

    # ========== 10. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)

        if epoch == start_epoch and start_step > 0:
            batch_sampler = SkipBatchSampler(
                train_sampler or range(len(train_ds)),
                args.batch_size,
                start_step + 1,
            )
            loader = DataLoader(
                train_ds,
                batch_sampler=batch_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
                collate_fn=collator,
            )
            Logger(f'Epoch [{epoch+1}/{args.epochs}]: 从 step {start_step+1} 继续')
            train_epoch(
                epoch, loader, len(loader) + start_step + 1,
                start_step, wandb, args.use_augmentation, augmentation,
            )
        else:
            loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
                collate_fn=collator,
            )
            train_epoch(
                epoch, loader, len(loader), 0,
                wandb, args.use_augmentation, augmentation,
            )
        if val_loader is not None:
            validate_epoch(val_loader)

    Logger("Training completed!")
