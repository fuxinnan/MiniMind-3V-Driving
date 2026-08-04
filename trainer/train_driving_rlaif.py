"""
MiniMind-Driving RLAIF 训练脚本

在多任务 SFT 损失上按样本 reward 加权：
    L = mean(reward · (L_text + λ_ctrl L_ctrl + λ_action L_action))
reward 来自 JSONL 的 safety_score / control_quality（或显式 reward）。
"""

import argparse
import os
import sys
import time
import warnings
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.driving_dataset import DrivingRLAIFCollator, DrivingRLAIFDataset
from model.driving.model_driving import DrivingConfig, MiniMindDriving
from trainer.train_utils import (
    Logger,
    SkipBatchSampler,
    get_lr,
    init_distributed_mode,
    is_main_process,
    lm_checkpoint,
    setup_seed,
)

warnings.filterwarnings("ignore")


def _move_batch(batch, device):
    tensors = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            tensors[key] = value.to(device)
    return tensors


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    for step, batch in enumerate(loader, start=start_step + 1):
        batch = _move_batch(batch, args.device)
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        reward = batch["reward"]

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with autocast_ctx:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=batch["pixel_values"],
                control_labels=batch.get("control_labels"),
                action_labels=batch.get("action_labels"),
                control_label_mask=batch.get("control_label_mask"),
                action_label_mask=batch.get("action_label_mask"),
                lidar_pointcloud=batch.get("lidar_pointcloud"),
                radar_data=batch.get("radar_data"),
                gps_imu=batch.get("gps_imu"),
                lidar_mask=batch.get("lidar_mask"),
                radar_mask=batch.get("radar_mask"),
                gps_imu_mask=batch.get("gps_imu_mask"),
                labels=input_ids.masked_fill(attention_mask == 0, -100),
            )
            if outputs.loss is None:
                raise RuntimeError("model returned no multitask loss")
            # Recompute a per-batch scalar then scale by mean reward.
            # Full per-token weighting would need model API changes; MVP uses
            # sample-level reward as a global multiplier on the multitask loss.
            weight = reward.mean().clamp_min(args.min_reward)
            total_loss = (weight * outputs.loss) / args.accumulation_steps
            zero = torch.zeros((), device=args.device)
            loss_text = outputs.losses.get("text") or zero
            loss_ctrl = outputs.losses.get("control") or zero
            loss_action = outputs.losses.get("action") or zero

        scaler.scale(total_loss).backward()
        if step % args.accumulation_steps == 0 or step == iters:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current = total_loss.item() * args.accumulation_steps
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(
                f"Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}) "
                f"loss:{current:.6f} text:{loss_text.item():.4f} "
                f"ctrl:{loss_ctrl.item():.4f} action:{loss_action.item():.4f} "
                f"reward:{reward.mean().item():.3f} "
                f"lr:{optimizer.param_groups[-1]['lr']:.12f} time:{eta_min}min"
            )
            if wandb:
                wandb.log({
                    "loss": current,
                    "loss_text": loss_text.item(),
                    "loss_ctrl": loss_ctrl.item(),
                    "loss_action": loss_action.item(),
                    "reward": reward.mean().item(),
                    "lr": optimizer.param_groups[-1]["lr"],
                })

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = "_moe" if model_to_save.config.use_moe else ""
            ckp = (
                f"{args.save_dir}/{args.save_weight}_"
                f"{model_to_save.config.hidden_size}{moe_suffix}.pth"
            )
            state_dict = model_to_save.state_dict()
            clean = {
                key: value.half().cpu()
                for key, value in state_dict.items()
                if not key.startswith("vision_encoder.")
            }
            torch.save(clean, ckp)
            lm_checkpoint(
                model_to_save.config,
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

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind-Driving RLAIF")
    parser.add_argument("--save_dir", type=str, default="./out/checkpoints")
    parser.add_argument("--save_weight", default="driving_rlaif", type=str)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=200)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--num_cameras", default=4, type=int)
    parser.add_argument("--num_history_frames", default=3, type=int)
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1])
    parser.add_argument(
        "--data_path",
        type=str,
        default="../dataset/driving/processed/rlaif_train.jsonl",
    )
    parser.add_argument(
        "--images_path", type=str, default="../dataset/driving/processed"
    )
    parser.add_argument("--from_weight", default="driving_sft", type=str)
    parser.add_argument("--from_resume", default=0, type=int, choices=[0, 1])
    parser.add_argument("--control_loss_weight", type=float, default=0.3)
    parser.add_argument("--action_loss_weight", type=float, default=0.2)
    parser.add_argument("--safety_weight", type=float, default=0.6)
    parser.add_argument("--control_quality_weight", type=float, default=0.4)
    parser.add_argument("--min_reward", type=float, default=0.05)
    parser.add_argument("--freeze_vision", action="store_true")
    parser.add_argument("--allow-random-vision", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Driving-RLAIF")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))
    os.makedirs(args.save_dir, exist_ok=True)

    driving_config = DrivingConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_position_embeddings=(
            args.max_seq_len + args.num_cameras * 196 + 3
        ),
        max_seq_len=args.max_seq_len,
        use_moe=bool(args.use_moe),
        num_cameras=args.num_cameras,
        num_history_frames=args.num_history_frames,
        freeze_vision_encoder=args.freeze_vision,
        loss_control_weight=args.control_loss_weight,
        loss_action_weight=args.action_loss_weight,
    )
    ckp_data = (
        lm_checkpoint(driving_config, weight=args.save_weight, save_dir=args.save_dir)
        if args.from_resume == 1 else None
    )

    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = (
        nullcontext() if device_type == "cpu"
        else torch.cuda.amp.autocast(dtype=dtype)
    )

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb.init(project=args.wandb_project, name="MiniMind-Driving-RLAIF")

    vision_encoder_path = os.path.join(
        os.path.dirname(__file__), "..", "model", "vision_model",
        "clip-vit-base-patch16",
    )
    model = MiniMindDriving(driving_config, vision_encoder_path=vision_encoder_path)
    if model.vision_encoder is None and not args.allow_random_vision:
        raise FileNotFoundError(
            f"CLIP checkpoint not found at {vision_encoder_path}; "
            "download it or pass --allow-random-vision for smoke tests"
        )

    if args.from_weight != "none":
        weight_path = os.path.join(
            args.save_dir, f"{args.from_weight}_{args.hidden_size}.pth"
        )
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Requested checkpoint not found: {weight_path}")
        weights = torch.load(weight_path, map_location=args.device)
        state = (
            weights["model"]
            if isinstance(weights, dict) and "model" in weights
            else weights
        )
        model.load_state_dict(state, strict=False)
        Logger(f"Loaded weights from {weight_path}")
    else:
        Logger("Training RLAIF from randomly initialized weights (smoke / from scratch)")

    if args.freeze_vision:
        for module in (model.camera_encoder, model.vision_encoder):
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad = False

    model = model.to(args.device)
    model_to_save = model
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(os.path.dirname(__file__), "..", "model")
    )

    train_ds = DrivingRLAIFDataset(
        data_path=args.data_path,
        config=driving_config,
        image_root=args.images_path,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        safety_weight=args.safety_weight,
        control_weight=args.control_quality_weight,
    )
    collator = DrivingRLAIFCollator(pad_token_id=tokenizer.pad_token_id or 0)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
    )

    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"], strict=False)
        optimizer.load_state_dict(ckp_data["optimizer"])
        if ckp_data.get("scaler"):
            scaler.load_state_dict(ckp_data["scaler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)

    if dist.is_initialized():
        model._ddc_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
        model_to_save = model.module

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
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
            train_epoch(epoch, loader, len(loader) + start_step + 1, start_step, wandb)
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
            train_epoch(epoch, loader, len(loader), 0, wandb)

    Logger("RLAIF training completed!")
