"""
MiniMind-Driving DPO 训练脚本

对同一多相机上下文上的 chosen / rejected 文本响应对齐偏好：
    L_dpo = -log σ(β · ((log πθ(yw|x) - log πθ(yl|x))
                        - (log πref(yw|x) - log πref(yl|x))))
可选叠加 chosen 控制/动作监督，稳定驾驶头。
"""

import argparse
import copy
import os
import sys
import time
import warnings
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.driving_dataset import DrivingDPOCollator, DrivingDPODataset
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


def _sensor_kwargs(batch, device):
    keys = (
        "lidar_pointcloud", "radar_data", "gps_imu",
        "lidar_mask", "radar_mask", "gps_imu_mask",
    )
    return {
        key: batch[key].to(device) for key in keys if key in batch
    }


def sequence_logprobs(logits, labels, attention_mask):
    """Mean token log-prob over non-pad positions (teacher forcing)."""
    log_probs = F.log_softmax(logits[:, :-1], dim=-1)
    target = labels[:, 1:]
    mask = attention_mask[:, 1:].float()
    token_logp = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    token_logp = token_logp * mask
    denom = mask.sum(dim=-1).clamp_min(1.0)
    return token_logp.sum(dim=-1) / denom


def forward_branch(model, batch, side, device):
    input_ids = batch[f"{side}_input_ids"].to(device)
    attention_mask = batch[f"{side}_attention_mask"].to(device)
    pixel_values = batch["pixel_values"].to(device)
    sensors = _sensor_kwargs(batch, device)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        labels=input_ids.masked_fill(attention_mask == 0, -100),
        control_labels=(
            batch[f"{side}_control_labels"].to(device)
            if batch.get(f"{side}_control_labels") is not None else None
        ),
        action_labels=(
            batch[f"{side}_action_labels"].to(device)
            if batch.get(f"{side}_action_labels") is not None else None
        ),
        control_label_mask=(
            batch[f"{side}_control_mask"].to(device)
            if f"{side}_control_mask" in batch else None
        ),
        action_label_mask=(
            batch[f"{side}_action_mask"].to(device)
            if f"{side}_action_mask" in batch else None
        ),
        **sensors,
    )
    logp = sequence_logprobs(outputs.logits, input_ids, attention_mask)
    return outputs, logp


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    for step, batch in enumerate(loader, start=start_step + 1):
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with autocast_ctx:
            policy_chosen, logp_chosen = forward_branch(
                model, batch, "chosen", args.device
            )
            policy_rejected, logp_rejected = forward_branch(
                model, batch, "rejected", args.device
            )
            with torch.no_grad():
                _, ref_chosen = forward_branch(ref_model, batch, "chosen", args.device)
                _, ref_rejected = forward_branch(
                    ref_model, batch, "rejected", args.device
                )

            logits = args.beta * (
                (logp_chosen - logp_rejected) - (ref_chosen - ref_rejected)
            )
            dpo_loss = -F.logsigmoid(logits).mean()

            aux = torch.zeros((), device=args.device)
            if policy_chosen.losses.get("control") is not None:
                aux = aux + args.control_loss_weight * policy_chosen.losses["control"]
            if policy_chosen.losses.get("action") is not None:
                aux = aux + args.action_loss_weight * policy_chosen.losses["action"]
            total_loss = (dpo_loss + aux) / args.accumulation_steps

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
                f"loss:{current:.6f} dpo:{dpo_loss.item():.4f} "
                f"aux:{aux.item():.4f} "
                f"lr:{optimizer.param_groups[-1]['lr']:.12f} time:{eta_min}min"
            )
            if wandb:
                wandb.log({
                    "loss": current,
                    "dpo_loss": dpo_loss.item(),
                    "aux_loss": aux.item(),
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
    parser = argparse.ArgumentParser(description="MiniMind-Driving DPO")
    parser.add_argument("--save_dir", type=str, default="./out/checkpoints")
    parser.add_argument("--save_weight", default="driving_dpo", type=str)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
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
        default="../dataset/driving/processed/dpo_train.jsonl",
    )
    parser.add_argument(
        "--images_path", type=str, default="../dataset/driving/processed"
    )
    parser.add_argument("--from_weight", default="driving_sft", type=str)
    parser.add_argument("--from_resume", default=0, type=int, choices=[0, 1])
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature")
    parser.add_argument("--control_loss_weight", type=float, default=0.1)
    parser.add_argument("--action_loss_weight", type=float, default=0.1)
    parser.add_argument("--freeze_vision", action="store_true")
    parser.add_argument(
        "--allow-random-vision",
        action="store_true",
        help="smoke test: allow missing CLIP",
    )
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Driving-DPO")
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
        wandb.init(project=args.wandb_project, name="MiniMind-Driving-DPO")

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
        Logger(f"Loaded policy weights from {weight_path}")
    else:
        Logger("Training DPO from randomly initialized policy (smoke / from scratch)")

    if args.freeze_vision:
        for module in (model.camera_encoder, model.vision_encoder):
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad = False

    ref_model = copy.deepcopy(model).to(args.device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    model = model.to(args.device)
    model_to_save = model
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(os.path.dirname(__file__), "..", "model")
    )

    train_ds = DrivingDPODataset(
        data_path=args.data_path,
        config=driving_config,
        image_root=args.images_path,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
    )
    collator = DrivingDPOCollator(pad_token_id=tokenizer.pad_token_id or 0)
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

    Logger("DPO training completed!")
