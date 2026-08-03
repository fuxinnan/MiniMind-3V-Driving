"""Shape-safe augmentation for synchronized driving camera sequences."""

import random
from typing import Optional

import torch
import torch.nn.functional as F


class DrivingDataAugmentation:
    """Apply photometric augmentation consistently to ``[B,Cam,T,C,H,W]``."""

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        noise_std: float = 0.01,
        blur_probability: float = 0.1,
        horizontal_flip_probability: float = 0.0,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.noise_std = noise_std
        self.blur_probability = blur_probability
        # Horizontal flips swap camera semantics and steering direction. Keep
        # disabled unless the caller also remaps cameras and labels.
        self.horizontal_flip_probability = horizontal_flip_probability

    def augment_images_batch(
        self,
        images: torch.Tensor,
        control_labels: Optional[torch.Tensor] = None,
    ):
        if images.ndim not in (5, 6):
            raise ValueError("images must be [Cam,T,C,H,W] or [B,Cam,T,C,H,W]")
        had_batch = images.ndim == 6
        values = images if had_batch else images.unsqueeze(0)
        output = values.clone()
        for batch_index in range(output.shape[0]):
            brightness = 1.0 + random.uniform(-self.brightness, self.brightness)
            contrast = 1.0 + random.uniform(-self.contrast, self.contrast)
            sample = output[batch_index]
            mean = sample.mean(dim=(-3, -2, -1), keepdim=True)
            sample = (sample - mean) * contrast + mean
            sample = sample * brightness
            if self.noise_std > 0:
                sample = sample + torch.randn_like(sample) * self.noise_std
            if random.random() < self.blur_probability:
                flat = sample.reshape(-1, *sample.shape[-3:])
                flat = F.avg_pool2d(flat, kernel_size=3, stride=1, padding=1)
                sample = flat.reshape_as(sample)
            output[batch_index] = sample
        if self.horizontal_flip_probability > 0:
            raise ValueError(
                "Driving flips require camera and steering remapping; "
                "use horizontal_flip_probability=0"
            )
        result = output if had_batch else output.squeeze(0)
        return (result, control_labels) if control_labels is not None else result

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return self.augment_images_batch(images)
