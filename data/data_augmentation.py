"""
驾驶数据增强

提供针对自动驾驶场景的数据增强方法:
    - 图像增强: 亮度、对比度、噪声、模糊、翻转
    - 传感器增强: 噪声注入、缺失模拟
    - 控制标签增强: 小扰动
    - 场景级增强: 天气、时间、季节变化模拟
"""

import os
import numpy as np
import torch
from typing import Dict, Optional, Tuple, List
from PIL import Image, ImageEnhance, ImageFilter


class DrivingDataAugmentation:
    """
    驾驶数据增强器

    支持多种增强策略，可组合使用
    """

    def __init__(
        self,
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        saturation_range: Tuple[float, float] = (0.8, 1.2),
        hue_range: Tuple[float, float] = (-0.1, 0.1),
        blur_prob: float = 0.3,
        noise_std: float = 0.01,
        flip_prob: float = 0.5,
        weather_augment: bool = True,
        control_noise_std: float = 0.02,
    ):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range
        self.blur_prob = blur_prob
        self.noise_std = noise_std
        self.flip_prob = flip_prob
        self.weather_augment = weather_augment
        self.control_noise_std = control_noise_std

    def augment_image(
        self,
        image: Image.Image,
        augment: bool = True,
    ) -> Image.Image:
        """
        单张图像增强

        Args:
            image: PIL Image
            augment: 是否执行增强

        Returns:
            增强后的图像
        """
        if not augment:
            return image

        # 亮度增强
        if self.brightness_range[0] < 1.0 or self.brightness_range[1] > 1.0:
            factor = np.random.uniform(*self.brightness_range)
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(factor)

        # 对比度增强
        if self.contrast_range[0] < 1.0 or self.contrast_range[1] > 1.0:
            factor = np.random.uniform(*self.contrast_range)
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(factor)

        # 饱和度增强
        if self.saturation_range[0] < 1.0 or self.saturation_range[1] > 1.0:
            factor = np.random.uniform(*self.saturation_range)
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(factor)

        # 模糊增强
        if np.random.random() < self.blur_prob:
            kernel_size = np.random.choice([3, 5, 7])
            image = image.filter(ImageFilter.GaussianBlur(kernel_size))

        # 噪声增强
        if self.noise_std > 0:
            image = self._add_noise(image)

        # 水平翻转
        if np.random.random() < self.flip_prob:
            image = Image.Image.flip(image, horizontal=True)

        return image

    def augment_images_batch(
        self,
        pixel_values: torch.Tensor,
        augment: bool = True,
    ) -> torch.Tensor:
        """
        批量图像增强

        Args:
            pixel_values: [num_cameras, num_frames, C, H, W]
            augment: 是否执行增强

        Returns:
            增强后的图像 tensor
        """
        if not augment or pixel_values is None:
            return pixel_values

        from torchvision import transforms

        # 随机亮度/对比度/饱和度/色调
        pixel_values = self._random_color_jitter(pixel_values)

        # 随机高斯噪声
        if self.noise_std > 0:
            noise = torch.randn_like(pixel_values) * self.noise_std
            pixel_values = torch.clamp(pixel_values + noise, 0.0, 1.0)

        # 随机模糊
        if np.random.random() < self.blur_prob:
            kernel_size = np.random.choice([3, 5, 7])
            pixel_values = self._gaussian_blur(pixel_values, kernel_size)

        return pixel_values

    def augment_controls(
        self,
        controls: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        控制标签增强 (小扰动)

        Args:
            controls: [4] 连续控制信号
            action: 离散动作标签

        Returns:
            增强后的控制信号和动作标签
        """
        if controls is None:
            return controls, action

        noise = torch.randn_like(controls) * self.control_noise_std
        augmented_controls = torch.clamp(controls + noise, -1.0, 1.0)

        return augmented_controls, action

    def augment_weather(
        self,
        pixel_values: torch.Tensor,
        weather: str = "rain",
    ) -> torch.Tensor:
        """
        天气效果增强

        Args:
            pixel_values: [num_cameras, num_frames, C, H, W]
            weather: "rain" / "fog" / "snow" / "night"

        Returns:
            天气效果增强后的图像
        """
        if pixel_values is None:
            return pixel_values

        if weather == "rain":
            return self._add_rain(pixel_values)
        elif weather == "fog":
            return self._add_fog(pixel_values)
        elif weather == "snow":
            return self._add_snow(pixel_values)
        elif weather == "night":
            return self._add_night(pixel_values)
        else:
            return pixel_values

    def _add_noise(self, image: Image.Image) -> Image.Image:
        """添加高斯噪声"""
        img_array = np.array(image, dtype=np.float32) / 255.0
        noise = np.random.randn(*img_array.shape) * self.noise_std
        img_array = np.clip(img_array + noise, 0.0, 1.0)
        return Image.fromarray((img_array * 255).astype(np.uint8))

    def _random_color_jitter(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """随机颜色抖动"""
        brightness = np.random.uniform(*self.brightness_range)
        contrast = np.random.uniform(*self.contrast_range)
        saturation = np.random.uniform(*self.saturation_range)

        # 对每个通道独立处理
        for c in range(pixel_values.shape[2]):
            channel = pixel_values[:, :, c, :, :]
            channel = channel * brightness * contrast * saturation
            pixel_values[:, :, c, :, :] = torch.clamp(channel, 0.0, 1.0)

        return pixel_values

    def _gaussian_blur(self, pixel_values: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
        """高斯模糊"""
        from torchvision import transforms
        blur = transforms.GaussianBlur(kernel_size=kernel_size)
        # 对每个相机和帧分别处理
        B, C, H, W = pixel_values.shape[0], pixel_values.shape[2], pixel_values.shape[3], pixel_values.shape[4]
        flat = pixel_values.view(-1, C, H, W)
        flat = blur(flat)
        return flat.view(*pixel_values.shape)

    def _add_rain(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """添加雨效果"""
        B, NC, NF, C, H, W = pixel_values.shape
        # 添加深色条纹模拟雨滴
        for b in range(B):
            for n in range(NC):
                for f in range(NF):
                    img = pixel_values[b, n, f].permute(1, 2, 0).numpy()
                    # 添加随机暗线
                    num_strokes = np.random.randint(5, 20)
                    for _ in range(num_strokes):
                        x1 = np.random.randint(0, W)
                        y1 = np.random.randint(0, H // 2)
                        x2 = x1 + np.random.randint(-20, 20)
                        y2 = y1 + np.random.randint(H // 4, H // 2)
                        # 简化处理: 添加随机暗色像素
                        mask = np.random.random((H, W)) < 0.02
                        img[mask] *= 0.5
                    pixel_values[b, n, f] = torch.clamp(
                        torch.tensor(img).permute(2, 0, 1), 0.0, 1.0
                    )
        return pixel_values

    def _add_fog(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """添加雾效果"""
        fog_density = np.random.uniform(0.1, 0.5)
        # 添加灰白色遮罩
        fog_color = torch.tensor([0.7, 0.7, 0.7])
        for b in range(pixel_values.shape[0]):
            for n in range(pixel_values.shape[1]):
                for f in range(pixel_values.shape[2]):
                    img = pixel_values[b, n, f]
                    fog = fog_color.view(3, 1, 1).expand_as(img)
                    pixel_values[b, n, f] = img * (1 - fog_density) + fog * fog_density
        return pixel_values

    def _add_snow(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """添加雪效果"""
        B, NC, NF, C, H, W = pixel_values.shape
        for b in range(B):
            for n in range(NC):
                for f in range(NF):
                    img = pixel_values[b, n, f].permute(1, 2, 0).numpy()
                    # 添加白色噪点
                    num_flakes = np.random.randint(50, 200)
                    for _ in range(num_flakes):
                        x = np.random.randint(0, W)
                        y = np.random.randint(0, H)
                        img[y, x] = [1.0, 1.0, 1.0]
                    pixel_values[b, n, f] = torch.clamp(
                        torch.tensor(img).permute(2, 0, 1), 0.0, 1.0
                    )
        return pixel_values

    def _add_night(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """添加夜间效果"""
        night_factor = np.random.uniform(0.2, 0.5)
        pixel_values = pixel_values * night_factor
        # 模拟路灯和车灯光晕
        return pixel_values

    def get_augmentation_config(self) -> Dict:
        """获取增强配置"""
        return {
            "brightness_range": self.brightness_range,
            "contrast_range": self.contrast_range,
            "saturation_range": self.saturation_range,
            "hue_range": self.hue_range,
            "blur_prob": self.blur_prob,
            "noise_std": self.noise_std,
            "flip_prob": self.flip_prob,
            "weather_augment": self.weather_augment,
            "control_noise_std": self.control_noise_std,
        }
