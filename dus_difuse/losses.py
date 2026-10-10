"""Paper-aligned objectives used to train GFCM and DRFM."""

from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F


class SobelGradient(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "kernel_x",
            torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
            )[None, None],
        )
        self.register_buffer(
            "kernel_y",
            torch.tensor(
                [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32
            )[None, None],
        )

    def forward(self, image: torch.Tensor):
        channels = image.shape[1]
        kernel_x = self.kernel_x.to(dtype=image.dtype).repeat(channels, 1, 1, 1)
        kernel_y = self.kernel_y.to(dtype=image.dtype).repeat(channels, 1, 1, 1)
        grad_x = F.conv2d(image, kernel_x, padding=1, groups=channels)
        grad_y = F.conv2d(image, kernel_y, padding=1, groups=channels)
        return grad_x, grad_y


def _ssim_loss(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    kernel_size = min(11, first.shape[-2], first.shape[-1])
    if kernel_size % 2 == 0:
        kernel_size -= 1
    coordinates = torch.arange(
        kernel_size, device=first.device, dtype=first.dtype
    ) - kernel_size // 2
    kernel_1d = torch.exp(-coordinates.square() / (2 * 1.5**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    channels = first.shape[1]
    kernel = kernel_2d.view(1, 1, kernel_size, kernel_size).repeat(
        channels, 1, 1, 1
    )

    def filter_image(image: torch.Tensor) -> torch.Tensor:
        return F.conv2d(image, kernel, groups=channels)

    mu_first = filter_image(first)
    mu_second = filter_image(second)
    sigma_first = filter_image(first.square()) - mu_first.square()
    sigma_second = filter_image(second.square()) - mu_second.square()
    covariance = filter_image(first * second) - mu_first * mu_second
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu_first * mu_second + c1) * (2 * covariance + c2)) / (
        (mu_first.square() + mu_second.square() + c1)
        * (sigma_first + sigma_second + c2)
    )
    return 1.0 - score.mean()


class DiffusionFusionConsistencyLoss(nn.Module):
    """GFCM objective corresponding to Eqs. (14)--(17) in the paper."""

    def __init__(
        self,
        lambda_gradient: float = 10.0,
        lambda_pixel: float = 20.0,
        lambda_ssim: float = 1.0,
    ) -> None:
        super().__init__()
        self.lambda_gradient = lambda_gradient
        self.lambda_pixel = lambda_pixel
        self.lambda_ssim = lambda_ssim
        self.gradient = SobelGradient()

    def forward(
        self,
        fused: torch.Tensor,
        enhanced_visible: torch.Tensor,
        enhanced_infrared: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        pixel_target = torch.maximum(enhanced_visible, enhanced_infrared)
        visible_x, visible_y = self.gradient(enhanced_visible)
        infrared_x, infrared_y = self.gradient(enhanced_infrared)
        fused_x, fused_y = self.gradient(fused)
        target_x = torch.where(visible_x.abs() >= infrared_x.abs(), visible_x, infrared_x)
        target_y = torch.where(visible_y.abs() >= infrared_y.abs(), visible_y, infrared_y)
        gradient = F.mse_loss(fused_x, target_x) + F.mse_loss(fused_y, target_y)
        pixel = F.mse_loss(fused, pixel_target)
        structural = _ssim_loss(fused, enhanced_visible) + _ssim_loss(
            fused, enhanced_infrared
        )
        total = (
            self.lambda_gradient * gradient
            + self.lambda_pixel * pixel
            + self.lambda_ssim * structural
        )
        return {
            "total": total,
            "gradient": gradient,
            "pixel": pixel,
            "ssim": structural,
        }


def _rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
    """Convert an RGB tensor in [0, 1] to full-range YCbCr."""
    red, green, blue = image.unbind(dim=1)
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    blue_chroma = -0.168736 * red - 0.331264 * green + 0.5 * blue + 0.5
    red_chroma = 0.5 * red - 0.418688 * green - 0.081312 * blue + 0.5
    return torch.stack((luminance, blue_chroma, red_chroma), dim=1)


class GFCMPseudoSupervisionLoss(nn.Module):
    """GFCM objective used by the test_clone-main reference training code."""

    def __init__(
        self,
        lambda_luminance: float = 20.0,
        lambda_chroma: float = 50.0,
        lambda_gradient: float = 10.0,
        lambda_ssim: float = 1.0,
    ) -> None:
        super().__init__()
        self.lambda_luminance = lambda_luminance
        self.lambda_chroma = lambda_chroma
        self.lambda_gradient = lambda_gradient
        self.lambda_ssim = lambda_ssim
        self.gradient = SobelGradient()

    def forward(
        self,
        fused: torch.Tensor,
        first_reference: torch.Tensor,
        second_reference: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        fused_ycbcr = _rgb_to_ycbcr(fused)
        first_ycbcr = _rgb_to_ycbcr(first_reference)
        second_ycbcr = _rgb_to_ycbcr(second_reference)

        luminance_target = torch.maximum(
            first_ycbcr[:, :1], second_ycbcr[:, :1]
        )
        chroma_target = first_ycbcr[:, 1:]

        first_x, first_y = self.gradient(first_ycbcr[:, :1])
        second_x, second_y = self.gradient(second_ycbcr[:, :1])
        fused_x, fused_y = self.gradient(fused_ycbcr[:, :1])
        first_x, first_y = first_x.abs(), first_y.abs()
        second_x, second_y = second_x.abs(), second_y.abs()
        fused_x, fused_y = fused_x.abs(), fused_y.abs()
        target_x = torch.maximum(first_x, second_x)
        target_y = torch.maximum(first_y, second_y)

        luminance = F.mse_loss(fused_ycbcr[:, :1], luminance_target)
        chroma = F.mse_loss(fused_ycbcr[:, 1:], chroma_target)
        gradient = F.mse_loss(fused_x, target_x) + F.mse_loss(fused_y, target_y)
        structural = _ssim_loss(
            fused_ycbcr[:, :1], first_ycbcr[:, :1]
        ) + _ssim_loss(fused_ycbcr[:, :1], second_ycbcr[:, :1])
        total = (
            self.lambda_luminance * luminance
            + self.lambda_chroma * chroma
            + self.lambda_gradient * gradient
            + self.lambda_ssim * structural
        )
        return {
            "total": total,
            "luminance": luminance,
            "chroma": chroma,
            "gradient": gradient,
            "ssim": structural,
        }


class DetailRestorationFidelityLoss(nn.Module):
    """YCbCr fidelity objective used by the reference DRFM training code."""

    def __init__(
        self,
        lambda_luminance: float = 20.0,
        lambda_chroma: float = 50.0,
        lambda_gradient: float = 10.0,
    ) -> None:
        super().__init__()
        self.lambda_luminance = lambda_luminance
        self.lambda_chroma = lambda_chroma
        self.lambda_gradient = lambda_gradient
        self.gradient = SobelGradient()

    def forward(
        self,
        reconstruction: torch.Tensor,
        enhanced_visible: torch.Tensor,
        enhanced_infrared: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        reconstruction_ycbcr = _rgb_to_ycbcr(reconstruction)
        visible_ycbcr = _rgb_to_ycbcr(enhanced_visible)
        infrared_ycbcr = _rgb_to_ycbcr(enhanced_infrared)

        luminance_target = torch.maximum(
            visible_ycbcr[:, :1], infrared_ycbcr[:, :1]
        )
        chroma_target = visible_ycbcr[:, 1:]

        visible_x, visible_y = self.gradient(visible_ycbcr[:, :1])
        infrared_x, infrared_y = self.gradient(infrared_ycbcr[:, :1])
        reconstruction_x, reconstruction_y = self.gradient(
            reconstruction_ycbcr[:, :1]
        )

        # Sobel_xy in the primary reference returns absolute gradients before
        # selecting the stronger response from the two modalities.
        visible_x, visible_y = visible_x.abs(), visible_y.abs()
        infrared_x, infrared_y = infrared_x.abs(), infrared_y.abs()
        reconstruction_x = reconstruction_x.abs()
        reconstruction_y = reconstruction_y.abs()
        target_x = torch.maximum(visible_x, infrared_x)
        target_y = torch.maximum(visible_y, infrared_y)

        luminance = F.mse_loss(reconstruction_ycbcr[:, :1], luminance_target)
        chroma = F.mse_loss(reconstruction_ycbcr[:, 1:], chroma_target)
        gradient = F.mse_loss(reconstruction_x, target_x) + F.mse_loss(
            reconstruction_y, target_y
        )
        total = (
            self.lambda_luminance * luminance
            + self.lambda_chroma * chroma
            + self.lambda_gradient * gradient
        )
        return {
            "total": total,
            "luminance": luminance,
            "chroma": chroma,
            "gradient": gradient,
        }
