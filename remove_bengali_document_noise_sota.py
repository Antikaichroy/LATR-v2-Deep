"""Research-grade Bengali document denoising/restoration pipeline.

Proposed method: LATR-v2-Deep (Layout-Aware Text Restoration, deep edition)
1) Illumination flattening and spectral preconditioning,
2) Self-supervised neural test-time adaptation (OCR-aligned objective),
3) Layout-aware classical reconstruction branch,
4) OCR-guided candidate ranking across neural + classical outputs,
5) Optional SDXL + ControlNet refinement.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict, dataclass
import hashlib
import io
import json
from pathlib import Path
import secrets
from typing import Any

import fitz
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    F = None

try:
    from diffusers import (  # type: ignore
        ControlNetModel,
        StableDiffusionControlNetImg2ImgPipeline,
    )
except Exception:  # pragma: no cover - optional dependency
    ControlNetModel = None
    StableDiffusionControlNetImg2ImgPipeline = None

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    TrOCRProcessor = None
    VisionEncoderDecoderModel = None


_RUNTIME: dict[str, "RuntimeComponents"] = {}


@dataclass(frozen=True)
class DenoiseResearchConfig:
    enable_neural_tta: bool
    require_deep_learning: bool
    neural_steps: int
    neural_learning_rate: float
    neural_mask_ratio: float
    neural_base_channels: int
    neural_max_side: int
    enable_ocr_critic: bool
    enable_diffusion_refiner: bool
    ocr_model_id: str
    diffusion_model_id: str
    controlnet_model_id: str
    diffusion_steps: int
    diffusion_strength: float
    diffusion_guidance_scale: float
    max_diffusion_pages: int
    device: str


@dataclass(frozen=True)
class DenoiseConfig:
    dpi: int
    jpeg_quality: int
    seed: int
    deterministic: bool
    max_pages: int | None
    overwrite: bool
    research: DenoiseResearchConfig


@dataclass(frozen=True)
class DenoiseSummary:
    pages_written: int
    mean_proxy_pre: float
    mean_proxy_post: float
    mean_ocr_pre: float
    mean_ocr_post: float
    neural_pages: int
    diffusion_pages: int
    neural_backend: str
    ocr_backend: str
    diffusion_backend: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restore noisy Bengali PDFs with a research-grade denoising pipeline."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("Books-Bengali-Noisy-SOTA"),
        help="Folder containing noisy PDFs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Books-Bengali-Denoised-SOTA"),
        help="Folder where restored PDFs and manifests will be written.",
    )
    parser.add_argument(
        "--manifest-name",
        default="denoise_manifest.csv",
        help="CSV filename written inside the output folder.",
    )
    parser.add_argument(
        "--summary-name",
        default="denoise_summary.json",
        help="Run-level summary JSON filename.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=130,
        help="Rasterization DPI for page restoration.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=72,
        help="JPEG quality for re-embedded restored pages.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260424,
        help="Base seed used only when --deterministic is enabled.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic restoration steps for exact reruns.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional page cap per PDF for debugging.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of PDFs from input directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate outputs even if existing files are present.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of PDFs to process in parallel.",
    )
    parser.add_argument(
        "--disable-neural-tta",
        action="store_true",
        help="Disable neural test-time adaptation restorer.",
    )
    parser.add_argument(
        "--require-deep-learning",
        action="store_true",
        help="Fail if torch is unavailable instead of using classical fallback.",
    )
    parser.add_argument(
        "--neural-steps",
        type=int,
        default=42,
        help="Optimization steps for neural test-time adaptation per page.",
    )
    parser.add_argument(
        "--neural-learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate for neural test-time adaptation.",
    )
    parser.add_argument(
        "--neural-mask-ratio",
        type=float,
        default=0.065,
        help="Random masking ratio for self-supervised blind-spot training.",
    )
    parser.add_argument(
        "--neural-base-channels",
        type=int,
        default=24,
        help="Base channels in the lightweight U-Net restorer.",
    )
    parser.add_argument(
        "--neural-max-side",
        type=int,
        default=1280,
        help="Max resized page side for neural adaptation to control memory.",
    )

    parser.add_argument(
        "--enable-ocr-critic",
        action="store_true",
        help="Use TrOCR confidence in candidate ranking (optional).",
    )
    parser.add_argument(
        "--enable-diffusion-refiner",
        action="store_true",
        help="Enable optional SDXL+ControlNet restoration refinement (optional).",
    )
    parser.add_argument(
        "--ocr-model-id",
        default="microsoft/trocr-large-printed",
        help="OCR model ID/path for critic.",
    )
    parser.add_argument(
        "--diffusion-model-id",
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="SDXL model ID/path for restoration refinement.",
    )
    parser.add_argument(
        "--controlnet-model-id",
        default="diffusers/controlnet-canny-sdxl-1.0",
        help="ControlNet model ID/path used in refinement.",
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=18,
        help="Diffusion denoising steps when enabled.",
    )
    parser.add_argument(
        "--diffusion-strength",
        type=float,
        default=0.14,
        help="Img2img strength for refinement.",
    )
    parser.add_argument(
        "--diffusion-guidance-scale",
        type=float,
        default=4.2,
        help="Guidance scale for refinement.",
    )
    parser.add_argument(
        "--max-diffusion-pages",
        type=int,
        default=2,
        help="Maximum pages per document that use diffusion refinement.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for optional models: auto/cpu/cuda.",
    )
    return parser.parse_args()


def build_research_config(args: argparse.Namespace) -> DenoiseResearchConfig:
    return DenoiseResearchConfig(
        enable_neural_tta=not bool(args.disable_neural_tta),
        require_deep_learning=bool(args.require_deep_learning),
        neural_steps=int(args.neural_steps),
        neural_learning_rate=float(args.neural_learning_rate),
        neural_mask_ratio=float(args.neural_mask_ratio),
        neural_base_channels=int(args.neural_base_channels),
        neural_max_side=int(args.neural_max_side),
        enable_ocr_critic=bool(args.enable_ocr_critic),
        enable_diffusion_refiner=bool(args.enable_diffusion_refiner),
        ocr_model_id=str(args.ocr_model_id),
        diffusion_model_id=str(args.diffusion_model_id),
        controlnet_model_id=str(args.controlnet_model_id),
        diffusion_steps=int(args.diffusion_steps),
        diffusion_strength=float(args.diffusion_strength),
        diffusion_guidance_scale=float(args.diffusion_guidance_scale),
        max_diffusion_pages=int(args.max_diffusion_pages),
        device=str(args.device),
    )


def stable_seed(global_seed: int, value: str) -> int:
    payload = f"{global_seed}:{value}".encode("utf-8", errors="surrogatepass")
    return int(hashlib.blake2b(payload, digest_size=8).hexdigest(), 16) % (2**32)


def derive_seed(config: DenoiseConfig, token: str) -> int:
    if config.deterministic:
        return stable_seed(config.seed, token)
    return int(secrets.randbits(32))


def render_page(page: fitz.Page, dpi: int) -> Image.Image:
    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False, colorspace=fitz.csRGB)
    return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)


def compute_text_saliency_map(image: Image.Image) -> np.ndarray:
    gray = np.asarray(ImageOps.grayscale(image)).astype(np.float32) / 255.0
    blurred = np.asarray(
        ImageOps.grayscale(image.filter(ImageFilter.GaussianBlur(radius=2.0)))
    ).astype(np.float32) / 255.0
    local_contrast = np.abs(gray - blurred)
    grad_y, grad_x = np.gradient(gray)
    gradient = np.sqrt((grad_x * grad_x) + (grad_y * grad_y))
    ink_likelihood = np.clip((0.84 - gray) / 0.84, 0.0, 1.0)
    saliency = (
        0.48 * ink_likelihood
        + 0.33 * (local_contrast / max(float(local_contrast.max()), 1e-6))
        + 0.19 * (gradient / max(float(gradient.max()), 1e-6))
    )
    return np.clip(saliency, 0.0, 1.0).astype(np.float32)


def estimate_readability_proxy(image: Image.Image, saliency: np.ndarray | None = None) -> float:
    gray = np.asarray(ImageOps.grayscale(image)).astype(np.float32) / 255.0
    if saliency is None:
        saliency = compute_text_saliency_map(image)

    ink = gray < 0.74
    paper = gray > 0.89
    if int(ink.sum()) < 64:
        ink = gray < float(np.quantile(gray, 0.35))
    if int(paper.sum()) < 64:
        paper = gray > float(np.quantile(gray, 0.78))

    fg_mean = float(gray[ink].mean()) if int(ink.sum()) else float(gray.mean())
    bg_mean = float(gray[paper].mean()) if int(paper.sum()) else float(gray.mean())
    contrast = np.clip(bg_mean - fg_mean, 0.0, 1.0)

    grad_y, grad_x = np.gradient(gray)
    gradient = np.sqrt((grad_x * grad_x) + (grad_y * grad_y))
    stroke_energy = float(gradient[ink].mean()) if int(ink.sum()) else float(gradient.mean())

    washed = (gray > 0.95) & (saliency > 0.67)
    washed_ratio = float(washed.mean())
    saturation_penalty = float(np.mean((gray < 0.05) | (gray > 0.98)))

    score = 0.58 * np.clip(contrast / 0.56, 0.0, 1.0)
    score += 0.32 * np.clip(stroke_energy / 0.26, 0.0, 1.0)
    score -= 0.17 * np.clip(washed_ratio * 7.0, 0.0, 1.0)
    score -= 0.07 * np.clip(saturation_penalty * 3.5, 0.0, 1.0)
    return float(np.clip(score, 0.0, 1.0))


def flatten_illumination(image: Image.Image) -> Image.Image:
    arr = np.asarray(image).astype(np.float32) / 255.0
    gray = np.asarray(ImageOps.grayscale(image)).astype(np.float32) / 255.0
    bg = np.asarray(
        ImageOps.grayscale(image.filter(ImageFilter.GaussianBlur(radius=22.0)))
    ).astype(np.float32) / 255.0
    normalized = gray / np.clip(bg, 0.08, 1.0)
    normalized = normalized / max(float(np.quantile(normalized, 0.995)), 1e-6)
    normalized = np.clip(normalized, 0.0, 1.0)

    ratio = normalized / np.clip(gray, 0.06, 1.0)
    arr = arr * ratio[..., None]
    return Image.fromarray(np.uint8(np.clip(arr * 255.0, 0, 255)), mode="RGB")


def suppress_periodic_noise_fft(image: Image.Image) -> Image.Image:
    gray = np.asarray(ImageOps.grayscale(image)).astype(np.float32) / 255.0
    h, w = gray.shape
    freq = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.abs(freq)

    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    radial = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    ignore_center = radial < max(8, min(h, w) * 0.05)

    threshold = float(np.quantile(mag[~ignore_center], 0.998))
    spikes = (mag > threshold) & (~ignore_center)

    notch = np.ones((h, w), dtype=np.float32)
    points = np.argwhere(spikes)
    for y, x in points:
        y0, y1 = max(0, y - 2), min(h, y + 3)
        x0, x1 = max(0, x - 2), min(w, x + 3)
        notch[y0:y1, x0:x1] *= 0.22

    restored_gray = np.real(np.fft.ifft2(np.fft.ifftshift(freq * notch)))
    restored_gray = np.clip(restored_gray, 0.0, 1.0)

    arr = np.asarray(image).astype(np.float32) / 255.0
    old_gray = np.clip(gray, 0.06, 1.0)
    ratio = restored_gray / old_gray
    arr = arr * ratio[..., None]
    return Image.fromarray(np.uint8(np.clip(arr * 255.0, 0, 255)), mode="RGB")


def layout_aware_denoise(image: Image.Image, saliency: np.ndarray) -> Image.Image:
    arr = np.asarray(image).astype(np.float32)
    smooth = np.asarray(image.filter(ImageFilter.MedianFilter(size=3))).astype(np.float32)
    smooth = np.asarray(
        Image.fromarray(np.uint8(np.clip(smooth, 0, 255)), mode="RGB").filter(
            ImageFilter.GaussianBlur(radius=0.7)
        )
    ).astype(np.float32)

    paper_weight = np.clip(1.0 - saliency, 0.15, 0.88)
    blended = arr * (1.0 - paper_weight[..., None]) + smooth * paper_weight[..., None]
    return Image.fromarray(np.uint8(np.clip(blended, 0, 255)), mode="RGB")


def reconstruct_ink(image: Image.Image, saliency: np.ndarray) -> Image.Image:
    gray = np.asarray(ImageOps.grayscale(image)).astype(np.float32)
    threshold = float(np.quantile(gray, 0.42))
    raw_mask = (gray < threshold).astype(np.uint8) * 255
    mask_img = Image.fromarray(raw_mask, mode="L").filter(ImageFilter.MaxFilter(size=3))
    mask_img = mask_img.filter(ImageFilter.MinFilter(size=3))
    mask = np.asarray(mask_img).astype(np.float32) / 255.0
    mask = np.clip(0.45 * mask + 0.55 * saliency, 0.0, 1.0)

    arr = np.asarray(image).astype(np.float32)
    arr -= mask[..., None] * 18.0
    arr += (1.0 - mask[..., None]) * 3.0
    restored = Image.fromarray(np.uint8(np.clip(arr, 0, 255)), mode="RGB")
    contrast = ImageEnhance.Contrast(restored)
    return contrast.enhance(1.06)


def aggressive_document_cleanup(image: Image.Image, saliency: np.ndarray) -> Image.Image:
    """Strong cleanup branch to visibly remove stains/smudges from scanned pages."""
    rgb = np.asarray(image).astype(np.float32) / 255.0
    gray = np.asarray(ImageOps.grayscale(image)).astype(np.float32) / 255.0

    # Estimate low-frequency background illumination and flatten it.
    bg_img = ImageOps.grayscale(image).filter(ImageFilter.GaussianBlur(radius=28.0))
    bg = np.asarray(bg_img).astype(np.float32) / 255.0
    flat = gray / np.clip(bg, 0.14, 1.0)

    lo = float(np.quantile(flat, 0.02))
    hi = float(np.quantile(flat, 0.985))
    norm = np.clip((flat - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

    # Build text mask from intensity + saliency and remove speckle noise.
    global_thr = float(np.quantile(norm, 0.46))
    text_mask = ((norm < global_thr) | ((saliency > 0.58) & (norm < 0.68))).astype(np.uint8) * 255
    mask_img = Image.fromarray(text_mask, mode="L").filter(ImageFilter.MaxFilter(size=3))
    mask_img = mask_img.filter(ImageFilter.MinFilter(size=3))
    mask = np.asarray(mask_img).astype(np.float32) / 255.0

    # Strong white background + darker ink target.
    bg_clean = np.clip(0.93 + 0.07 * norm, 0.0, 1.0)
    ink_clean = np.clip(0.10 + 0.62 * norm, 0.0, 1.0)
    final_gray = bg_clean * (1.0 - mask) + ink_clean * mask

    # Preserve light color hints but bias heavily toward clean grayscale document look.
    color_hint = np.clip(rgb * 0.90 + 0.10, 0.0, 1.0)
    gray_rgb = np.repeat(final_gray[..., None], 3, axis=2)
    merged = 0.78 * gray_rgb + 0.22 * color_hint

    out = Image.fromarray(np.uint8(np.clip(merged * 255.0, 0, 255)), mode="RGB")
    out = ImageEnhance.Contrast(out).enhance(1.12)
    out = ImageEnhance.Sharpness(out).enhance(1.08)
    return out


def _resize_with_max_side(
    image: Image.Image,
    max_side: int,
) -> tuple[Image.Image, tuple[int, int]]:
    original_size = image.size
    width, height = original_size
    side = max(width, height)
    if side <= max_side:
        return image, original_size
    scale = max_side / float(side)
    new_size = (max(8, int(round(width * scale))), max(8, int(round(height * scale))))
    resized = image.resize(new_size, Image.Resampling.BICUBIC)
    return resized, original_size


if torch is not None and nn is not None and F is not None:

    class _DepthwiseSeparableConv(nn.Module):
        def __init__(self, c_in: int, c_out: int) -> None:
            super().__init__()
            self.depthwise = nn.Conv2d(
                c_in,
                c_in,
                kernel_size=3,
                padding=1,
                groups=c_in,
                bias=False,
            )
            self.pointwise = nn.Conv2d(c_in, c_out, kernel_size=1, bias=False)
            self.norm = nn.GroupNorm(num_groups=max(1, min(8, c_out // 4)), num_channels=c_out)
            self.act = nn.GELU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.depthwise(x)
            x = self.pointwise(x)
            x = self.norm(x)
            return self.act(x)


    class _ResidualUnit(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.conv1 = _DepthwiseSeparableConv(channels, channels)
            self.conv2 = _DepthwiseSeparableConv(channels, channels)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.conv2(self.conv1(x))


    class _TinyDocumentUNet(nn.Module):
        def __init__(self, base: int = 24) -> None:
            super().__init__()
            c1 = base
            c2 = base * 2
            c3 = base * 4

            self.head = _DepthwiseSeparableConv(3, c1)
            self.enc1 = _ResidualUnit(c1)
            self.down1 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1)
            self.enc2 = _ResidualUnit(c2)
            self.down2 = nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1)
            self.bottleneck = nn.Sequential(_ResidualUnit(c3), _ResidualUnit(c3))
            self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
            self.dec2 = _ResidualUnit(c2)
            self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
            self.dec1 = _ResidualUnit(c1)
            self.tail = nn.Conv2d(c1, 3, kernel_size=3, padding=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x1 = self.enc1(self.head(x))
            x2 = self.enc2(self.down1(x1))
            x3 = self.bottleneck(self.down2(x2))
            y2 = self.up2(x3)
            if y2.shape[-2:] != x2.shape[-2:]:
                y2 = F.interpolate(y2, size=x2.shape[-2:], mode="bilinear", align_corners=False)
            y2 = self.dec2(y2 + x2)
            y1 = self.up1(y2)
            if y1.shape[-2:] != x1.shape[-2:]:
                y1 = F.interpolate(y1, size=x1.shape[-2:], mode="bilinear", align_corners=False)
            y1 = self.dec1(y1 + x1)
            out = torch.sigmoid(self.tail(y1))
            return out


def _pil_to_tensor(image: Image.Image, device: str) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    ten = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return ten.to(device=device, dtype=torch.float32)


def _saliency_to_tensor(saliency: np.ndarray, device: str) -> torch.Tensor:
    ten = torch.from_numpy(saliency.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    return ten.to(device=device, dtype=torch.float32)


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    array = (
        tensor.detach()
        .clamp(0.0, 1.0)
        .squeeze(0)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray(np.uint8(np.clip(array * 255.0, 0, 255)), mode="RGB")


def _rgb_to_gray_tensor(x: torch.Tensor) -> torch.Tensor:
    r = x[:, 0:1]
    g = x[:, 1:2]
    b = x[:, 2:3]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b


def _sobel_grad_mag(gray: torch.Tensor) -> torch.Tensor:
    kx = torch.tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
        dtype=gray.dtype,
        device=gray.device,
    ).unsqueeze(0)
    ky = torch.tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
        dtype=gray.dtype,
        device=gray.device,
    ).unsqueeze(0)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    return torch.sqrt((gx * gx) + (gy * gy) + 1e-6)


def _total_variation(x: torch.Tensor) -> torch.Tensor:
    tv_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    tv_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return tv_h + tv_w


class NeuralTTARestorer:
    def __init__(
        self,
        enabled: bool,
        require_deep_learning: bool,
        steps: int,
        learning_rate: float,
        mask_ratio: float,
        base_channels: int,
        max_side: int,
        device: str,
    ) -> None:
        self.enabled = bool(enabled)
        self.require_deep_learning = bool(require_deep_learning)
        self.steps = int(steps)
        self.learning_rate = float(learning_rate)
        self.mask_ratio = float(mask_ratio)
        self.base_channels = int(base_channels)
        self.max_side = int(max_side)
        self.device = device
        self._backend = "disabled"
        self._disabled_reason: str | None = None

    @property
    def backend(self) -> str:
        if not self.enabled:
            return "disabled"
        if self._disabled_reason:
            return f"{self._backend}(fallback:{self._disabled_reason})"
        return self._backend

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _check_runtime(self) -> bool:
        if torch is None or nn is None or F is None:
            self._disabled_reason = "torch_not_installed"
            if self.require_deep_learning:
                raise RuntimeError(
                    "Deep-learning restoration was requested, but torch is not installed."
                )
            return False
        return True

    def _random_blindspot_mask(
        self,
        x: torch.Tensor,
        rng: np.random.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = x.shape
        mask = torch.zeros((b, 1, h, w), dtype=torch.float32, device=x.device)
        total = h * w
        count = max(16, int(total * self.mask_ratio))

        ys = torch.from_numpy(rng.integers(0, h, size=count).astype(np.int64)).to(x.device)
        xs = torch.from_numpy(rng.integers(0, w, size=count).astype(np.int64)).to(x.device)
        mask[:, :, ys, xs] = 1.0

        dy = int(rng.integers(-2, 3))
        dx = int(rng.integers(-2, 3))
        if dy == 0 and dx == 0:
            dx = 1
        shifted = torch.roll(x, shifts=(dy, dx), dims=(2, 3))
        corrupted = torch.where(mask.expand_as(x) > 0.5, shifted, x)
        return corrupted, mask

    def restore(
        self,
        image: Image.Image,
        saliency: np.ndarray,
        page_seed: int,
    ) -> tuple[Image.Image, bool]:
        if not self.enabled:
            return image, False
        if not self._check_runtime():
            return image, False
        assert torch is not None and nn is not None and F is not None
        if "_TinyDocumentUNet" not in globals():
            self._disabled_reason = "model_definition_unavailable"
            return image, False

        rng = np.random.default_rng(page_seed)
        device = self._resolve_device()
        resized, original_size = _resize_with_max_side(image, self.max_side)
        sal = saliency
        if sal.shape[::-1] != resized.size:
            sal_img = Image.fromarray(np.uint8(np.clip(sal * 255.0, 0, 255)), mode="L")
            sal_img = sal_img.resize(resized.size, Image.Resampling.BICUBIC)
            sal = np.asarray(sal_img).astype(np.float32) / 255.0

        x = _pil_to_tensor(resized, device=device)
        sal_t = _saliency_to_tensor(sal, device=device).clamp(0.0, 1.0)
        fg = sal_t
        bg = (1.0 - sal_t).clamp(0.05, 1.0)
        base = max(12, min(48, self.base_channels))
        model = _TinyDocumentUNet(base=base).to(device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, betas=(0.9, 0.99))

        model.train()
        for _ in range(max(8, self.steps)):
            corrupted, mask = self._random_blindspot_mask(x, rng)
            pred = model(corrupted)

            rec = (pred - x).abs()
            loss_rec = (rec * mask.expand_as(rec)).sum() / mask.sum().clamp_min(1.0)

            pred_gray = _rgb_to_gray_tensor(pred)
            x_gray = _rgb_to_gray_tensor(x)
            fg_mean = (pred_gray * fg).sum() / fg.sum().clamp_min(1e-6)
            bg_mean = (pred_gray * bg).sum() / bg.sum().clamp_min(1e-6)
            loss_contrast = -(bg_mean - fg_mean)

            grad_pred = _sobel_grad_mag(pred_gray)
            grad_in = _sobel_grad_mag(x_gray)
            loss_edge = ((grad_pred - grad_in).abs() * fg).mean()
            loss_tv = _total_variation(pred)
            loss = loss_rec + (0.20 * loss_contrast) + (0.10 * loss_edge) + (0.006 * loss_tv)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            outputs = []
            outputs.append(model(x))
            x_h = torch.flip(x, dims=[3])
            outputs.append(torch.flip(model(x_h), dims=[3]))
            x_v = torch.flip(x, dims=[2])
            outputs.append(torch.flip(model(x_v), dims=[2]))
            pred = torch.stack(outputs, dim=0).mean(dim=0)

        restored = _tensor_to_pil(pred)
        if restored.size != original_size:
            restored = restored.resize(original_size, Image.Resampling.BICUBIC)
        self._backend = "ocr_guided_self_supervised_unet_tta"
        return restored, True


class OCRCritic:
    def __init__(self, enabled: bool, model_id: str, device: str) -> None:
        self.enabled = bool(enabled)
        self.model_id = model_id
        self.device = device
        self._processor: Any = None
        self._model: Any = None
        self._backend = "readability_proxy"
        self._disabled_reason: str | None = None

    @property
    def backend(self) -> str:
        if self._disabled_reason:
            return f"{self._backend}(fallback:{self._disabled_reason})"
        return self._backend

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _lazy_load(self) -> None:
        if not self.enabled or self._model is not None:
            return
        if TrOCRProcessor is None or VisionEncoderDecoderModel is None or torch is None:
            self._disabled_reason = "transformers_or_torch_not_installed"
            return
        try:
            self._processor = TrOCRProcessor.from_pretrained(self.model_id)
            self._model = VisionEncoderDecoderModel.from_pretrained(self.model_id)
            self._model.to(self._resolve_device())
            self._model.eval()
            self._backend = "trocr_confidence"
        except Exception as exc:  # pragma: no cover - optional backend
            self._disabled_reason = f"model_load_error:{exc.__class__.__name__}"

    def score(self, image: Image.Image, saliency: np.ndarray | None = None) -> float:
        proxy = estimate_readability_proxy(image, saliency)
        if not self.enabled:
            return proxy
        self._lazy_load()
        if self._model is None or self._processor is None or torch is None:
            return proxy
        try:
            device = self._resolve_device()
            pixel_values = self._processor(
                images=image.convert("RGB"), return_tensors="pt"
            ).pixel_values.to(device)
            with torch.no_grad():
                generated = self._model.generate(
                    pixel_values,
                    max_new_tokens=64,
                    num_beams=1,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            if not generated.scores:
                return proxy
            conf = []
            for token_scores in generated.scores:
                probs = torch.softmax(token_scores, dim=-1)
                conf.append(float(probs.max(dim=-1).values.mean().item()))
            if not conf:
                return proxy
            trocr_score = float(np.clip(np.mean(conf), 0.0, 1.0))
            return float(np.clip((0.50 * proxy) + (0.50 * trocr_score), 0.0, 1.0))
        except Exception as exc:  # pragma: no cover - optional backend
            self._disabled_reason = f"inference_error:{exc.__class__.__name__}"
            return proxy


class DiffusionRefiner:
    def __init__(
        self,
        enabled: bool,
        diffusion_model_id: str,
        controlnet_model_id: str,
        steps: int,
        strength: float,
        guidance_scale: float,
        max_pages: int,
        device: str,
    ) -> None:
        self.enabled = bool(enabled)
        self.diffusion_model_id = diffusion_model_id
        self.controlnet_model_id = controlnet_model_id
        self.steps = int(steps)
        self.strength = float(strength)
        self.guidance_scale = float(guidance_scale)
        self.max_pages = int(max_pages)
        self.device = device
        self._pipe: Any = None
        self._backend = "none"
        self._disabled_reason: str | None = None

    @property
    def backend(self) -> str:
        if not self.enabled:
            return "disabled"
        if self._disabled_reason:
            return f"{self._backend}(fallback:{self._disabled_reason})"
        return self._backend

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _lazy_load(self) -> None:
        if not self.enabled or self._pipe is not None:
            return
        if (
            torch is None
            or ControlNetModel is None
            or StableDiffusionControlNetImg2ImgPipeline is None
        ):
            self._disabled_reason = "diffusers_or_torch_not_installed"
            return
        try:
            device = self._resolve_device()
            dtype = torch.float16 if device.startswith("cuda") else torch.float32
            controlnet = ControlNetModel.from_pretrained(
                self.controlnet_model_id,
                torch_dtype=dtype,
            )
            self._pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                self.diffusion_model_id,
                controlnet=controlnet,
                torch_dtype=dtype,
            )
            self._pipe.to(device)
            if hasattr(self._pipe, "set_progress_bar_config"):
                self._pipe.set_progress_bar_config(disable=True)
            if hasattr(self._pipe, "enable_attention_slicing"):
                self._pipe.enable_attention_slicing()
            self._backend = "sdxl_controlnet_img2img"
        except Exception as exc:  # pragma: no cover - optional backend
            self._disabled_reason = f"model_load_error:{exc.__class__.__name__}"

    def refine(
        self,
        image: Image.Image,
        saliency: np.ndarray,
        page_seed: int,
        page_index: int,
    ) -> tuple[Image.Image, bool]:
        if not self.enabled or page_index >= self.max_pages:
            return image, False
        self._lazy_load()
        if self._pipe is None or torch is None:
            return image, False
        try:
            edge = image.filter(ImageFilter.FIND_EDGES).convert("L")
            edge_arr = np.asarray(ImageOps.autocontrast(edge)).astype(np.float32)
            edge_arr *= (0.45 + 0.55 * np.clip(saliency, 0.0, 1.0))
            control = Image.fromarray(np.uint8(np.clip(edge_arr, 0, 255)), mode="L").convert(
                "RGB"
            )

            device = self._resolve_device()
            generator = torch.Generator(device=device).manual_seed(int(page_seed))
            output = self._pipe(
                prompt=(
                    "clean archival scanned document page, crisp text strokes, "
                    "flat illumination, preserved layout, no added content"
                ),
                negative_prompt=(
                    "new text, handwriting, decorative graphics, fantasy scene, watermark"
                ),
                image=image,
                control_image=control,
                strength=float(np.clip(self.strength, 0.05, 0.35)),
                num_inference_steps=max(8, self.steps),
                guidance_scale=float(np.clip(self.guidance_scale, 1.0, 12.0)),
                generator=generator,
            )
            refined = output.images[0].convert("RGB")
            return refined, True
        except Exception as exc:  # pragma: no cover - optional backend
            self._disabled_reason = f"inference_error:{exc.__class__.__name__}"
            return image, False


class RuntimeComponents:
    def __init__(self, config: DenoiseConfig) -> None:
        self.neural = NeuralTTARestorer(
            enabled=config.research.enable_neural_tta,
            require_deep_learning=config.research.require_deep_learning,
            steps=config.research.neural_steps,
            learning_rate=config.research.neural_learning_rate,
            mask_ratio=config.research.neural_mask_ratio,
            base_channels=config.research.neural_base_channels,
            max_side=config.research.neural_max_side,
            device=config.research.device,
        )
        self.ocr = OCRCritic(
            enabled=config.research.enable_ocr_critic,
            model_id=config.research.ocr_model_id,
            device=config.research.device,
        )
        self.diffusion = DiffusionRefiner(
            enabled=config.research.enable_diffusion_refiner,
            diffusion_model_id=config.research.diffusion_model_id,
            controlnet_model_id=config.research.controlnet_model_id,
            steps=config.research.diffusion_steps,
            strength=config.research.diffusion_strength,
            guidance_scale=config.research.diffusion_guidance_scale,
            max_pages=config.research.max_diffusion_pages,
            device=config.research.device,
        )


def runtime_for(config: DenoiseConfig) -> RuntimeComponents:
    key = json.dumps(asdict(config.research), sort_keys=True) + f"|det={config.deterministic}"
    item = _RUNTIME.get(key)
    if item is not None:
        return item
    runtime = RuntimeComponents(config)
    _RUNTIME[key] = runtime
    return runtime


def candidate_quality_score(
    candidate: Image.Image,
    runtime: RuntimeComponents,
    rng: np.random.Generator,
) -> float:
    saliency = compute_text_saliency_map(candidate)
    proxy = estimate_readability_proxy(candidate, saliency)
    ocr_like = runtime.ocr.score(candidate, saliency)

    gray = np.asarray(ImageOps.grayscale(candidate)).astype(np.float32) / 255.0
    bg_mask = saliency < 0.22
    if int(bg_mask.sum()) < 128:
        bg_mask = gray > float(np.quantile(gray, 0.74))

    bg_std = float(gray[bg_mask].std()) if int(bg_mask.sum()) else float(gray.std())
    bg_clean = 1.0 - float(np.clip(bg_std / 0.14, 0.0, 1.0))

    # Speckle noise penalty on near-white background regions.
    speckle = float(np.mean(((gray < 0.10) | (gray > 0.985)) & bg_mask))
    speckle_penalty = float(np.clip(speckle * 8.0, 0.0, 1.0))

    score = (0.56 * ocr_like) + (0.24 * proxy) + (0.28 * bg_clean) - (0.12 * speckle_penalty)
    score += float(rng.uniform(-0.0015, 0.0015))
    return score


def restore_page(
    image: Image.Image,
    page_index: int,
    config: DenoiseConfig,
    runtime: RuntimeComponents,
) -> tuple[Image.Image, dict[str, float | int]]:
    seed = derive_seed(config, f"page:{page_index}")
    rng = np.random.default_rng(seed)

    saliency = compute_text_saliency_map(image)
    pre_proxy = estimate_readability_proxy(image, saliency)
    pre_ocr = runtime.ocr.score(image, saliency)

    stage1 = flatten_illumination(image)
    stage2 = suppress_periodic_noise_fft(stage1)
    stage3 = layout_aware_denoise(stage2, saliency)
    stage4 = reconstruct_ink(stage3, saliency)
    stage5 = aggressive_document_cleanup(stage4, compute_text_saliency_map(stage4))

    neural_used = 0
    neural_candidate, used_neural = runtime.neural.restore(
        stage2,
        saliency=compute_text_saliency_map(stage2),
        page_seed=derive_seed(config, f"{seed}:neural"),
    )
    if used_neural:
        neural_used = 1

    variants = [stage4, stage5]
    if used_neural:
        variants.append(neural_candidate)
        variants.append(Image.blend(stage4, neural_candidate, alpha=0.58))
        variants.append(Image.blend(stage5, neural_candidate, alpha=0.52))
    variants.append(ImageEnhance.Contrast(stage4).enhance(1.08))
    variants.append(ImageEnhance.Sharpness(stage4).enhance(1.12))
    variants.append(Image.blend(stage3, stage4, alpha=0.65))
    variants.append(Image.blend(stage4, stage5, alpha=0.60))

    best_image = variants[0]
    best_score = -1e9
    for candidate in variants:
        score = candidate_quality_score(candidate, runtime, rng)
        if score > best_score:
            best_score = score
            best_image = candidate

    diffusion_used = 0
    refined, used = runtime.diffusion.refine(
        best_image,
        saliency=compute_text_saliency_map(best_image),
        page_seed=derive_seed(config, f"{seed}:diffusion"),
        page_index=page_index,
    )
    if used:
        best_image = refined
        diffusion_used = 1

    final = ImageEnhance.Contrast(best_image).enhance(1.03)
    final = ImageEnhance.Sharpness(final).enhance(1.05)

    post_saliency = compute_text_saliency_map(final)
    post_proxy = estimate_readability_proxy(final, post_saliency)
    post_ocr = runtime.ocr.score(final, post_saliency)

    stats = {
        "pre_proxy": pre_proxy,
        "post_proxy": post_proxy,
        "pre_ocr": pre_ocr,
        "post_ocr": post_ocr,
        "neural_used": neural_used,
        "diffusion_used": diffusion_used,
    }
    return final, stats


def write_denoised_pdf(
    source_pdf: Path,
    output_pdf: Path,
    config: DenoiseConfig,
    runtime: RuntimeComponents,
) -> DenoiseSummary:
    source_doc = fitz.open(source_pdf)
    restored_doc = fitz.open()
    page_total = source_doc.page_count
    page_limit = page_total if config.max_pages is None else min(page_total, config.max_pages)

    pre_proxy = 0.0
    post_proxy = 0.0
    pre_ocr = 0.0
    post_ocr = 0.0
    neural_pages = 0
    diffusion_pages = 0

    try:
        for page_index in range(page_limit):
            page = source_doc.load_page(page_index)
            rendered = render_page(page, config.dpi)
            restored, stats = restore_page(
                rendered, page_index=page_index, config=config, runtime=runtime
            )

            pre_proxy += float(stats["pre_proxy"])
            post_proxy += float(stats["post_proxy"])
            pre_ocr += float(stats["pre_ocr"])
            post_ocr += float(stats["post_ocr"])
            neural_pages += int(stats["neural_used"])
            diffusion_pages += int(stats["diffusion_used"])

            image_bytes = io.BytesIO()
            restored.save(
                image_bytes,
                format="JPEG",
                quality=config.jpeg_quality,
                optimize=True,
                progressive=False,
            )
            new_page = restored_doc.new_page(width=page.rect.width, height=page.rect.height)
            new_page.insert_image(page.rect, stream=image_bytes.getvalue())

        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        restored_doc.save(output_pdf, garbage=4, deflate=True)
    finally:
        restored_doc.close()
        source_doc.close()

    denom = float(max(page_limit, 1))
    return DenoiseSummary(
        pages_written=page_limit,
        mean_proxy_pre=float(pre_proxy / denom),
        mean_proxy_post=float(post_proxy / denom),
        mean_ocr_pre=float(pre_ocr / denom),
        mean_ocr_post=float(post_ocr / denom),
        neural_pages=int(neural_pages),
        diffusion_pages=int(diffusion_pages),
        neural_backend=runtime.neural.backend,
        ocr_backend=runtime.ocr.backend,
        diffusion_backend=runtime.diffusion.backend,
    )


def collect_pdfs(input_dir: Path, limit: int | None) -> list[Path]:
    pdfs = sorted(input_dir.glob("*.pdf"))
    if limit is not None:
        return pdfs[:limit]
    return pdfs


def build_manifest_row(
    source_pdf: Path,
    output_pdf: Path,
    summary: DenoiseSummary,
    config: DenoiseConfig,
    source_page_count: int,
) -> dict[str, str | int | float]:
    output_size = output_pdf.stat().st_size if output_pdf.exists() else 0
    return {
        "source_noisy_pdf_path": str(source_pdf.resolve()),
        "restored_pdf_path": str(output_pdf.resolve()),
        "source_size_bytes": source_pdf.stat().st_size,
        "restored_size_bytes": output_size,
        "source_page_count": source_page_count,
        "pages_written": summary.pages_written,
        "pipeline_name": "LATR_v2_deep",
        "deterministic": int(config.deterministic),
        "mean_readability_proxy_pre": round(summary.mean_proxy_pre, 5),
        "mean_readability_proxy_post": round(summary.mean_proxy_post, 5),
        "mean_ocr_critic_pre": round(summary.mean_ocr_pre, 5),
        "mean_ocr_critic_post": round(summary.mean_ocr_post, 5),
        "neural_backend": summary.neural_backend,
        "neural_pages": summary.neural_pages,
        "ocr_backend": summary.ocr_backend,
        "diffusion_backend": summary.diffusion_backend,
        "diffusion_pages": summary.diffusion_pages,
        "dpi": config.dpi,
        "jpeg_quality": config.jpeg_quality,
    }


def process_pdf_task(
    index: int,
    total: int,
    source_pdf_text: str,
    output_dir_text: str,
    config: DenoiseConfig,
) -> tuple[int, dict[str, str | int | float], str]:
    source_pdf = Path(source_pdf_text)
    output_dir = Path(output_dir_text)
    output_pdf = output_dir / source_pdf.name.replace("__noisy", "__denoised")
    runtime = runtime_for(config)

    source_doc = fitz.open(source_pdf)
    source_page_count = source_doc.page_count
    source_doc.close()

    if output_pdf.exists() and not config.overwrite:
        pages_written = min(
            source_page_count,
            config.max_pages if config.max_pages is not None else source_page_count,
        )
        summary = DenoiseSummary(
            pages_written=pages_written,
            mean_proxy_pre=0.0,
            mean_proxy_post=0.0,
            mean_ocr_pre=0.0,
            mean_ocr_post=0.0,
            neural_pages=0,
            diffusion_pages=0,
            neural_backend=runtime.neural.backend,
            ocr_backend=runtime.ocr.backend,
            diffusion_backend=runtime.diffusion.backend,
        )
        message = f"[skip] {source_pdf.name} -> {output_pdf.name}"
    else:
        summary = write_denoised_pdf(
            source_pdf=source_pdf,
            output_pdf=output_pdf,
            config=config,
            runtime=runtime,
        )
        message = (
            f"[{index + 1}/{total}] {source_pdf.name} restored | "
            f"proxy {summary.mean_proxy_pre:.3f}->{summary.mean_proxy_post:.3f}"
        )

    row = build_manifest_row(
        source_pdf=source_pdf,
        output_pdf=output_pdf,
        summary=summary,
        config=config,
        source_page_count=source_page_count,
    )
    return index, row, message


def build_run_summary(
    rows: list[dict[str, str | int | float]],
    config: DenoiseConfig,
) -> dict[str, Any]:
    pre_proxy = [float(row["mean_readability_proxy_pre"]) for row in rows]
    post_proxy = [float(row["mean_readability_proxy_post"]) for row in rows]
    pre_ocr = [float(row["mean_ocr_critic_pre"]) for row in rows]
    post_ocr = [float(row["mean_ocr_critic_post"]) for row in rows]
    neural_pages = int(sum(int(row["neural_pages"]) for row in rows))
    diffusion_pages = int(sum(int(row["diffusion_pages"]) for row in rows))
    pages_written = int(sum(int(row["pages_written"]) for row in rows))

    def _mean(values: list[float]) -> float:
        if not values:
            return 0.0
        return float(np.mean(values))

    return {
        "num_documents": len(rows),
        "total_pages_written": pages_written,
        "total_neural_pages": neural_pages,
        "total_diffusion_pages": diffusion_pages,
        "mean_readability_proxy_pre": round(_mean(pre_proxy), 5),
        "mean_readability_proxy_post": round(_mean(post_proxy), 5),
        "mean_ocr_critic_pre": round(_mean(pre_ocr), 5),
        "mean_ocr_critic_post": round(_mean(post_ocr), 5),
        "config": asdict(config),
        "method": {
            "name": "LATR_v2_deep",
            "stages": [
                "illumination_flattening",
                "spectral_periodic_suppression",
                "self_supervised_neural_tta_restoration",
                "layout_aware_denoising_and_ink_reconstruction",
                "ocr_guided_variant_selection",
                "optional_diffusion_refinement",
            ],
        },
    }


def main() -> None:
    args = parse_args()
    input_dir = args.input
    output_dir = args.output
    manifest_path = output_dir / args.manifest_name
    summary_path = output_dir / args.summary_name

    config = DenoiseConfig(
        dpi=int(args.dpi),
        jpeg_quality=int(args.jpeg_quality),
        seed=int(args.seed),
        deterministic=bool(args.deterministic),
        max_pages=args.max_pages,
        overwrite=bool(args.overwrite),
        research=build_research_config(args),
    )

    if config.research.require_deep_learning and torch is None:
        raise RuntimeError(
            "Deep-learning mode was required but torch is not installed. "
            "Install torch or run without --require-deep-learning."
        )

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    pdfs = collect_pdfs(input_dir, args.limit)
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in: {input_dir}")

    workers = max(1, int(args.workers))
    if (
        (config.research.enable_diffusion_refiner or config.research.enable_neural_tta)
        and workers > 1
    ):
        print(
            "[warn] Deep restoration components enabled; forcing workers=1 to avoid"
            " repeated heavy model loading/adaptation in worker processes."
        )
        workers = 1

    rows_by_index: dict[int, dict[str, str | int | float]] = {}
    if workers == 1 or len(pdfs) == 1:
        for index, source_pdf in enumerate(pdfs):
            row_index, row, message = process_pdf_task(
                index=index,
                total=len(pdfs),
                source_pdf_text=str(source_pdf),
                output_dir_text=str(output_dir),
                config=config,
            )
            print(message)
            rows_by_index[row_index] = row
    else:
        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        process_pdf_task,
                        index,
                        len(pdfs),
                        str(source_pdf),
                        str(output_dir),
                        config,
                    )
                    for index, source_pdf in enumerate(pdfs)
                ]
                for future in as_completed(futures):
                    row_index, row, message = future.result()
                    print(message)
                    rows_by_index[row_index] = row
        except (OSError, PermissionError) as exc:
            print(
                "[warn] Parallel workers unavailable in this environment "
                f"({exc}). Falling back to sequential processing."
            )
            rows_by_index.clear()
            for index, source_pdf in enumerate(pdfs):
                row_index, row, message = process_pdf_task(
                    index=index,
                    total=len(pdfs),
                    source_pdf_text=str(source_pdf),
                    output_dir_text=str(output_dir),
                    config=config,
                )
                print(message)
                rows_by_index[row_index] = row

    rows = [rows_by_index[index] for index in sorted(rows_by_index)]
    if not rows:
        raise RuntimeError("No denoise rows were produced.")

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = build_run_summary(rows, config)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"Wrote {len(rows)} denoise rows to {manifest_path}")
    print(f"Wrote denoise summary to {summary_path}")


if __name__ == "__main__":
    main()
