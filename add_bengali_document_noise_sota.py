"""Research-grade Bengali document noise generation with SOTA hooks.

This script upgrades the baseline procedural noising pipeline with:
1) An adaptive readability curriculum (novel controller),
2) Layout-aware adversarial corruption (novel corruption module),
3) Optional SOTA backends (DocLayout-YOLO, TrOCR, SDXL + ControlNet).

The code is fully reproducible and writes both per-document manifests and a
run-level JSON summary suitable for paper appendices.
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
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

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

try:
    import doclayout_yolo  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    doclayout_yolo = None


NOISE_DESCRIPTIONS = {
    "scan_geometry": "strong skew, shear, page shift, and scanner border fill",
    "uneven_illumination": "heavy lighting drift, vignette, page yellowing, and shadow",
    "paper_texture": "fibers, dust, staining, and aggressive paper-grain variation",
    "ink_bleed_fade": "glyph edge bleed, ink fading, stroke dropout, and weak patches",
    "sensor_compression": "sensor noise, row banding, blur, and harsh JPEG artifacts",
    "occlusion_damage": "soft staining, smears, edge wear, fold shadows, and partial text loss",
    "layout_adversarial_dropout": (
        "layout-aware text-region dropout and local desynchronization to stress OCR robustness"
    ),
    "periodic_moire": "periodic moire, striping, and scanner-frequency interference artifacts",
}

BASELINE_FAMILIES = (
    "scan_geometry",
    "uneven_illumination",
    "paper_texture",
    "ink_bleed_fade",
    "sensor_compression",
    "occlusion_damage",
)

NOVEL_FAMILIES = (
    "layout_adversarial_dropout",
    "periodic_moire",
)

_RUNTIME_COMPONENTS: dict[str, "RuntimeComponents"] = {}


@dataclass(frozen=True)
class ResearchConfig:
    pipeline_mode: str
    enable_adaptive_curriculum: bool
    readability_target_low: float
    readability_target_high: float
    enable_layout_prior: bool
    enable_ocr_critic: bool
    enable_diffusion_refiner: bool
    layout_model_id: str
    ocr_model_id: str
    diffusion_model_id: str
    controlnet_model_id: str
    diffusion_steps: int
    diffusion_strength: float
    diffusion_guidance_scale: float
    max_diffusion_pages: int
    device: str


@dataclass(frozen=True)
class NoiseConfig:
    dpi: int
    jpeg_quality: int
    seed: int
    min_noises: int
    max_noises: int
    max_pages: int | None
    overwrite: bool
    deterministic: bool
    research: ResearchConfig


@dataclass(frozen=True)
class DocumentProfile:
    seed: int
    noise_types: tuple[str, ...]
    severity: float
    jpeg_quality: int
    target_readability: float
    curriculum_gain: float


@dataclass(frozen=True)
class WriteSummary:
    pages_written: int
    mean_proxy_pre: float
    mean_proxy_post: float
    mean_ocr_pre: float
    mean_ocr_post: float
    diffusion_pages: int
    layout_backend: str
    ocr_backend: str
    diffusion_backend: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate publication-grade noisy Bengali PDFs and research metadata."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("Books-Bengali"),
        help="Folder containing source Bengali PDFs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Books-Bengali-Noisy-SOTA"),
        help="Folder where noisy PDFs and manifests will be written.",
    )
    parser.add_argument(
        "--manifest-name",
        default="noise_manifest.csv",
        help="CSV filename written inside the output folder.",
    )
    parser.add_argument(
        "--summary-name",
        default="run_summary.json",
        help="Run-level JSON summary written inside the output folder.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=130,
        help="Rasterization DPI. Use 150-200 for stronger visual fidelity.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=50,
        help="Base JPEG quality for noisy page images embedded into PDF.",
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
        help="Enable deterministic profiles/pages for exact reruns.",
    )
    parser.add_argument(
        "--min-noises",
        type=int,
        default=5,
        help="Minimum number of noise families sampled per document.",
    )
    parser.add_argument(
        "--max-noises",
        type=int,
        default=8,
        help="Maximum number of noise families sampled per document.",
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
        help="Optional cap on number of PDFs from the input directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate output PDFs if they already exist.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of PDFs to process in parallel.",
    )

    parser.add_argument(
        "--pipeline-mode",
        choices=("baseline", "hybrid", "sota"),
        default="hybrid",
        help="baseline: legacy families; hybrid: +novel modules; sota: +optional FM backends.",
    )
    parser.add_argument(
        "--disable-adaptive-curriculum",
        action="store_true",
        help="Disable adaptive readability controller (novel contribution).",
    )
    parser.add_argument(
        "--readability-target-low",
        type=float,
        default=0.23,
        help="Lower bound of target readability interval for curriculum control.",
    )
    parser.add_argument(
        "--readability-target-high",
        type=float,
        default=0.50,
        help="Upper bound of target readability interval for curriculum control.",
    )

    parser.add_argument(
        "--enable-layout-prior",
        action="store_true",
        help="Use DocLayout-YOLO if installed; otherwise fallback to heuristic saliency.",
    )
    parser.add_argument(
        "--enable-ocr-critic",
        action="store_true",
        help="Use TrOCR confidence as a secondary hardness metric if available.",
    )
    parser.add_argument(
        "--enable-diffusion-refiner",
        action="store_true",
        help="Use SDXL + ControlNet refinement in sota mode if dependencies are installed.",
    )
    parser.add_argument(
        "--layout-model-id",
        default="juliozhao/DocLayout-YOLO-DocStructBench",
        help="Model ID/path for DocLayout-YOLO backend.",
    )
    parser.add_argument(
        "--ocr-model-id",
        default="microsoft/trocr-large-printed",
        help="Model ID/path for TrOCR backend.",
    )
    parser.add_argument(
        "--diffusion-model-id",
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Model ID/path for SDXL image-to-image refinement.",
    )
    parser.add_argument(
        "--controlnet-model-id",
        default="diffusers/controlnet-canny-sdxl-1.0",
        help="ControlNet model ID/path for SDXL-controlled refinement.",
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=20,
        help="Diffusion denoising steps when refiner is enabled.",
    )
    parser.add_argument(
        "--diffusion-strength",
        type=float,
        default=0.20,
        help="SDXL img2img strength. Lower values preserve more layout.",
    )
    parser.add_argument(
        "--diffusion-guidance-scale",
        type=float,
        default=4.0,
        help="Guidance scale for SDXL ControlNet refinement.",
    )
    parser.add_argument(
        "--max-diffusion-pages",
        type=int,
        default=2,
        help="Cap on pages per PDF using diffusion refinement.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for optional SOTA models: auto/cpu/cuda.",
    )
    return parser.parse_args()


def build_research_config(args: argparse.Namespace) -> ResearchConfig:
    pipeline_mode = str(args.pipeline_mode)
    return ResearchConfig(
        pipeline_mode=pipeline_mode,
        enable_adaptive_curriculum=not args.disable_adaptive_curriculum,
        readability_target_low=float(args.readability_target_low),
        readability_target_high=float(args.readability_target_high),
        enable_layout_prior=bool(args.enable_layout_prior),
        enable_ocr_critic=bool(args.enable_ocr_critic),
        enable_diffusion_refiner=bool(args.enable_diffusion_refiner),
        layout_model_id=str(args.layout_model_id),
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


def derive_seed(config: NoiseConfig, token: str) -> int:
    if config.deterministic:
        return stable_seed(config.seed, token)
    return int(secrets.randbits(32))


def get_noise_catalog(config: NoiseConfig) -> list[str]:
    if config.research.pipeline_mode == "baseline":
        return list(BASELINE_FAMILIES)
    return list(BASELINE_FAMILIES + NOVEL_FAMILIES)


def make_profile(pdf_path: Path, config: NoiseConfig, index: int) -> DocumentProfile:
    doc_seed = derive_seed(config, f"{index}:{pdf_path.resolve()}")
    rng = np.random.default_rng(doc_seed)

    available = get_noise_catalog(config)
    min_noises = max(1, min(config.min_noises, len(available)))
    max_noises = max(min_noises, min(config.max_noises, len(available)))
    count = int(rng.integers(min_noises, max_noises + 1))

    sampled = list(rng.choice(available, size=count, replace=False))
    if "occlusion_damage" not in sampled:
        sampled[0] = "occlusion_damage"

    guaranteed = available[index % len(available)]
    if guaranteed not in sampled:
        replace_at = 1 if len(sampled) > 1 and sampled[0] == "occlusion_damage" else 0
        sampled[replace_at] = guaranteed

    if config.research.pipeline_mode != "baseline" and "layout_adversarial_dropout" not in sampled:
        replace_at = min(len(sampled) - 1, 2)
        sampled[replace_at] = "layout_adversarial_dropout"

    severity_low = 1.10 if config.research.pipeline_mode == "baseline" else 1.20
    severity_high = 1.85 if config.research.pipeline_mode == "baseline" else 2.25
    severity = float(rng.uniform(severity_low, severity_high))

    jitter = int(rng.integers(-10, 7))
    jpeg_quality = int(np.clip(config.jpeg_quality + jitter, 24, 78))

    r_low = min(config.research.readability_target_low, config.research.readability_target_high)
    r_high = max(config.research.readability_target_low, config.research.readability_target_high)
    target_readability = float(rng.uniform(r_low, r_high))
    curriculum_gain = float(rng.uniform(0.65, 1.35))
    return DocumentProfile(
        seed=doc_seed,
        noise_types=tuple(sampled),
        severity=severity,
        jpeg_quality=jpeg_quality,
        target_readability=target_readability,
        curriculum_gain=curriculum_gain,
    )


def render_page(page: fitz.Page, dpi: int) -> Image.Image:
    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False, colorspace=fitz.csRGB)
    return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)


def _resize_noise_field(field: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    field_min = float(field.min())
    field_max = float(field.max())
    scaled = (field - field_min) / max(field_max - field_min, 1e-6)
    image = Image.fromarray(np.uint8(scaled * 255), mode="L")
    image = image.resize(size, Image.Resampling.BICUBIC)
    resized = np.asarray(image).astype(np.float32) / 255.0
    return (resized - 0.5) * 2.0


def compute_text_saliency_map(image: Image.Image) -> np.ndarray:
    gray = np.asarray(ImageOps.grayscale(image)).astype(np.float32) / 255.0
    blurred = np.asarray(
        ImageOps.grayscale(image.filter(ImageFilter.GaussianBlur(radius=2.0)))
    ).astype(np.float32) / 255.0

    local_contrast = np.abs(gray - blurred)
    grad_y, grad_x = np.gradient(gray)
    gradient = np.sqrt((grad_x * grad_x) + (grad_y * grad_y))
    ink_likelihood = np.clip((0.84 - gray) / 0.84, 0.0, 1.0)

    lc_scale = local_contrast / max(float(local_contrast.max()), 1e-6)
    grad_scale = gradient / max(float(gradient.max()), 1e-6)
    saliency = 0.48 * ink_likelihood + 0.33 * lc_scale + 0.19 * grad_scale
    return np.clip(saliency, 0.0, 1.0).astype(np.float32)


def estimate_readability_proxy(image: Image.Image, saliency_map: np.ndarray | None = None) -> float:
    gray = np.asarray(ImageOps.grayscale(image)).astype(np.float32) / 255.0
    if saliency_map is None:
        saliency_map = compute_text_saliency_map(image)

    ink = gray < 0.74
    paper = gray > 0.89
    if int(ink.sum()) < 64:
        threshold = float(np.quantile(gray, 0.35))
        ink = gray < threshold
    if int(paper.sum()) < 64:
        threshold = float(np.quantile(gray, 0.78))
        paper = gray > threshold

    fg_mean = float(gray[ink].mean()) if int(ink.sum()) else float(gray.mean())
    bg_mean = float(gray[paper].mean()) if int(paper.sum()) else float(gray.mean())
    contrast = np.clip(bg_mean - fg_mean, 0.0, 1.0)

    grad_y, grad_x = np.gradient(gray)
    gradient = np.sqrt((grad_x * grad_x) + (grad_y * grad_y))
    stroke_energy = float(gradient[ink].mean()) if int(ink.sum()) else float(gradient.mean())

    washed = (gray > 0.95) & (saliency_map > 0.67)
    washed_ratio = float(washed.mean())
    saturation_penalty = float(np.mean((gray < 0.05) | (gray > 0.98)))

    score = 0.58 * np.clip(contrast / 0.56, 0.0, 1.0)
    score += 0.32 * np.clip(stroke_energy / 0.26, 0.0, 1.0)
    score -= 0.17 * np.clip(washed_ratio * 7.0, 0.0, 1.0)
    score -= 0.07 * np.clip(saturation_penalty * 3.5, 0.0, 1.0)
    return float(np.clip(score, 0.0, 1.0))


def apply_scan_geometry(
    image: Image.Image, rng: np.random.Generator, severity: float
) -> Image.Image:
    width, height = image.size
    fill = tuple(int(x) for x in rng.integers(236, 250, size=3))

    angle = float(rng.normal(0.0, 0.85 * severity))
    shifted = image.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=fill,
    )

    shear = float(rng.normal(0.0, 0.0032 * severity))
    x_shift = abs(shear) * height
    new_width = width + int(x_shift) + 2
    transformed = shifted.transform(
        (new_width, height),
        Image.Transform.AFFINE,
        (1, shear, -x_shift if shear > 0 else 0, 0, 1, 0),
        resample=Image.Resampling.BICUBIC,
        fillcolor=fill,
    )

    left = max(0, (new_width - width) // 2)
    transformed = transformed.crop((left, 0, left + width, height))

    dx = int(rng.integers(-5, 6))
    dy = int(rng.integers(-5, 6))
    canvas = Image.new("RGB", (width, height), fill)
    canvas.paste(transformed, (dx, dy))
    return canvas.crop((0, 0, width, height))


def apply_uneven_illumination(
    image: Image.Image, rng: np.random.Generator, severity: float
) -> Image.Image:
    width, height = image.size
    arr = np.asarray(image).astype(np.float32)

    grid_h = max(5, height // 180)
    grid_w = max(5, width // 180)
    low_freq = rng.normal(0.0, 1.0, size=(grid_h, grid_w))
    field = _resize_noise_field(low_freq, (width, height))

    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    vignette = -(x * x + y * y)
    gradient = (
        float(rng.uniform(-0.75, 0.75)) * x
        + float(rng.uniform(-0.75, 0.75)) * y
    )

    multiplier = 1.0 + (0.135 * severity * field) + (0.085 * severity * gradient)
    multiplier += 0.095 * severity * vignette
    arr *= multiplier[..., None]

    tint = np.array(
        [
            1.0 + float(rng.uniform(0.012, 0.040)) * severity,
            1.0 + float(rng.uniform(0.002, 0.020)) * severity,
            1.0 - float(rng.uniform(0.018, 0.060)) * severity,
        ],
        dtype=np.float32,
    )
    arr *= tint
    return Image.fromarray(np.uint8(np.clip(arr, 0, 255)), mode="RGB")


def apply_paper_texture(
    image: Image.Image, rng: np.random.Generator, severity: float
) -> Image.Image:
    width, height = image.size
    arr = np.asarray(image).astype(np.float32)
    gray = arr.mean(axis=2)
    paper_mask = np.clip((gray - 95.0) / 150.0, 0.0, 1.0)

    grain = rng.normal(0.0, 8.6 * severity, size=(height, width, 1))
    arr += grain * (0.50 + 1.10 * paper_mask[..., None])

    fiber_field = rng.normal(
        0.0, 1.0, size=(max(6, height // 120), max(6, width // 120))
    )
    fibers = _resize_noise_field(fiber_field, (width, height))
    arr += fibers[..., None] * (7.5 * severity) * paper_mask[..., None]

    dark_prob = 0.00045 * severity
    light_prob = 0.00024 * severity
    dark_speckles = rng.random((height, width)) < dark_prob
    light_speckles = rng.random((height, width)) < light_prob
    arr[dark_speckles] -= rng.uniform(45, 135)
    arr[light_speckles] += rng.uniform(25, 85)

    textured = Image.fromarray(np.uint8(np.clip(arr, 0, 255)), mode="RGB")
    draw = ImageDraw.Draw(textured, "RGBA")
    fiber_count = int(max(12, (width * height) / 110_000) * severity)
    for _ in range(fiber_count):
        x0 = int(rng.integers(0, width))
        y0 = int(rng.integers(0, height))
        length = int(rng.integers(max(12, width // 45), max(24, width // 18)))
        slope = float(rng.normal(0.0, 0.18))
        color_value = int(rng.integers(85, 172))
        alpha = int(rng.integers(18, 48))
        draw.line(
            (x0, y0, x0 + length, int(y0 + slope * length)),
            fill=(color_value, color_value, color_value, alpha),
            width=1,
        )
    return textured


def apply_ink_bleed_fade(
    image: Image.Image, rng: np.random.Generator, severity: float
) -> Image.Image:
    gray = np.asarray(ImageOps.grayscale(image)).astype(np.float32)
    ink_mask = gray < float(rng.uniform(92, 138))

    bleed = image.filter(ImageFilter.MinFilter(3))
    if severity > 1.35 and rng.random() < 0.45:
        bleed = bleed.filter(ImageFilter.MinFilter(3))
    image = Image.blend(image, bleed, alpha=float(rng.uniform(0.075, 0.18)) * severity)

    arr = np.asarray(image).astype(np.float32)
    height, width = gray.shape
    fade_field = rng.normal(0.0, 1.0, size=(max(5, height // 150), max(5, width // 150)))
    fade = _resize_noise_field(fade_field, (width, height))
    fade_strength = (18.0 + 34.0 * np.maximum(fade, 0.0)) * severity
    arr[ink_mask] += fade_strength[ink_mask, None]

    dropout = (rng.random((height, width)) < (0.0028 * severity)) & ink_mask
    arr[dropout] += rng.uniform(48, 130)

    weak_patch_field = rng.normal(
        0.0, 1.0, size=(max(4, height // 210), max(4, width // 210))
    )
    weak_patches = _resize_noise_field(weak_patch_field, (width, height))
    patch_mask = (weak_patches > float(rng.uniform(0.30, 0.62))) & ink_mask
    arr[patch_mask] += rng.uniform(18, 70)

    result = Image.fromarray(np.uint8(np.clip(arr, 0, 255)), mode="RGB")
    if rng.random() < 0.82:
        result = result.filter(
            ImageFilter.GaussianBlur(radius=float(rng.uniform(0.18, 0.55)))
        )
    return result


def _draw_irregular_mask_blob(
    draw: ImageDraw.ImageDraw,
    rng: np.random.Generator,
    center_x: float,
    center_y: float,
    radius_x: float,
    radius_y: float,
    fill: int,
    points: int = 32,
) -> None:
    angles = np.linspace(0, 2 * np.pi, points, endpoint=False)
    coords = []
    for angle in angles:
        jitter = float(rng.uniform(0.45, 1.35))
        x = center_x + np.cos(angle) * radius_x * jitter
        y = center_y + np.sin(angle) * radius_y * jitter
        coords.append((float(x), float(y)))
    draw.polygon(coords, fill=int(fill))


def _organic_mask(
    width: int,
    height: int,
    rng: np.random.Generator,
    count: int,
    radius_x_range: tuple[float, float],
    radius_y_range: tuple[float, float],
    blur_range: tuple[float, float],
    fill_range: tuple[int, int] = (150, 255),
) -> np.ndarray:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for _ in range(count):
        _draw_irregular_mask_blob(
            draw=draw,
            rng=rng,
            center_x=float(rng.integers(0, width)),
            center_y=float(rng.integers(0, height)),
            radius_x=width * float(rng.uniform(*radius_x_range)),
            radius_y=height * float(rng.uniform(*radius_y_range)),
            fill=int(rng.integers(fill_range[0], fill_range[1] + 1)),
            points=int(rng.integers(18, 42)),
        )
    mask = mask.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(*blur_range))))
    return np.asarray(mask).astype(np.float32) / 255.0


def apply_occlusion_damage(
    image: Image.Image, rng: np.random.Generator, severity: float
) -> Image.Image:
    width, height = image.size
    arr = np.asarray(image).astype(np.float32)
    gray = arr.mean(axis=2)
    ink_mask = (gray < float(rng.uniform(120, 155))).astype(np.float32)[..., None]
    paper_mask = (gray > float(rng.uniform(105, 138))).astype(np.float32)[..., None]

    stain_count = int(rng.integers(2, 5))
    stain_mask = _organic_mask(
        width=width,
        height=height,
        rng=rng,
        count=stain_count,
        radius_x_range=(0.08, 0.34),
        radius_y_range=(0.05, 0.22),
        blur_range=(14.0, 38.0),
        fill_range=(95, 190),
    )[..., None]
    stain_color = np.array(
        [
            float(rng.uniform(132, 177)),
            float(rng.uniform(116, 154)),
            float(rng.uniform(82, 118)),
        ],
        dtype=np.float32,
    )
    stain_strength = float(rng.uniform(0.10, 0.28)) * min(severity, 1.7)
    arr = arr * (1.0 - stain_mask * stain_strength) + stain_color * (
        stain_mask * stain_strength
    )
    arr += ink_mask * stain_mask * float(rng.uniform(10, 34)) * min(severity, 1.8)

    tear_count = int(rng.integers(1, 4))
    tear_mask = Image.new("L", (width, height), 0)
    tear_draw = ImageDraw.Draw(tear_mask)
    for _ in range(tear_count):
        side = str(rng.choice(["left", "right", "top", "bottom"]))
        if side in {"left", "right"}:
            edge_x = 0 if side == "left" else width
            y0 = int(rng.integers(0, height))
            depth = int(width * float(rng.uniform(0.025, 0.12)))
            span = int(height * float(rng.uniform(0.06, 0.22)))
            x_inner = depth if side == "left" else width - depth
            points = [
                (edge_x, y0),
                (edge_x, min(height, y0 + span)),
                (x_inner, y0 + span // 2),
            ]
        else:
            edge_y = 0 if side == "top" else height
            x0 = int(rng.integers(0, width))
            depth = int(height * float(rng.uniform(0.02, 0.10)))
            span = int(width * float(rng.uniform(0.08, 0.25)))
            y_inner = depth if side == "top" else height - depth
            points = [
                (x0, edge_y),
                (min(width, x0 + span), edge_y),
                (x0 + span // 2, y_inner),
            ]
        tear_draw.polygon(points, fill=int(rng.integers(170, 245)))
    tear_mask = tear_mask.filter(
        ImageFilter.GaussianBlur(radius=float(rng.uniform(1.0, 3.5)))
    )
    tear = np.asarray(tear_mask).astype(np.float32)[..., None] / 255.0
    torn_paper = np.array(
        [
            float(rng.uniform(226, 246)),
            float(rng.uniform(220, 241)),
            float(rng.uniform(204, 230)),
        ],
        dtype=np.float32,
    )
    arr = arr * (1.0 - tear) + torn_paper * tear

    smear_mask = Image.new("L", (width, height), 0)
    smear_draw = ImageDraw.Draw(smear_mask)
    smear_count = int(rng.integers(4, 9))
    for _ in range(smear_count):
        y = int(rng.integers(0, height))
        x = int(rng.integers(0, width))
        length = int(width * float(rng.uniform(0.18, 0.75)))
        thickness = int(max(4, height * float(rng.uniform(0.006, 0.028))))
        alpha = int(rng.integers(40, 120))
        smear_draw.line(
            (x, y, min(width, x + length), int(y + rng.normal(0, thickness * 1.5))),
            fill=alpha,
            width=thickness,
        )
    smear_mask = smear_mask.filter(
        ImageFilter.GaussianBlur(radius=float(rng.uniform(3, 11)))
    )
    smear = np.asarray(smear_mask).astype(np.float32)[..., None] / 255.0
    arr *= 1.0 - smear * float(rng.uniform(0.18, 0.40))
    arr += ink_mask * smear * float(rng.uniform(12, 35))

    wash_mask = Image.new("L", (width, height), 0)
    wash_draw = ImageDraw.Draw(wash_mask)
    wash_count = int(rng.integers(2, 6))
    for _ in range(wash_count):
        y0 = int(rng.integers(height // 10, max(height // 10 + 1, height * 9 // 10)))
        x0 = int(rng.integers(0, max(1, width // 5)))
        x1 = int(rng.integers(max(width // 2, x0 + 1), width))
        segments = int(rng.integers(7, 15))
        points = []
        for step in range(segments + 1):
            x = x0 + (x1 - x0) * step / segments
            y = y0 + float(rng.normal(0, height * 0.006))
            points.append((x, y))
        wash_draw.line(
            points,
            fill=int(rng.integers(80, 170)),
            width=int(max(2, height * float(rng.uniform(0.004, 0.015)))),
        )
    wash_mask = wash_mask.filter(
        ImageFilter.GaussianBlur(radius=float(rng.uniform(2.0, 7.0)))
    )
    wash = np.asarray(wash_mask).astype(np.float32)[..., None] / 255.0
    arr += ink_mask * wash * float(rng.uniform(35, 90)) * min(severity, 1.8)
    arr += paper_mask * wash * float(rng.uniform(2, 18))

    shadow_layer = Image.new("L", (width, height), 0)
    shadow_draw = ImageDraw.Draw(shadow_layer)
    if rng.random() < 0.88:
        side = str(rng.choice(["left", "right", "top", "bottom"]))
        if side in {"left", "right"}:
            band_w = int(width * float(rng.uniform(0.035, 0.14)))
            x0 = 0 if side == "left" else width - band_w
            shadow_draw.rectangle(
                (x0, 0, x0 + band_w, height),
                fill=int(rng.integers(85, 175)),
            )
        else:
            band_h = int(height * float(rng.uniform(0.025, 0.11)))
            y0 = 0 if side == "top" else height - band_h
            shadow_draw.rectangle(
                (0, y0, width, y0 + band_h),
                fill=int(rng.integers(70, 155)),
            )
    fold_count = int(rng.integers(1, 4))
    for _ in range(fold_count):
        vertical = bool(rng.random() < 0.45)
        line_alpha = int(rng.integers(38, 110))
        if vertical:
            x = int(rng.integers(width // 8, max(width // 8 + 1, width * 7 // 8)))
            shadow_draw.line(
                (x, 0, x + int(rng.normal(0, 12)), height),
                fill=line_alpha,
                width=int(rng.integers(3, 10)),
            )
        else:
            y = int(rng.integers(height // 8, max(height // 8 + 1, height * 7 // 8)))
            shadow_draw.line(
                (0, y, width, y + int(rng.normal(0, 12))),
                fill=line_alpha,
                width=int(rng.integers(3, 10)),
            )
    shadow_layer = shadow_layer.filter(
        ImageFilter.GaussianBlur(radius=float(rng.uniform(4, 14)))
    )
    shadow = np.asarray(shadow_layer).astype(np.float32)[..., None] / 255.0
    arr *= 1.0 - shadow * float(rng.uniform(0.16, 0.36))

    y_grid = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    x_grid = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    left = np.broadcast_to((x_grid + 1.0) / 2.0, (height, width))
    right = np.broadcast_to((1.0 - x_grid) / 2.0, (height, width))
    top = np.broadcast_to((y_grid + 1.0) / 2.0, (height, width))
    bottom = np.broadcast_to((1.0 - y_grid) / 2.0, (height, width))
    edge_distance = np.minimum(np.minimum(left, right), np.minimum(top, bottom))
    edge_grime = np.clip(1.0 - edge_distance / float(rng.uniform(0.08, 0.18)), 0.0, 1.0)
    arr *= 1.0 - edge_grime[..., None] * float(rng.uniform(0.04, 0.13)) * min(severity, 1.6)

    return Image.fromarray(np.uint8(np.clip(arr, 0, 255)), mode="RGB")


def apply_sensor_compression(
    image: Image.Image,
    rng: np.random.Generator,
    severity: float,
    jpeg_quality: int,
) -> Image.Image:
    width, height = image.size
    arr = np.asarray(image).astype(np.float32)

    noise = rng.normal(0.0, 6.8 * severity, size=arr.shape)
    arr += noise

    y = np.arange(height, dtype=np.float32)
    row_wave = np.sin((y * float(rng.uniform(0.035, 0.085))) + float(rng.uniform(0, 6.28)))
    row_noise = rng.normal(0.0, 1.0, size=height)
    bands = (row_wave * 4.2 * severity) + (row_noise * 1.2 * severity)
    arr += bands[:, None, None]

    compressed = Image.fromarray(np.uint8(np.clip(arr, 0, 255)), mode="RGB")
    blur_radius = float(rng.uniform(0.12, 0.85)) * severity
    if blur_radius > 0.05:
        compressed = compressed.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    buffer = io.BytesIO()
    quality = int(np.clip(jpeg_quality + int(rng.integers(-12, 6)), 22, 72))
    compressed.save(buffer, format="JPEG", quality=quality, optimize=True)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def apply_layout_adversarial_dropout(
    image: Image.Image,
    rng: np.random.Generator,
    severity: float,
    saliency_map: np.ndarray | None = None,
) -> Image.Image:
    width, height = image.size
    arr = np.asarray(image).astype(np.float32)
    if saliency_map is None:
        saliency_map = compute_text_saliency_map(image)
    saliency = np.clip(saliency_map, 0.0, 1.0)

    drop_prob = (0.0018 + 0.0034 * severity) * (0.30 + 0.70 * saliency)
    dropout = rng.random((height, width)) < drop_prob
    if np.any(dropout):
        arr[dropout] += rng.uniform(65, 175)

    line_mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(line_mask)
    streaks = int(rng.integers(5, 14))
    for _ in range(streaks):
        y = int(rng.integers(0, height))
        x0 = int(rng.integers(0, width))
        x1 = min(width, x0 + int(width * float(rng.uniform(0.12, 0.44))))
        thickness = int(max(1, height * float(rng.uniform(0.0015, 0.0085))))
        draw.line(
            (x0, y, x1, int(y + rng.normal(0, max(1.0, thickness * 1.2)))),
            fill=int(rng.integers(75, 190)),
            width=thickness,
        )
    line_mask = line_mask.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(1.0, 4.2))))
    smear = np.asarray(line_mask).astype(np.float32) / 255.0
    arr += smear[..., None] * saliency[..., None] * float(rng.uniform(18, 58))

    row_shift_prob = float(np.clip(0.015 * severity, 0.01, 0.085))
    row_selector = rng.random(height) < row_shift_prob
    if np.any(row_selector):
        shifted = arr.copy()
        for row in np.where(row_selector)[0]:
            shift = int(rng.integers(-8, 9))
            shifted[row] = np.roll(shifted[row], shift=shift, axis=0)
        alpha = float(rng.uniform(0.12, 0.26))
        arr = (1.0 - alpha) * arr + alpha * shifted

    return Image.fromarray(np.uint8(np.clip(arr, 0, 255)), mode="RGB")


def apply_periodic_moire(
    image: Image.Image, rng: np.random.Generator, severity: float
) -> Image.Image:
    width, height = image.size
    arr = np.asarray(image).astype(np.float32)
    gray = arr.mean(axis=2)
    paper_mask = np.clip((gray - 90.0) / 155.0, 0.0, 1.0)

    y, x = np.mgrid[0:height, 0:width]
    angle = float(rng.uniform(0, np.pi))
    frequency = float(rng.uniform(0.015, 0.065))
    phase = float(rng.uniform(0, 2 * np.pi))
    axis = np.cos(angle) * x + np.sin(angle) * y
    wave = np.sin(axis * frequency + phase)

    second_wave = np.sin(
        axis * float(rng.uniform(0.023, 0.095)) + float(rng.uniform(0, 2 * np.pi))
    )
    combined = 0.62 * wave + 0.38 * second_wave
    amplitude = float(rng.uniform(6.0, 21.0)) * severity
    arr += combined[..., None] * amplitude * (0.35 + 0.65 * paper_mask[..., None])

    stripe_period = int(rng.integers(7, 31))
    stripe_strength = float(rng.uniform(1.5, 7.5)) * severity
    stripe = (((y + int(rng.integers(0, stripe_period))) % stripe_period) == 0).astype(np.float32)
    arr += stripe[..., None] * stripe_strength

    return Image.fromarray(np.uint8(np.clip(arr, 0, 255)), mode="RGB")


def _extract_layout_boxes_from_result(result_obj: Any) -> list[tuple[int, int, int, int]]:
    boxes_list: list[tuple[int, int, int, int]] = []
    if result_obj is None:
        return boxes_list

    boxes_obj = getattr(result_obj, "boxes", None)
    if boxes_obj is None:
        return boxes_list
    xyxy = getattr(boxes_obj, "xyxy", None)
    if xyxy is None:
        return boxes_list
    if hasattr(xyxy, "cpu"):
        xyxy = xyxy.cpu().numpy()
    for row in xyxy:
        if len(row) < 4:
            continue
        x0, y0, x1, y1 = [int(v) for v in row[:4]]
        if x1 > x0 and y1 > y0:
            boxes_list.append((x0, y0, x1, y1))
    return boxes_list


class LayoutPriorExtractor:
    def __init__(self, enabled: bool, model_id: str) -> None:
        self.enabled = bool(enabled)
        self.model_id = model_id
        self._model: Any = None
        self._backend = "heuristic_saliency"
        self._disabled_reason: str | None = None

    @property
    def backend(self) -> str:
        if self._disabled_reason:
            return f"{self._backend}(fallback:{self._disabled_reason})"
        return self._backend

    def _lazy_load(self) -> None:
        if self._model is not None or not self.enabled:
            return
        if doclayout_yolo is None:
            self._disabled_reason = "doclayout_yolo_not_installed"
            return
        try:
            if hasattr(doclayout_yolo, "YOLOv10"):
                self._model = doclayout_yolo.YOLOv10(self.model_id)
                self._backend = "doclayout_yolo"
            elif hasattr(doclayout_yolo, "DocLayoutYOLO"):
                self._model = doclayout_yolo.DocLayoutYOLO(self.model_id)
                self._backend = "doclayout_yolo"
            else:
                self._disabled_reason = "unknown_doclayout_api"
        except Exception as exc:  # pragma: no cover - optional backend
            self._disabled_reason = f"model_load_error:{exc.__class__.__name__}"

    def extract_map(self, image: Image.Image) -> np.ndarray:
        base = compute_text_saliency_map(image)
        if not self.enabled:
            return base
        self._lazy_load()
        if self._model is None:
            return base
        try:
            image_np = np.asarray(image)
            prediction = self._model.predict(image_np, verbose=False)
            results = prediction if isinstance(prediction, list) else [prediction]
            boxes = _extract_layout_boxes_from_result(results[0] if results else None)
            if not boxes:
                return base

            height, width = base.shape
            mask = np.zeros((height, width), dtype=np.float32)
            for x0, y0, x1, y1 in boxes:
                x0 = int(np.clip(x0, 0, width - 1))
                x1 = int(np.clip(x1, x0 + 1, width))
                y0 = int(np.clip(y0, 0, height - 1))
                y1 = int(np.clip(y1, y0 + 1, height))
                mask[y0:y1, x0:x1] = 1.0

            smoothed = Image.fromarray(np.uint8(mask * 255), mode="L").filter(
                ImageFilter.GaussianBlur(radius=8.0)
            )
            mask_soft = np.asarray(smoothed).astype(np.float32) / 255.0
            return np.clip(0.45 * base + 0.55 * mask_soft, 0.0, 1.0)
        except Exception as exc:  # pragma: no cover - optional backend
            self._disabled_reason = f"inference_error:{exc.__class__.__name__}"
            return base


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

    def score(self, image: Image.Image, saliency_map: np.ndarray | None = None) -> float:
        proxy = estimate_readability_proxy(image, saliency_map=saliency_map)
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
            confidence = []
            for token_scores in generated.scores:
                probs = torch.softmax(token_scores, dim=-1)
                confidence.append(float(probs.max(dim=-1).values.mean().item()))
            if not confidence:
                return proxy
            trocr_score = float(np.clip(np.mean(confidence), 0.0, 1.0))
            return float(np.clip((0.55 * proxy) + (0.45 * trocr_score), 0.0, 1.0))
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
        self._pipeline: Any = None
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
        if not self.enabled or self._pipeline is not None:
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
            self._pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                self.diffusion_model_id,
                controlnet=controlnet,
                torch_dtype=dtype,
            )
            self._pipeline.to(device)
            if hasattr(self._pipeline, "set_progress_bar_config"):
                self._pipeline.set_progress_bar_config(disable=True)
            if hasattr(self._pipeline, "enable_attention_slicing"):
                self._pipeline.enable_attention_slicing()
            self._backend = "sdxl_controlnet_img2img"
        except Exception as exc:  # pragma: no cover - optional backend
            self._disabled_reason = f"model_load_error:{exc.__class__.__name__}"

    def refine(
        self,
        image: Image.Image,
        saliency_map: np.ndarray,
        page_seed: int,
        page_index: int,
    ) -> tuple[Image.Image, bool]:
        if not self.enabled or page_index >= self.max_pages:
            return image, False
        self._lazy_load()
        if self._pipeline is None or torch is None:
            return image, False

        try:
            edge = image.filter(ImageFilter.FIND_EDGES).convert("L")
            edge_arr = np.asarray(ImageOps.autocontrast(edge)).astype(np.float32)
            edge_arr *= (0.35 + 0.65 * np.clip(saliency_map, 0.0, 1.0))
            edge_arr = np.uint8(np.clip(edge_arr, 0, 255))
            control = Image.fromarray(edge_arr, mode="L").convert("RGB")

            device = self._resolve_device()
            generator = torch.Generator(device=device).manual_seed(int(page_seed))
            prompt = (
                "highly realistic archival scanned document, degraded print texture, "
                "local stains, subtle warping, OCR challenging but plausible"
            )
            negative_prompt = (
                "clean digital render, cartoon, surreal objects, extra handwriting, "
                "new text content, modern UI elements"
            )
            output = self._pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                control_image=control,
                strength=float(np.clip(self.strength, 0.05, 0.45)),
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
    def __init__(self, research: ResearchConfig) -> None:
        self.layout = LayoutPriorExtractor(
            enabled=research.enable_layout_prior,
            model_id=research.layout_model_id,
        )
        self.ocr = OCRCritic(
            enabled=research.enable_ocr_critic,
            model_id=research.ocr_model_id,
            device=research.device,
        )
        enable_diffusion = (
            research.pipeline_mode == "sota" and research.enable_diffusion_refiner
        )
        self.diffusion = DiffusionRefiner(
            enabled=enable_diffusion,
            diffusion_model_id=research.diffusion_model_id,
            controlnet_model_id=research.controlnet_model_id,
            steps=research.diffusion_steps,
            strength=research.diffusion_strength,
            guidance_scale=research.diffusion_guidance_scale,
            max_pages=research.max_diffusion_pages,
            device=research.device,
        )


def runtime_components_for(config: NoiseConfig) -> RuntimeComponents:
    key = json.dumps(asdict(config.research), sort_keys=True)
    cached = _RUNTIME_COMPONENTS.get(key)
    if cached is not None:
        return cached
    runtime = RuntimeComponents(config.research)
    _RUNTIME_COMPONENTS[key] = runtime
    return runtime


def apply_noises(
    image: Image.Image,
    profile: DocumentProfile,
    page_index: int,
    config: NoiseConfig,
    runtime: RuntimeComponents,
) -> tuple[Image.Image, dict[str, float | int]]:
    page_seed = derive_seed(config, f"{profile.seed}:page:{page_index}")
    rng = np.random.default_rng(page_seed)

    saliency = runtime.layout.extract_map(image)
    pre_proxy = estimate_readability_proxy(image, saliency)
    pre_ocr = runtime.ocr.score(image, saliency)

    result = image
    ordered_types = list(profile.noise_types)
    rng.shuffle(ordered_types)

    # Keep heavy structural corruptions toward the tail.
    def _priority(noise_name: str) -> int:
        if noise_name == "occlusion_damage":
            return 2
        if noise_name == "layout_adversarial_dropout":
            return 1
        return 0

    ordered_types.sort(key=_priority)

    controller = 1.0
    last_proxy = pre_proxy
    for noise_type in ordered_types:
        page_severity = float(
            np.clip(
                profile.severity * rng.uniform(0.86, 1.32) * controller,
                0.80,
                2.85,
            )
        )
        if noise_type == "scan_geometry":
            result = apply_scan_geometry(result, rng, page_severity)
        elif noise_type == "uneven_illumination":
            result = apply_uneven_illumination(result, rng, page_severity)
        elif noise_type == "paper_texture":
            result = apply_paper_texture(result, rng, page_severity)
        elif noise_type == "ink_bleed_fade":
            result = apply_ink_bleed_fade(result, rng, page_severity)
        elif noise_type == "sensor_compression":
            result = apply_sensor_compression(
                result,
                rng,
                page_severity,
                profile.jpeg_quality,
            )
        elif noise_type == "occlusion_damage":
            result = apply_occlusion_damage(result, rng, page_severity)
        elif noise_type == "layout_adversarial_dropout":
            result = apply_layout_adversarial_dropout(
                result,
                rng,
                page_severity,
                saliency_map=saliency,
            )
        elif noise_type == "periodic_moire":
            result = apply_periodic_moire(result, rng, page_severity)

        if config.research.enable_adaptive_curriculum:
            saliency = runtime.layout.extract_map(result)
            current_proxy = estimate_readability_proxy(result, saliency)
            gap = current_proxy - profile.target_readability
            controller = float(
                np.clip(controller + (0.26 * gap * profile.curriculum_gain), 0.65, 1.78)
            )
            last_proxy = current_proxy

    if (
        config.research.pipeline_mode != "baseline"
        and last_proxy > profile.target_readability + 0.08
    ):
        result = apply_layout_adversarial_dropout(
            result,
            rng,
            severity=float(np.clip(profile.severity * 1.2, 0.9, 2.9)),
            saliency_map=saliency,
        )

    diffusion_used = 0
    if config.research.pipeline_mode == "sota":
        refined, used = runtime.diffusion.refine(
            result,
            saliency_map=saliency,
            page_seed=derive_seed(config, f"{page_seed}:diffusion"),
            page_index=page_index,
        )
        result = refined
        diffusion_used = int(used)

    contrast = ImageEnhance.Contrast(result)
    result = contrast.enhance(float(rng.uniform(0.75, 1.12)))
    sharpness = ImageEnhance.Sharpness(result)
    result = sharpness.enhance(float(rng.uniform(0.54, 0.98)))

    post_saliency = runtime.layout.extract_map(result)
    post_proxy = estimate_readability_proxy(result, post_saliency)
    post_ocr = runtime.ocr.score(result, post_saliency)
    stats = {
        "pre_proxy": pre_proxy,
        "post_proxy": post_proxy,
        "pre_ocr": pre_ocr,
        "post_ocr": post_ocr,
        "target_readability": profile.target_readability,
        "diffusion_used": diffusion_used,
    }
    return result, stats


def write_noisy_pdf(
    source_pdf: Path,
    output_pdf: Path,
    profile: DocumentProfile,
    config: NoiseConfig,
    runtime: RuntimeComponents,
) -> WriteSummary:
    source_doc = fitz.open(source_pdf)
    noisy_doc = fitz.open()
    page_total = source_doc.page_count
    page_limit = page_total if config.max_pages is None else min(page_total, config.max_pages)

    sum_pre_proxy = 0.0
    sum_post_proxy = 0.0
    sum_pre_ocr = 0.0
    sum_post_ocr = 0.0
    diffusion_pages = 0

    try:
        for page_index in range(page_limit):
            source_page = source_doc.load_page(page_index)
            rendered = render_page(source_page, config.dpi)
            noisy_image, stats = apply_noises(
                rendered,
                profile=profile,
                page_index=page_index,
                config=config,
                runtime=runtime,
            )
            sum_pre_proxy += float(stats["pre_proxy"])
            sum_post_proxy += float(stats["post_proxy"])
            sum_pre_ocr += float(stats["pre_ocr"])
            sum_post_ocr += float(stats["post_ocr"])
            diffusion_pages += int(stats["diffusion_used"])

            image_bytes = io.BytesIO()
            noisy_image.save(
                image_bytes,
                format="JPEG",
                quality=profile.jpeg_quality,
                optimize=True,
                progressive=False,
            )

            new_page = noisy_doc.new_page(
                width=source_page.rect.width,
                height=source_page.rect.height,
            )
            new_page.insert_image(source_page.rect, stream=image_bytes.getvalue())

        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        noisy_doc.save(output_pdf, garbage=4, deflate=True)
    finally:
        noisy_doc.close()
        source_doc.close()

    denom = float(max(page_limit, 1))
    return WriteSummary(
        pages_written=page_limit,
        mean_proxy_pre=float(sum_pre_proxy / denom),
        mean_proxy_post=float(sum_post_proxy / denom),
        mean_ocr_pre=float(sum_pre_ocr / denom),
        mean_ocr_post=float(sum_post_ocr / denom),
        diffusion_pages=int(diffusion_pages),
        layout_backend=runtime.layout.backend,
        ocr_backend=runtime.ocr.backend,
        diffusion_backend=runtime.diffusion.backend,
    )


def build_manifest_row(
    source_pdf: Path,
    output_pdf: Path,
    profile: DocumentProfile,
    summary: WriteSummary,
    config: NoiseConfig,
    source_page_count: int,
) -> dict[str, str | int | float]:
    output_size = output_pdf.stat().st_size if output_pdf.exists() else 0
    return {
        "source_pdf_path": str(source_pdf.resolve()),
        "noisy_pdf_path": str(output_pdf.resolve()),
        "source_size_bytes": source_pdf.stat().st_size,
        "noisy_size_bytes": output_size,
        "source_page_count": source_page_count,
        "pages_written": summary.pages_written,
        "noise_types": "|".join(profile.noise_types),
        "noise_descriptions": json.dumps(
            {name: NOISE_DESCRIPTIONS[name] for name in profile.noise_types},
            ensure_ascii=False,
        ),
        "profile_seed": profile.seed,
        "severity": round(profile.severity, 4),
        "target_readability": round(profile.target_readability, 4),
        "curriculum_gain": round(profile.curriculum_gain, 4),
        "pipeline_mode": config.research.pipeline_mode,
        "deterministic": int(config.deterministic),
        "adaptive_curriculum": int(config.research.enable_adaptive_curriculum),
        "layout_backend": summary.layout_backend,
        "ocr_backend": summary.ocr_backend,
        "diffusion_backend": summary.diffusion_backend,
        "diffusion_pages": summary.diffusion_pages,
        "mean_readability_proxy_pre": round(summary.mean_proxy_pre, 5),
        "mean_readability_proxy_post": round(summary.mean_proxy_post, 5),
        "mean_ocr_critic_pre": round(summary.mean_ocr_pre, 5),
        "mean_ocr_critic_post": round(summary.mean_ocr_post, 5),
        "dpi": config.dpi,
        "jpeg_quality": profile.jpeg_quality,
    }


def collect_pdfs(input_dir: Path, limit: int | None) -> list[Path]:
    pdfs = sorted(input_dir.glob("*.pdf"))
    if limit is not None:
        return pdfs[:limit]
    return pdfs


def process_pdf_task(
    index: int,
    total: int,
    source_pdf_text: str,
    output_dir_text: str,
    config: NoiseConfig,
) -> tuple[int, dict[str, str | int | float], str]:
    source_pdf = Path(source_pdf_text)
    output_dir = Path(output_dir_text)
    output_pdf = output_dir / f"{source_pdf.stem}__noisy.pdf"
    profile = make_profile(source_pdf, config, index)
    runtime = runtime_components_for(config)

    source_doc = fitz.open(source_pdf)
    source_page_count = source_doc.page_count
    source_doc.close()

    if output_pdf.exists() and not config.overwrite:
        pages_written = min(
            source_page_count,
            config.max_pages if config.max_pages is not None else source_page_count,
        )
        summary = WriteSummary(
            pages_written=pages_written,
            mean_proxy_pre=0.0,
            mean_proxy_post=0.0,
            mean_ocr_pre=0.0,
            mean_ocr_post=0.0,
            diffusion_pages=0,
            layout_backend=runtime.layout.backend,
            ocr_backend=runtime.ocr.backend,
            diffusion_backend=runtime.diffusion.backend,
        )
        message = f"[skip] {source_pdf.name} -> {output_pdf.name}"
    else:
        summary = write_noisy_pdf(
            source_pdf=source_pdf,
            output_pdf=output_pdf,
            profile=profile,
            config=config,
            runtime=runtime,
        )
        message = (
            f"[{index + 1}/{total}] {source_pdf.name}: "
            f"{', '.join(profile.noise_types)} | target={profile.target_readability:.2f}"
        )

    row = build_manifest_row(
        source_pdf=source_pdf,
        output_pdf=output_pdf,
        profile=profile,
        summary=summary,
        config=config,
        source_page_count=source_page_count,
    )
    return index, row, message


def build_run_summary(
    rows: list[dict[str, str | int | float]],
    config: NoiseConfig,
) -> dict[str, Any]:
    post_proxy = [float(row["mean_readability_proxy_post"]) for row in rows]
    post_ocr = [float(row["mean_ocr_critic_post"]) for row in rows]
    pre_proxy = [float(row["mean_readability_proxy_pre"]) for row in rows]
    pre_ocr = [float(row["mean_ocr_critic_pre"]) for row in rows]
    diffusion_pages = int(sum(int(row["diffusion_pages"]) for row in rows))
    pages_written = int(sum(int(row["pages_written"]) for row in rows))

    def _safe_mean(values: list[float]) -> float:
        if not values:
            return 0.0
        return float(np.mean(values))

    return {
        "num_documents": len(rows),
        "total_pages_written": pages_written,
        "total_diffusion_pages": diffusion_pages,
        "mean_readability_proxy_pre": round(_safe_mean(pre_proxy), 5),
        "mean_readability_proxy_post": round(_safe_mean(post_proxy), 5),
        "mean_ocr_critic_pre": round(_safe_mean(pre_ocr), 5),
        "mean_ocr_critic_post": round(_safe_mean(post_ocr), 5),
        "config": asdict(config),
        "notes": {
            "novelty": (
                "Adaptive readability curriculum + layout-aware adversarial dropout"
            ),
            "sota_backends": {
                "layout": "DocLayout-YOLO (optional)",
                "ocr_critic": "TrOCR confidence (optional)",
                "diffusion_refiner": "SDXL + ControlNet Img2Img (optional)",
            },
        },
    }


def main() -> None:
    args = parse_args()
    input_dir = args.input
    output_dir = args.output
    manifest_path = output_dir / args.manifest_name
    summary_path = output_dir / args.summary_name

    config = NoiseConfig(
        dpi=int(args.dpi),
        jpeg_quality=int(args.jpeg_quality),
        seed=int(args.seed),
        min_noises=int(args.min_noises),
        max_noises=int(args.max_noises),
        max_pages=args.max_pages,
        overwrite=bool(args.overwrite),
        deterministic=bool(args.deterministic),
        research=build_research_config(args),
    )

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    pdfs = collect_pdfs(input_dir, args.limit)
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in: {input_dir}")

    workers = max(1, int(args.workers))
    if config.research.enable_diffusion_refiner and workers > 1:
        print(
            "[warn] Diffusion refiner is enabled; forcing workers=1 to avoid"
            " repeated heavy model loading across worker processes."
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
        raise RuntimeError("No rows were produced. Generation failed unexpectedly.")

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    run_summary = build_run_summary(rows, config)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, ensure_ascii=False, indent=2)

    print(f"Wrote {len(rows)} manifest rows to {manifest_path}")
    print(f"Wrote run summary to {summary_path}")


if __name__ == "__main__":
    main()
