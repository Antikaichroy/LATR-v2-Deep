"""Generate realistically degraded Bengali PDF variants.

The script treats each PDF page as a scanned document image, applies a
deterministic mixture of natural document noises, writes noisy PDFs to a new
folder, and records a reproducible manifest CSV.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import fitz
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps


NOISE_DESCRIPTIONS = {
    "scan_geometry": "strong skew, shear, page shift, and scanner border fill",
    "uneven_illumination": "heavy lighting drift, vignette, page yellowing, and shadow",
    "paper_texture": "fibers, dust, staining, and aggressive paper-grain variation",
    "ink_bleed_fade": "glyph edge bleed, ink fading, stroke dropout, and weak patches",
    "sensor_compression": "sensor noise, row banding, blur, and harsh JPEG artifacts",
    "occlusion_damage": "soft staining, smears, edge wear, fold shadows, and partial text loss",
}


@dataclass(frozen=True)
class DocumentProfile:
    seed: int
    noise_types: tuple[str, ...]
    severity: float
    jpeg_quality: int


@dataclass(frozen=True)
class NoiseConfig:
    dpi: int
    jpeg_quality: int
    seed: int
    min_noises: int
    max_noises: int
    max_pages: int | None
    overwrite: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create noisy Bengali PDF variants and a CSV manifest."
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
        default=Path("Books-Bengali-Noisy"),
        help="Folder where noisy PDFs and the manifest CSV will be written.",
    )
    parser.add_argument(
        "--manifest-name",
        default="noise_manifest.csv",
        help="CSV filename written inside the output folder.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=110,
        help="Rasterization DPI. Use 140-180 for higher-fidelity experiments.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=48,
        help="Base JPEG quality for noisy page images embedded into the PDF.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260423,
        help="Global seed used to make the corpus reproducible.",
    )
    parser.add_argument(
        "--min-noises",
        type=int,
        default=4,
        help="Minimum number of noise families sampled for each PDF.",
    )
    parser.add_argument(
        "--max-noises",
        type=int,
        default=6,
        help="Maximum number of noise families sampled for each PDF.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional debug cap on pages per PDF. Omit for full documents.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional debug cap on number of PDFs. Omit for the full folder.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate PDFs even if matching output files already exist.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of PDFs to process in parallel. Use 1 for deterministic console order.",
    )
    return parser.parse_args()


def stable_seed(global_seed: int, value: str) -> int:
    payload = f"{global_seed}:{value}".encode("utf-8", errors="surrogatepass")
    return int(hashlib.blake2b(payload, digest_size=8).hexdigest(), 16) % (2**32)


def make_profile(pdf_path: Path, config: NoiseConfig, index: int) -> DocumentProfile:
    doc_seed = stable_seed(config.seed, f"{index}:{pdf_path.resolve()}")
    rng = np.random.default_rng(doc_seed)
    available = list(NOISE_DESCRIPTIONS)
    min_noises = max(1, min(config.min_noises, len(available)))
    max_noises = max(min_noises, min(config.max_noises, len(available)))
    count = int(rng.integers(min_noises, max_noises + 1))

    sampled = list(rng.choice(available, size=count, replace=False))
    # Every severe profile includes true text occlusion, then rotates coverage.
    if "occlusion_damage" not in sampled:
        sampled[0] = "occlusion_damage"
    guaranteed = available[index % len(available)]
    if guaranteed not in sampled:
        replace_at = 1 if len(sampled) > 1 and sampled[0] == "occlusion_damage" else 0
        sampled[replace_at] = guaranteed

    severity = float(rng.uniform(1.18, 1.85))
    jitter = int(rng.integers(-12, 7))
    jpeg_quality = int(np.clip(config.jpeg_quality + jitter, 25, 76))
    return DocumentProfile(
        seed=doc_seed,
        noise_types=tuple(sampled),
        severity=severity,
        jpeg_quality=jpeg_quality,
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

    fiber_field = rng.normal(0.0, 1.0, size=(max(6, height // 120), max(6, width // 120)))
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
        result = result.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.18, 0.55))))
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
    tear_mask = tear_mask.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(1.0, 3.5))))
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
    smear_mask = smear_mask.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(3, 11))))
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
    wash_mask = wash_mask.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(2.0, 7.0))))
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
            shadow_draw.line((x, 0, x + int(rng.normal(0, 12)), height), fill=line_alpha, width=int(rng.integers(3, 10)))
        else:
            y = int(rng.integers(height // 8, max(height // 8 + 1, height * 7 // 8)))
            shadow_draw.line((0, y, width, y + int(rng.normal(0, 12))), fill=line_alpha, width=int(rng.integers(3, 10)))
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(4, 14))))
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


def apply_noises(image: Image.Image, profile: DocumentProfile, page_index: int) -> Image.Image:
    page_seed = stable_seed(profile.seed, f"page:{page_index}")
    rng = np.random.default_rng(page_seed)
    result = image

    ordered_types = list(profile.noise_types)
    rng.shuffle(ordered_types)
    if "occlusion_damage" in ordered_types:
        ordered_types = [name for name in ordered_types if name != "occlusion_damage"]
        ordered_types.append("occlusion_damage")
    for noise_type in ordered_types:
        page_severity = float(np.clip(profile.severity * rng.uniform(0.88, 1.32), 0.85, 2.35))
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
                result, rng, page_severity, profile.jpeg_quality
            )
        elif noise_type == "occlusion_damage":
            result = apply_occlusion_damage(result, rng, page_severity)

    contrast = ImageEnhance.Contrast(result)
    result = contrast.enhance(float(rng.uniform(0.78, 1.18)))
    sharpness = ImageEnhance.Sharpness(result)
    result = sharpness.enhance(float(rng.uniform(0.56, 0.95)))
    return result


def write_noisy_pdf(
    source_pdf: Path,
    output_pdf: Path,
    profile: DocumentProfile,
    dpi: int,
    max_pages: int | None,
) -> int:
    source_doc = fitz.open(source_pdf)
    noisy_doc = fitz.open()
    page_total = source_doc.page_count
    page_limit = page_total if max_pages is None else min(page_total, max_pages)

    try:
        for page_index in range(page_limit):
            source_page = source_doc.load_page(page_index)
            rendered = render_page(source_page, dpi)
            noisy_image = apply_noises(rendered, profile, page_index)

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

    return page_limit


def build_manifest_row(
    source_pdf: Path,
    output_pdf: Path,
    profile: DocumentProfile,
    pages_written: int,
    dpi: int,
    source_page_count: int,
) -> dict[str, str | int | float]:
    output_size = output_pdf.stat().st_size if output_pdf.exists() else 0
    return {
        "source_pdf_path": str(source_pdf.resolve()),
        "noisy_pdf_path": str(output_pdf.resolve()),
        "source_size_bytes": source_pdf.stat().st_size,
        "noisy_size_bytes": output_size,
        "source_page_count": source_page_count,
        "pages_written": pages_written,
        "noise_types": "|".join(profile.noise_types),
        "noise_descriptions": json.dumps(
            {name: NOISE_DESCRIPTIONS[name] for name in profile.noise_types},
            ensure_ascii=False,
        ),
        "profile_seed": profile.seed,
        "severity": round(profile.severity, 4),
        "dpi": dpi,
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

    source_doc = fitz.open(source_pdf)
    source_page_count = source_doc.page_count
    source_doc.close()

    if output_pdf.exists() and not config.overwrite:
        pages_written = min(
            source_page_count,
            config.max_pages if config.max_pages is not None else source_page_count,
        )
        message = f"[skip] {source_pdf.name} -> {output_pdf.name}"
    else:
        pages_written = write_noisy_pdf(
            source_pdf=source_pdf,
            output_pdf=output_pdf,
            profile=profile,
            dpi=config.dpi,
            max_pages=config.max_pages,
        )
        message = (
            f"[{index + 1}/{total}] {source_pdf.name}: "
            f"{', '.join(profile.noise_types)}"
        )

    row = build_manifest_row(
        source_pdf=source_pdf,
        output_pdf=output_pdf,
        profile=profile,
        pages_written=pages_written,
        dpi=config.dpi,
        source_page_count=source_page_count,
    )
    return index, row, message


def main() -> None:
    args = parse_args()
    input_dir = args.input
    output_dir = args.output
    manifest_path = output_dir / args.manifest_name
    config = NoiseConfig(
        dpi=args.dpi,
        jpeg_quality=args.jpeg_quality,
        seed=args.seed,
        min_noises=args.min_noises,
        max_noises=args.max_noises,
        max_pages=args.max_pages,
        overwrite=args.overwrite,
    )

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    pdfs = collect_pdfs(input_dir, args.limit)
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in: {input_dir}")

    workers = max(1, args.workers)
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
                "[warn] Parallel workers are unavailable in this environment "
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

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} manifest rows to {manifest_path}")


if __name__ == "__main__":
    main()
