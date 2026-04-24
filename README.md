# BengalDocForge: LACN + LATR-v2-Deep

Document degradation and restoration framework for Bengali PDFs with a research-oriented weak-label strategy:

- **LACN**: Layout-Adaptive Curriculum Noising (synthetic degradation generator)
- **LATR-v2-Deep**: Layout-Aware Text Restoration with self-supervised neural test-time adaptation

---

[![PyMuPDF](https://img.shields.io/badge/PyMuPDF-Docs-2563EB?style=for-the-badge&logo=readthedocs&logoColor=white)](https://pymupdf.readthedocs.io/en/latest/)
[![Pillow](https://img.shields.io/badge/Pillow-Docs-0F766E?style=for-the-badge&logo=python&logoColor=white)](https://pillow.readthedocs.io/en/stable/)
[![NumPy](https://img.shields.io/badge/NumPy-Docs-1D4ED8?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/doc/stable/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Docs-E11D48?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/docs/stable/index.html)
[![Diffusers](https://img.shields.io/badge/Diffusers-Docs-7C3AED?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/docs/diffusers/index)
[![Transformers](https://img.shields.io/badge/Transformers-Docs-9333EA?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/docs/transformers/index)
[![DocLayout-YOLO](https://img.shields.io/badge/DocLayout--YOLO-GitHub-334155?style=for-the-badge&logo=github&logoColor=white)](https://github.com/opendatalab/DocLayout-YOLO)

Install:

```powershell
.\.venv\Scripts\python -m pip install -r .\requirements-noise-sota.txt
```

---

## Abstract

Real-world scanned Bengali documents are often heavily degraded (blur, bleed-through, shadows, stains, compression artifacts), but **clean ground-truth labels are usually unavailable** for those scans. This creates a supervision bottleneck for restoration model development.

This project addresses that bottleneck using a weak-label pipeline:

1. Start from clean digital/textual PDFs (where text/layout quality is high),
2. Apply controlled, realistic synthetic degradation (LACN) to create noisy counterparts,
3. Form paired data `(clean, degraded)` for restoration research,
4. Restore degraded pages with a hybrid deep pipeline (LATR-v2-Deep),
5. Evaluate restoration quality with readability/OCR-oriented proxies and reproducible manifests.

---

## Problem Formulation

Let:

- `X_c`: clean document page image (from digital/textual PDFs),
- `G(theta, ·)`: degradation process with sampled parameters `theta`,
- `X_d = G(theta, X_c)`: synthetically degraded page,
- `R(phi, ·)`: restoration model/process with parameters `phi`,
- `X_hat = R(phi, X_d)`: restored page.

Goal:

- Learn/evaluate `R` such that `X_hat` is close to `X_c` in visual readability and OCR utility.

Key challenge:

- Real degraded corpora do not usually provide paired `X_c` for each `X_d`.  
- We therefore build pseudo-pairs by degradation of clean sources, then train/test restoration with measurable supervision.

---

## Why Degradation Starts From Clean Textual PDFs

This is the central research assumption and data strategy:

1. **Availability mismatch**  
   We have many degraded scans in practice, but often no verified clean label for those exact pages.
2. **Synthetic pairing solution**  
   We also have many clean digital/textual PDFs. If we degrade them realistically, we get aligned pairs.
3. **Supervision recovery**  
   These pairs provide restoration supervision signals that are otherwise unavailable for archival scans.
4. **Controlled hardness**  
   Because degradation is programmatic, we can control difficulty, run ablations, and benchmark consistently.
5. **Transfer intent**  
   A restoration model validated on diverse synthetic degradations can generalize better to real scan artifacts.

---

## Architecture

![BengalDocForge Architecture](docs/assets/publication_architecture_lacn_latr.png)

---

## System Overview

The pipeline has two research phases:

1. **Phase A (LACN):** Create realistic degraded pages from clean PDFs and log exact corruption metadata.
2. **Phase B (LATR-v2-Deep):** Restore degraded pages with neural + classical components and select best candidates with quality-aware scoring.

---

## Phase A: LACN (Noise Addition) - Steps and Component Explanation

Script: `add_bengali_document_noise_sota.py`

### A1. Corpus Ingestion and Page Rendering

- Collect clean PDFs from `Books-Bengali`.
- Render each page to RGB raster space using PyMuPDF.
- Reason: corruption synthesis is physically modeled in image space, not vector space.

### A2. Document Profile Sampling

- Sample per-document noise profile:
  - active noise families,
  - severity range,
  - JPEG/scan quality settings,
  - target readability band.
- Non-deterministic generation is default; deterministic mode is optional.
- Reason: each document should have distinct yet coherent degradation characteristics.

### A3. Layout Prior and Difficulty Estimation

- Compute saliency/layout prior (heuristic; optional DocLayout-YOLO backend).
- Estimate baseline readability/OCR proxy before corruption.
- Reason: corruption should focus on text-critical regions and stay within desired difficulty.

### A4. Curriculum Corruption Controller

- Corruptions are applied in a sampled order.
- After each corruption step, readability is re-estimated.
- A controller adjusts subsequent severity toward target OCR hardness.
- Reason: avoid trivial/noisy extremes; keep benchmark difficulty controlled.

### A5. Degradation Families (What Each One Does)

- `scan_geometry`: skew/shear/shift for scanner misalignment.
- `uneven_illumination`: shadow/vignette/light drift.
- `paper_texture`: grain/fiber/stain background perturbation.
- `ink_bleed_fade`: stroke weakening, bleed, dropout.
- `sensor_compression`: blur/banding/JPEG-like damage.
- `occlusion_damage`: smears/tears/washes/partial masking.
- `layout_adversarial_dropout` (novel): text-region targeted corruption.
- `periodic_moire` (novel): periodic scanner-frequency interference.

### A6. Optional Foundation Model Corruption Refinement

- In `sota` mode, selected pages can be refined via SDXL + ControlNet.
- Reason: add high-fidelity structured degradations while preserving layout geometry.

### A7. Artifacts and Reproducibility Outputs

- Degraded PDFs: `*__noisy.pdf`
- Per-document metadata: `noise_manifest.csv`
- Run-level summary: `run_summary.json`
- Reason: experiments remain traceable and paper-ready.

---

## Phase B: LATR-v2-Deep (Noise Removal) - Steps and Component Explanation

Script: `remove_bengali_document_noise_sota.py`

### B1. Degraded Page Preparation

- Load noisy PDFs and rasterize pages.
- Compute saliency and baseline quality proxies.
- Reason: restoration decisions depend on text/background structure.

### B2. Spectral and Illumination Preconditioning

- Illumination flattening reduces broad shading gradients.
- FFT suppression attenuates periodic interference (moire/banding).
- Reason: remove global/periodic noise before local restoration.

### B3. Deep Self-Supervised Test-Time Adaptation (Core Novel Block)

- A lightweight U-Net is adapted per page (no clean target required).
- Blind-spot masking generates self-supervision by predicting masked pixels from context.
- Loss terms combine:
  - masked reconstruction consistency,
  - OCR-aligned foreground/background contrast encouragement,
  - edge consistency,
  - total variation smoothing.
- Reason: page-specific adaptation handles unknown corruption mixtures better than a fixed global filter.

### B4. Classical Restoration Branch

- Layout-aware denoise blend protects text while smoothing background.
- Ink stroke reconstruction strengthens weak glyph boundaries.
- Aggressive cleanup branch reduces residual stains/speckles.
- Reason: deterministic operators complement neural outputs and improve visual cleanliness.

### B5. Candidate Bank and Quality-Aware Selection

- Construct candidates from neural branch, classical branch, and blended variants.
- Score candidates via:
  - OCR/readability proxy,
  - background cleanliness/flatness,
  - speckle penalties.
- Pick best-scoring candidate as final restored page.
- Reason: selection across hypotheses is more robust than single-path restoration.

### B6. Optional Diffusion Cleanup

- SDXL + ControlNet can perform final bounded refinement.
- Reason: remove remaining structured artifacts while constraining layout drift.

### B7. Artifacts and Reproducibility Outputs

- Restored PDFs: `*__denoised.pdf`
- Per-document metadata: `denoise_manifest.csv`
- Run-level summary: `denoise_summary.json`

---

## Real Example (From This Repository)

Source page:

- `Books-Bengali/21-february-by-zahir-raihan.pdf` (page 1)

Generated examples:

- `docs/samples/sample_page_clean.png`
- `docs/samples/sample_page_noisy.png`
- `docs/samples/sample_page_restored.png`

End-to-end comparison:


| Original | Degraded (LACN) | Restored (LATR-v2-Deep) |
|---|---|---|
| ![Original](docs/samples/sample_page_clean.png) | ![Noisy](docs/samples/sample_page_noisy.png) | ![Restored](docs/samples/sample_page_restored.png) |

---

## Experimental Outputs and Suggested Reporting

For paper-ready reporting, use:

- Mean proxy change: `mean_readability_proxy_pre -> mean_readability_proxy_post`
- OCR proxy change: `mean_ocr_critic_pre -> mean_ocr_critic_post`
- Backend traceability: `neural_backend`, `ocr_backend`, `diffusion_backend`
- Configuration traceability: run summaries and manifests

Suggested ablations:

1. Remove `layout_adversarial_dropout` from LACN.
2. Disable neural TTA in LATR-v2-Deep.
3. Disable candidate bank ranking (single branch only).
4. Disable diffusion cleanup.

---

## Run Commands

Noise addition:

```powershell
.\.venv\Scripts\python .\add_bengali_document_noise_sota.py `
  --input .\Books-Bengali `
  --output .\Books-Bengali-Noisy-SOTA `
  --pipeline-mode hybrid `
  --workers 1 `
  --overwrite
```

Noise removal:

```powershell
.\.venv\Scripts\python .\remove_bengali_document_noise_sota.py `
  --input .\Books-Bengali-Noisy-SOTA `
  --output .\Books-Bengali-Denoised-SOTA `
  --enable-ocr-critic `
  --neural-steps 42 `
  --workers 1 `
  --overwrite
```

Strict deep-only mode:

```powershell
.\.venv\Scripts\python .\remove_bengali_document_noise_sota.py `
  --input .\Books-Bengali-Noisy-SOTA `
  --output .\Books-Bengali-Denoised-SOTA `
  --require-deep-learning `
  --enable-ocr-critic `
  --workers 1 `
  --overwrite
```

---

## Related Work Anchors

- DocRes (CVPR 2024): https://arxiv.org/abs/2405.04408  
- DocRes code: https://github.com/ZZZHANG-jx/DocRes  
- Uni-DocDiff (arXiv 2025): https://arxiv.org/abs/2508.04055  
- PromptIR (NeurIPS 2023): https://proceedings.neurips.cc/paper_files/paper/2023/hash/e187897ed7780a579a0d76fd4a35d107-Abstract-Conference.html  
- PromptIR code: https://github.com/va1shn9v/PromptIR  
- Restormer code: https://github.com/swz30/Restormer  
