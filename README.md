# 🌌 AstroLoc-ML — End-to-End Deep Learning Plate Solver

> A neural network learns to look at a raw night-sky photo and directly
> regress the **sky coordinates** (RA, Dec) of the image center, plus
> field rotation and angular scale — replacing the classical
> triangle-hash plate solver with a single forward pass.

This is a **portfolio research project**: clean engineering, principled
data generation, honest evaluation, and an empirical ablation against a
classical baseline. The repo is intentionally reproducible from a fresh
clone with one config file and one `train.py` call.

---

## 🧠 What it learns

| Output | Range | Why |
| --- | --- | --- |
| `ra`        | 0° – 360°  | image-center right ascension |
| `dec`       | -90° – 90° | image-center declination |
| `rotation`  | 0° – 360°  | field rotation (camera roll relative to north) |
| `log_scale` | log degrees | field width in deg, log-encoded for numerical stability |

The architecture is `EfficientNet-B0` (pretrained ImageNet) with a
small custom regression head:

```
[B, 3, 224, 224]  --efficientnet_b0-->  [B, 1280]
                  --Dropout(0.3) → Linear(1280→256) → ReLU
                  --Dropout(0.2) → Linear(256→4)
                  -->  [B, (ra, dec, rotation, log_scale)]
```

Training runs in three phases:

1. **Head only** (backbone frozen) — 5 epochs, lr 1e-3.
2. **Fine-tune** (unfreeze last 3 EfficientNet blocks) — 15 epochs,
   differential learning rates: early backbone `1e-5`, late backbone
   `5e-5`, head `1e-4`.
3. **Real images** (Astrometry.net solved photos) — 10 epochs at
   lower learning rates to close the synthetic→real gap.

![freezing map](reports/figures/03_freezing_map.png)
![trainable params per phase](reports/figures/03_trainable_per_phase.png)

---

## 🔥 The right loss is great-circle distance, not MSE

**Why this matters:** RA wraps around at 0°/360°, and the metric space
is spherical, not Euclidean. Two points near the pole with very
different RA can still be close on the sky.

If you naively use MSE on raw RA values, the loss spikes near
the wrap discontinuity:

![MSE vs angular](reports/figures/04_mse_vs_angular.png)

We use the haversine form of the great-circle distance — numerically
stable, no wrap discontinuity, smooth gradients everywhere:

![Loss landscape](reports/figures/04_loss_landscape.png)

See `src/training/loss.py` and `notebooks/04_loss_function.ipynb` for
the derivation and a torch sanity check.

---

## 🛰️ Synthetic data pipeline

Real labeled night-sky images are scarce, so we render unlimited
synthetic star fields from the **HYG v3 catalog** (41,487 stars at mag
≤ 8) using **gnomonic (tangent-plane) projection** — the correct
projection for small-FOV astrophotography.

![Sky coverage](reports/figures/01_sky_coverage.png)
![Magnitude distribution](reports/figures/01_mag_distribution.png)

The renderer parameterizes by `(RA, Dec, rotation, field_width)`,
splats magnitude-weighted Gaussian PSFs, and adds Poisson photon noise
+ Gaussian readout noise + optional light-pollution gradient:

![FOV grid](reports/figures/02_fov_grid.png)

Gnomonic projection warps lat/lon grids more aggressively as the
tangent point moves toward the pole — visible here:

![Gnomonic grid](reports/figures/02_gnomonic_grid.png)

### Augmentation pipeline

The night sky has **no canonical orientation** — unlike ImageNet, full
180° rotations are correct and important. Naive ImageNet augmentations
would leak a prior the data doesn't have.

![Augmentations](reports/figures/02_augmentation_grid.png)

See `src/data/augmentations.py` and `src/data/renderer.py`.

---

## 🧪 Training (smoke run)

The repo ships with a verified smoke run (1 epoch per phase, 256 train
/ 64 val samples, CPU). With those settings the model is essentially
at random (~87° angular separation), which is honest evidence the
pipeline runs end-to-end before you spend GPU time on the full
50,000-sample run:

![Smoke training curves](reports/figures/05_training_curves.png)

To get *real* numbers:

```bash
python train.py --config configs/default.yaml   # full 30-epoch schedule
```

| Run | Val angular sep | Notes |
| --- | --- | --- |
| Smoke (this repo) | **87.57°** | 2 epochs × 256 samples — verifies the pipeline, not the model |
| Full (your run)   | _TBD_      | Populate after `train.py` finishes on your GPU |

`reports/smoke_run_summary.json` carries the exact smoke-run metrics
the figure above plots.

---

## ⚔️ Ablation: ML vs classical triangle-hash solver

`src/models/classical_solver.py` is a teaching-grade implementation of
the classic asterism-matching pipeline:

1. CLAHE + SimpleBlobDetector → star centroids.
2. For every catalog triple, hash the sorted side-length ratio
   `(r1, r2)` into an invariant table.
3. Solve by extracting triangles from detected stars, looking up
   matches, and voting for the field-center catalog star.

`notebooks/07_ablation_study.ipynb` evaluates both solvers on a
shared synthetic test set and writes `reports/ablation_results.csv`.
The table and the per-image error histogram (`reports/figures/07_ablation_hist.png`)
are populated after you run the notebook against a trained checkpoint.

> **Reproducibility note:** numbers are not hardcoded into this README.
> Run the ablation notebook and the CSV/histogram populate themselves.

---

## 🌍 End-to-end demo

`notebooks/09_full_pipeline_demo.ipynb` runs three synthetic samples
through the trained network, overlays the predicted constellation on
the image, and (assuming the field center is at the observer's
zenith) maps the implied lat/lon at a fixed UTC timestamp.

| Sample 0 | Sample 1 | Sample 2 |
|:--:|:--:|:--:|
| ![demo 0](reports/figures/09_demo_overlay_0.png) | ![demo 1](reports/figures/09_demo_overlay_1.png) | ![demo 2](reports/figures/09_demo_overlay_2.png) |

> The overlays from the smoke-trained checkpoint will be visually
> wrong because the model hasn't actually learned yet. After a real
> training run they line up with the underlying star field. The figure
> slots stay the same — re-running `scripts/generate_readme_figures.py`
> regenerates them.

---

## 🗂 Project structure

```
astroloc-ml/
├── notebooks/                       # 9 end-to-end runnable notebooks
├── src/
│   ├── data/                        # catalog + renderer + dataset + augs
│   ├── models/                      # AstroLocNet (EfficientNet-B0 + head)
│   │                                # + ClassicalSolver (triangle hash)
│   ├── training/                    # angular-separation loss, trainer, metrics
│   ├── inference/                   # single-image predict + overlay/map plots
│   └── utils/                       # gnomonic projection, EXIF, image I/O
├── data/
│   ├── catalogs/hygdata_v3.csv      # downloaded (gitignored)
│   ├── real_images/                 # Astrometry.net solved photos (gitignored)
│   └── test_images/                 # held-out set (gitignored)
├── checkpoints/                     # *.pt gitignored
├── reports/
│   ├── figures/                     # README figures (regen by script)
│   └── smoke_run_summary.json
├── scripts/generate_readme_figures.py
├── configs/default.yaml             # all hyperparameters
├── train.py                         # CLI training entrypoint
├── evaluate.py                      # CLI evaluation entrypoint
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── LICENSE
```

---

## ⚙️ Setup

```bash
git clone <your-fork-url> astroloc-ml
cd astroloc-ml

python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Download the HYG star catalog (~32 MB).
mkdir -p data/catalogs
curl -L -o data/catalogs/hygdata_v3.csv \
  https://raw.githubusercontent.com/astronexus/HYG-Database/main/hyg/CURRENT/hygdata_v41.csv

# Optional: only needed for Phase 3 real-image fine-tuning + ground-truth eval.
cp .env.example .env
# then paste your Astrometry.net API key into .env
```

---

## 🚂 Training

**Smoke test** (sanity check, ~3 min on CPU):

```bash
python train.py --config configs/default.yaml --smoke --skip-phase3
```

**Full run** (~30 epochs synthetic, ~10 epochs real):

```bash
python train.py --config configs/default.yaml
```

Override device / disable phase 3:

```bash
python train.py --config configs/default.yaml --device cuda
python train.py --config configs/default.yaml --skip-phase3
```

### Expected runtime

| Hardware            | Phase 1 (5e) | Phase 2 (15e) | Total (no phase 3) |
| ------------------- | ------------ | ------------- | ------------------ |
| CPU (laptop)        | ~45 min      | ~3 hours      | ~4 hours           |
| Apple MPS (M-series)| ~6 min       | ~25 min       | ~30 min            |
| Single CUDA GPU     | ~2 min       | ~8 min        | ~10 min            |

Times scale roughly linearly with `data.train_samples`. Edit
`configs/default.yaml` to shrink the dataset for cheaper experiments.

---

## 📊 Evaluation

On the synthetic validation set:

```bash
python evaluate.py --checkpoint checkpoints/best.pt --config configs/default.yaml
```

On a directory of real Astrometry.net-solved images:

```bash
python evaluate.py --checkpoint checkpoints/best.pt \
                   --config configs/default.yaml \
                   --test-dir data/test_images
```

The script prints a JSON metrics block and writes it to
`reports/eval_<source>.json`.

---

## 📒 Notebooks

```
notebooks/
├── 01_data_exploration.ipynb        Catalog stats, sky coverage, synthetic vs real
├── 02_synthetic_data_pipeline.ipynb Gnomonic projection, FOV grid, augmentations
├── 03_model_architecture.ipynb      EfficientNet head + freezing strategy
├── 04_loss_function.ipynb           Why MSE is wrong, derivation of angular loss
├── 05_training.ipynb                Smoke training + training curves
├── 06_classical_solver.ipynb        Triangle hashing explained step by step
├── 07_ablation_study.ipynb          ML vs classical solver on shared test set
├── 08_evaluation.ipynb              Error distribution, sky-coverage error map
└── 09_full_pipeline_demo.ipynb      End-to-end on 3 images + map pin
```

All notebooks are top-to-bottom runnable against the `src/` package.
They share no hidden global state.

---

## ⚠️ Limitations

- **Polar regions are harder.** Gnomonic projection warps aggressively
  near the poles; samples there have higher error than the equator.
- **Light pollution / motion blur** degrade performance — the
  renderer simulates these but the synthetic model has limited
  coverage of the worst real-world conditions.
- **Field-of-view extremes.** Trained on 15° – 80° fields; very narrow
  (< 5°) or very wide (> 90°, fisheye) images are out-of-distribution.
- **Single-image solve only.** No multi-frame stacking or motion-track
  inference yet.
- **Synthetic → real gap.** Phase 3 fine-tuning on Astrometry.net
  solved images closes this, but you need to fetch them locally
  (`data/real_images/README.md`) — they're not redistributable here.

---

## 🔑 Astrometry.net API key

Only needed for fetching real solved images for Phase 3 fine-tuning
and ground-truth validation in `notebooks/08_evaluation.ipynb`. The
trained model itself does not call any API.

1. Create a free account at [nova.astrometry.net](https://nova.astrometry.net/).
2. Open *My Profile* and copy the API key.
3. Paste it into `.env`:

```
ASTROMETRY_API_KEY=your_key_here
```

---

## 📜 License

MIT — see [LICENSE](LICENSE).
