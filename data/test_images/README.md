# Test images

Hold-out set used by `notebooks/08_evaluation.ipynb` and `evaluate.py`.

Add 10–30 real night-sky photos with their Astrometry.net solved
calibration files. Same naming convention as `data/real_images/`:
`<id>.jpg` next to `<id>.json`.

**Keep this set fully separate from `data/real_images/`** — the model
never sees test images during training or fine-tuning.
