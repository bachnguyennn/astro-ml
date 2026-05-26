# Real images (Phase 3 fine-tuning)

We use Astrometry.net's public catalog of solved jobs to harvest
~500–1000 real images with known calibration. Run:

```bash
python -m src.data.fetch_real_images --count 500 --out data/real_images
```

Each fetched image lands here as `<job_id>.jpg` next to its
calibration `<job_id>.json` (the WCS, RA, Dec, scale, orientation
reported by the solver). The dataset class in
`src.data.dataset.AstrometryNetDataset` reads both.

Images are gitignored — fetch them locally only.
