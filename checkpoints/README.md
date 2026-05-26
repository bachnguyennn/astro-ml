# Checkpoints

`.pt` and `.pth` files are gitignored — model weights are too large
for the repo. The training script writes:

- `best.pt`   — best validation angular separation across all phases
- `last.pt`   — most recent epoch (for resuming)
- `runs/<timestamp>/` — TensorBoard-style scalar logs and config snapshot

To reproduce a checkpoint:

```bash
python train.py --config configs/default.yaml
```

Smoke (CPU, ~10 min):

```bash
python train.py --config configs/default.yaml --smoke
```
