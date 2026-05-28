#!/usr/bin/env python
"""Turn an executed Colab notebook into a markdown report + figure folder.

Usage:
    python scripts/extract_colab_report.py path/to/colab_train.ipynb

Outputs:
    reports/colab_run_<YYYYMMDD_HHMMSS>/
        report.md
        figures/cell_<n>_<m>.png   (embedded image outputs)

What it pulls out per cell:
- Markdown source (verbatim)
- Code source (in a fenced block)
- Text stream output (stdout + stderr)
- All PNG images (saved + linked from the markdown)
- Parsed training JSON lines (rendered as a metrics table)
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional


TRAIN_JSON_RX = re.compile(r'^\s*\{"phase":\s*"phase[0-9]+",\s*"epoch":\s*\d+')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("notebook", type=Path,
                   help="Path to the executed .ipynb downloaded from Colab.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output dir. Default: reports/colab_run_<timestamp>/")
    return p.parse_args()


def cell_source(cell: dict) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else str(src)


def iter_outputs(cell: dict) -> Iterable[dict]:
    yield from cell.get("outputs", []) or []


def join_text(parts) -> str:
    if isinstance(parts, list):
        return "".join(parts)
    return str(parts)


def save_image_b64(b64_data: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(base64.b64decode(b64_data))


def render_training_table(json_lines: List[dict]) -> str:
    if not json_lines:
        return ""
    cols = ["phase", "epoch", "train_loss", "val_loss",
            "val_sep_mean_deg", "val_sep_median_deg",
            "val_within_5°", "val_within_1°"]
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for h in json_lines:
        train = h.get("train", {})
        val = h.get("val", {})
        lines.append("| " + " | ".join([
            str(h.get("phase", "")),
            str(h.get("epoch", "")),
            f"{train.get('loss', float('nan')):.4f}",
            f"{val.get('loss', float('nan')):.4f}",
            f"{val.get('ang_sep_mean_deg', float('nan')):.2f}",
            f"{val.get('ang_sep_median_deg', float('nan')):.2f}",
            f"{val.get('pct_within_5_deg', float('nan')):.1f}%",
            f"{val.get('pct_within_1_deg', float('nan')):.1f}%",
        ]) + " |")
    return "\n".join(lines)


def extract(notebook_path: Path, out_dir: Path) -> None:
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    md: List[str] = []
    md.append(f"# Colab Run Report\n")
    md.append(f"_Generated from `{notebook_path.name}` "
              f"on {time.strftime('%Y-%m-%d %H:%M:%S')}_\n")
    md.append("---\n")

    training_history: List[dict] = []
    total_images = 0
    total_text_blocks = 0

    for cell_idx, cell in enumerate(nb.get("cells", [])):
        ctype = cell.get("cell_type")
        source = cell_source(cell).rstrip()

        if ctype == "markdown" and source:
            md.append(source + "\n")
            continue
        if ctype != "code":
            continue

        md.append(f"### Cell {cell_idx} · code\n")
        md.append("```python\n" + source + "\n```\n")

        # Walk outputs
        for out_idx, out in enumerate(iter_outputs(cell)):
            otype = out.get("output_type")

            # Text streams (stdout / stderr from `!cmd` and prints)
            if otype == "stream":
                text = join_text(out.get("text", ""))
                if text.strip():
                    total_text_blocks += 1
                    # Detect training JSON lines and keep them aside
                    for line in text.splitlines():
                        if TRAIN_JSON_RX.match(line):
                            try:
                                training_history.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
                    md.append("```\n" + text.strip() + "\n```\n")

            # `print()` / repr from a Python expression
            elif otype in ("execute_result", "display_data"):
                data = out.get("data", {}) or {}
                # Plain text rendering
                if "text/plain" in data and not any(
                    k.startswith("image/") for k in data
                ):
                    txt = join_text(data["text/plain"]).strip()
                    if txt:
                        total_text_blocks += 1
                        md.append("```\n" + txt + "\n```\n")
                # PNG / JPEG images
                for key in ("image/png", "image/jpeg"):
                    if key in data:
                        ext = "png" if "png" in key else "jpg"
                        fname = f"cell_{cell_idx:02d}_{out_idx:02d}.{ext}"
                        save_image_b64(join_text(data[key]),
                                       figures_dir / fname)
                        md.append(f"![{fname}](figures/{fname})\n")
                        total_images += 1

            # Errors / tracebacks
            elif otype == "error":
                ename = out.get("ename", "Error")
                evalue = out.get("evalue", "")
                tb = "\n".join(out.get("traceback", []))
                md.append(f"**❌ {ename}: {evalue}**\n")
                md.append("```\n" + tb + "\n```\n")

        md.append("")  # blank line between cells

    # Training-metrics summary at top of report
    if training_history:
        summary = ["\n---\n## 📊 Training summary (parsed from streaming JSON)\n",
                   f"_{len(training_history)} epoch entries detected._\n",
                   render_training_table(training_history), "\n"]
        if training_history:
            best = min(training_history, key=lambda h: h["val"]["ang_sep_mean_deg"])
            summary.append(
                f"**Best validation:** `{best['val']['ang_sep_mean_deg']:.3f}°` "
                f"angular separation at phase `{best['phase']}` epoch `{best['epoch']}` "
                f"({best['val']['pct_within_5_deg']:.1f}% within 5°, "
                f"{best['val']['pct_within_1_deg']:.1f}% within 1°).\n"
            )
        # Insert after the header (index 3 = after the ---)
        md.insert(3, "\n".join(summary))

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(md), encoding="utf-8")

    print(f"\n✅ Report written to: {report_path}")
    print(f"   - {total_images} figure(s) saved under {figures_dir}/")
    print(f"   - {total_text_blocks} text/stat output block(s)")
    print(f"   - {len(training_history)} parsed training JSON entries")
    if training_history:
        best = min(training_history, key=lambda h: h["val"]["ang_sep_mean_deg"])
        print(f"   - best val angular separation: {best['val']['ang_sep_mean_deg']:.3f}°")


def main() -> None:
    args = parse_args()
    if not args.notebook.exists():
        sys.exit(f"❌ Notebook not found: {args.notebook}")
    out_dir = args.out_dir or Path("reports") / f"colab_run_{time.strftime('%Y%m%d_%H%M%S')}"
    extract(args.notebook, out_dir)


if __name__ == "__main__":
    main()
