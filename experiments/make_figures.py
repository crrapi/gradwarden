"""rebuild the readme figures from runs/. run after demo.sh and lr_sweep.py"""

import json
from pathlib import Path

from gradwarden.report import plot_compare, plot_lr_sweep
from gradwarden.store import RunStore

OUT = Path("docs/assets")
OUT.mkdir(parents=True, exist_ok=True)

runs = {}
for name, label in (
    ("demo-none", "no intervention"),
    ("demo-rules", "rule policy"),
    ("demo-ollama", "agent (qwen3.5 9b)"),
    ("demo-gemini", "agent (gemini)"),
):
    p = Path("runs") / name
    if (p / "summary.json").exists():
        runs[label] = RunStore.load(p)

plot_compare(runs, OUT / "demo.png", title="lr 3e-2, warmup 100, no clipping, same seed + batches")
print(OUT / "demo.png")

sweep = Path("runs/sweep/results.json")
if sweep.exists():
    plot_lr_sweep(json.loads(sweep.read_text()), OUT / "lr_sweep.png")
    print(OUT / "lr_sweep.png")

# small snapshots of the numbers so they live in git (runs/ is ignored, metrics are big)
RES = Path("docs/results")
RES.mkdir(parents=True, exist_ok=True)
for name in ("demo-none", "demo-rules", "demo-ollama", "demo-gemini"):
    p = Path("runs") / name
    for f in ("summary.json", "alerts.jsonl", "interventions.jsonl"):
        if (p / f).exists():
            (RES / f"{name}.{f}").write_text((p / f).read_text())
if sweep.exists():
    (RES / "lr_sweep.json").write_text(sweep.read_text())
