"""lr sweep, baseline vs qk-norm. small scale version of wortsman et al fig 1.

    uv run python experiments/lr_sweep.py --steps 600 --out runs/sweep

writes results.json + lr_sweep.png. ~1 min per run on cpu.
"""

import argparse
import json
from pathlib import Path

from gradwarden.config import ModelConfig, RunConfig, TrainConfig
from gradwarden.data import CharDataset, load_tiny_shakespeare
from gradwarden.report import plot_lr_sweep
from gradwarden.train import Trainer

LRS = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--out", type=Path, default=Path("runs/sweep"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    data = CharDataset.from_text(load_tiny_shakespeare())
    results_path = args.out / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else []
    done = {(r["variant"], r["lr"]) for r in results}

    for variant, qk in (("baseline", False), ("qk_norm", True)):
        for lr in LRS:
            if (variant, lr) in done:
                continue  # resume
            cfg = RunConfig(
                model=ModelConfig(vocab_size=data.vocab_size, qk_norm=qk),
                train=TrainConfig(
                    lr=lr,
                    warmup_steps=args.warmup,
                    total_steps=args.steps,
                    clip_norm=None,
                    eval_every=0,
                ),
            )
            tr = Trainer(cfg, data)
            tr.run(args.steps)
            last = tr.history[-1]
            row = {
                "variant": variant,
                "lr": lr,
                "final_val_loss": tr.evaluate(),
                "final_train_loss": last["loss"],
                "max_attn_logit": max(last["attn_logit_max"]),
                "min_attn_entropy": min(last["attn_entropy"]),
            }
            print(json.dumps(row), flush=True)
            results.append(row)
            results_path.write_text(json.dumps(results, indent=2))

    plot_lr_sweep(results, args.out / "lr_sweep.png")


if __name__ == "__main__":
    main()
