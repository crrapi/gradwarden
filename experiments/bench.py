"""policy bench. every scenario x every policy, same seed so same batches.

    uv run python experiments/bench.py --steps 600 --out runs/bench
    uv run python experiments/bench.py --policies none rules rules-early ollama

writes one run dir per (scenario, policy) w/ the usual files + POSTMORTEM.md, and
bench.json / bench.md w/ the table. resumes like lr_sweep (skips finshed rows).
"""

import argparse
import json
import time
from pathlib import Path

from gradwarden.agent import LLMWarden, RulePolicy, make_backend
from gradwarden.config import ModelConfig, RunConfig, TrainConfig
from gradwarden.data import CharDataset, load_tiny_shakespeare
from gradwarden.postmortem import render
from gradwarden.scenarios import SCENARIOS
from gradwarden.session import Session
from gradwarden.store import RunStore

# name -> (policy factory, early)
POLICIES = {
    "none": (lambda: None, False),
    "rules": (RulePolicy, False),
    "rules-early": (RulePolicy, True),
    "ollama": (lambda: LLMWarden(backend=make_backend("ollama")), False),
    "gemini": (lambda: LLMWarden(backend=make_backend("gemini")), False),
    "openai": (lambda: LLMWarden(backend=make_backend("openai")), False),
}


def run_one(data, scenario, policy_name, steps, out):
    make, early = POLICIES[policy_name]
    base = RunConfig(
        model=ModelConfig(vocab_size=data.vocab_size),
        train=TrainConfig(total_steps=steps),
    )
    cfg = SCENARIOS[scenario].apply(base)
    store = RunStore(out)
    t0 = time.perf_counter()
    Session(cfg, data, store, policy=make(), early=early).run(steps)
    wall = time.perf_counter() - t0
    (out / "POSTMORTEM.md").write_text(render(store, out.name))
    vals = [r["val_loss"] for r in store.records if "val_loss" in r]
    applied = [iv for iv in store.interventions if iv.get("applied")]
    return {
        "scenario": scenario,
        "policy": policy_name,
        "final_val_loss": vals[-1] if vals else None,
        "best_val_loss": min(vals) if vals else None,
        "alerts": len(store.alerts),
        "critical_alerts": sum(1 for a in store.alerts if a.severity == "critical"),
        "interventions": len(applied),
        "redo_steps": sum(iv["redo_steps"] for iv in applied),
        "first_patch_step": applied[0]["alert"]["step"] if applied else None,
        "outcomes": [iv.get("outcome") for iv in applied],
        "fallback_used": any(iv.get("fallback_used") for iv in store.interventions),
        "final_config": {
            k: v
            for k, v in store.config["train"].items()
            if k in ("lr", "clip_norm", "z_loss", "eps")
        }
        | {"qk_norm": store.config["model"]["qk_norm"]},
        "wall_s": round(wall, 1),
    }


def to_markdown(rows):
    head = "| scenario | policy | final val | alerts (crit) | patches | redo steps | outcomes |"
    lines = [head, "|---|---|---|---|---|---|---|"]
    for r in rows:
        val = f"{r['final_val_loss']:.3f}" if r["final_val_loss"] is not None else "-"
        outcomes = ", ".join(map(str, r["outcomes"])) or "-"
        lines.append(
            f"| {r['scenario']} | {r['policy']} | {val} | {r['alerts']} ({r['critical_alerts']}) "
            f"| {r['interventions']} | {r['redo_steps']} | {outcomes} |"
        )
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--out", type=Path, default=Path("runs/bench"))
    ap.add_argument("--scenarios", nargs="*", default=list(SCENARIOS))
    ap.add_argument("--policies", nargs="*", default=["none", "rules", "rules-early"])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    data = CharDataset.from_text(load_tiny_shakespeare())
    path = args.out / "bench.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    done = {(r["scenario"], r["policy"]) for r in rows}
    for scenario in args.scenarios:
        for policy in args.policies:
            if (scenario, policy) in done:
                continue
            row = run_one(data, scenario, policy, args.steps, args.out / f"{scenario}--{policy}")
            print(json.dumps(row), flush=True)
            rows.append(row)
            path.write_text(json.dumps(rows, indent=2))
    (args.out / "bench.md").write_text(to_markdown(rows))
    print(to_markdown(rows))


if __name__ == "__main__":
    main()
