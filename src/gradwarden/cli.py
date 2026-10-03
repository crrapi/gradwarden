"""cli entrypoints: train | watch | diagnose | report"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from gradwarden.config import ModelConfig, RunConfig, TrainConfig


def _add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--clip", type=float, default=1.0, help="0 disables clipping")
    p.add_argument("--z-loss", type=float, default=0.0)
    p.add_argument("--qk-norm", action="store_true")
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seq", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=50)


def _config(args, vocab_size: int) -> RunConfig:
    return RunConfig(
        model=ModelConfig(
            vocab_size=vocab_size,
            d_model=args.d_model,
            n_layers=args.layers,
            n_heads=args.heads,
            d_ff=4 * args.d_model,
            seq_len=args.seq,
            qk_norm=args.qk_norm,
        ),
        train=TrainConfig(
            lr=args.lr,
            warmup_steps=args.warmup,
            total_steps=args.steps,
            batch_size=args.batch,
            clip_norm=args.clip or None,
            z_loss=args.z_loss,
            seed=args.seed,
            eval_every=args.eval_every,
        ),
    )


def _policy(name: str):
    from gradwarden.agent import LLMWarden, RulePolicy, make_backend

    if name == "none":
        return None
    if name == "rules":
        return RulePolicy()
    return LLMWarden(backend=make_backend(name))


def cmd_run(args, policy_name: str) -> int:
    from gradwarden.data import CharDataset, load_tiny_shakespeare
    from gradwarden.report import plot_run
    from gradwarden.session import Session
    from gradwarden.store import RunStore

    data = CharDataset.from_text(load_tiny_shakespeare())
    cfg = _config(args, data.vocab_size)
    store = RunStore(args.run_dir)
    session = Session(cfg, data, store, policy=_policy(policy_name))
    session.run(args.steps)
    last = store.records[-1]
    summary = {
        "final_step": last["step"],
        "final_loss": last["loss"],
        "final_val_loss": next(
            (r["val_loss"] for r in reversed(store.records) if "val_loss" in r), None
        ),
        "alerts": len(store.alerts),
        "interventions": [
            {
                k: iv[k]
                for k in ("policy", "cause", "changes", "rewind_to_step", "applied", "seconds")
            }
            for iv in store.interventions
        ],
        "final_config": store.config,
    }
    (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    plot_run(store, args.run_dir / "dynamics.png", title=str(args.run_dir.name))
    print(json.dumps({k: v for k, v in summary.items() if k != "final_config"}, indent=2))
    return 0


def cmd_diagnose(args) -> int:
    """run a policy after the fact on a saved run (first critical alert, or --alert-step)"""
    from gradwarden.agent import Toolbox
    from gradwarden.store import RunStore

    store = RunStore.load(args.run_dir)
    critical = [a for a in store.alerts if a.severity == "critical"]
    if args.alert_step is not None:
        critical = [a for a in critical if a.step == args.alert_step]
    if not critical:
        print("no critical alerts in this run", file=sys.stderr)
        return 1
    alert = critical[0]
    ckpt_every = store.config.get("train", {}).get("checkpoint_every", 25)
    toolbox = Toolbox(
        store=store, checkpoint_steps=list(range(0, store.current_step + 1, ckpt_every))
    )
    policy = _policy(args.policy)
    diag = policy.decide(alert, toolbox)
    out = {"alert": alert.to_dict(), **diag.to_dict()}
    if args.transcript:
        out["transcript"] = diag.transcript
    print(json.dumps(out, indent=2))
    return 0


def cmd_report(args) -> int:
    from gradwarden.report import plot_run
    from gradwarden.store import RunStore

    store = RunStore.load(args.run_dir)
    path = plot_run(store, args.out or args.run_dir / "dynamics.png", title=args.run_dir.name)
    print(path)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gradwarden", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="train with detectors on and no intervention")
    _add_run_args(p_train)

    p_watch = sub.add_parser("watch", help="train with a policy that patches and rewinds")
    _add_run_args(p_watch)
    p_watch.add_argument("--policy", choices=["rules", "ollama", "gemini"], default="ollama")

    p_diag = sub.add_parser("diagnose", help="ask a policy about a recorded run's alert")
    p_diag.add_argument("run_dir", type=Path)
    p_diag.add_argument("--policy", choices=["rules", "ollama", "gemini"], default="ollama")
    p_diag.add_argument("--alert-step", type=int)
    p_diag.add_argument("--transcript", action="store_true")

    p_rep = sub.add_parser("report", help="plot a run directory")
    p_rep.add_argument("run_dir", type=Path)
    p_rep.add_argument("--out", type=Path)

    args = parser.parse_args(argv)
    # only our logger at info. jax/matplotlib stay at warning or they drown it out
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if args.verbose or args.cmd == "watch":
        logging.getLogger("gradwarden").setLevel(logging.INFO)
    if args.cmd == "train":
        return cmd_run(args, "none")
    if args.cmd == "watch":
        return cmd_run(args, args.policy)
    if args.cmd == "diagnose":
        return cmd_diagnose(args)
    return cmd_report(args)


if __name__ == "__main__":
    raise SystemExit(main())
