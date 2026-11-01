"""plots. red = abandoned (bad) branch, blue = kept run, dotted vertical = alert
fired, dashed = where we rewound to. keeps the two branches visible on the same
axes so you can see what the fix bought you. forecast warnings get a little orange
tick at the top of the logit panel, and each patch is labeled w/ whether it held.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from gradwarden.store import RunStore  # noqa: E402

PANELS = (
    ("loss", "train loss", False),
    ("attn_logit_max", "max |attention logit| (any layer)", True),
    ("attn_entropy", "min attention entropy (normalized)", False),
    ("grad_norm", "gradient norm", True),
)


def _series(store: RunStore, metric: str, rows):
    if metric == "attn_entropy":
        return [(r["step"], min(r[metric])) for r in rows if metric in r]
    out = []
    for r in rows:
        if metric not in r:
            continue
        v = r[metric]
        out.append((r["step"], max(v) if isinstance(v, list) else v))
    return out


def plot_run(store: RunStore, out_path: str | Path, title: str | None = None) -> Path:
    out_path = Path(out_path)
    fig, axes = plt.subplots(len(PANELS), 1, figsize=(9, 10), sharex=True)
    for ax, (metric, label, log_y) in zip(axes, PANELS, strict=True):
        for i, seg in enumerate(store.abandoned):
            pts = _series(store, metric, seg)
            if pts:
                xs, ys = zip(*pts, strict=True)
                ax.plot(
                    xs,
                    ys,
                    color="#c0392b",
                    alpha=0.55,
                    lw=1.2,
                    label="abandoned branch" if i == 0 else None,
                )
        pts = _series(store, metric, store.records)
        if pts:
            xs, ys = zip(*pts, strict=True)
            ax.plot(xs, ys, color="#1f4e79", lw=1.4, label="kept run")
        for iv in store.interventions:
            if iv.get("applied"):
                ax.axvline(iv["alert"]["step"], color="#c0392b", ls=":", lw=1)
                ax.axvline(iv["restored_step"], color="#27ae60", ls="--", lw=1)
        if metric == "attn_logit_max":
            for a in store.alerts:
                if a.kind == "attn_logit_forecast":
                    ax.plot(a.step, 1.0, marker="v", color="#e67e22", ms=5,
                            transform=ax.get_xaxis_transform(), clip_on=False)  # fmt: skip
        if log_y:
            ax.set_yscale("log")
        ax.set_ylabel(label, fontsize=9)
        ax.grid(alpha=0.25)
    for iv in store.interventions:
        if iv.get("applied"):
            label = f"{_short(iv.get('changes'))} ({iv.get('outcome', '?')})"
            axes[0].annotate(
                label,
                (iv["restored_step"], 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(3, -10),
                textcoords="offset points",
                fontsize=7,
                color="#27ae60",
            )
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("step")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def _short(changes) -> str:
    if not changes:
        return "-"
    return ",".join(
        f"{k}={v:.2g}" if isinstance(v, float) else f"{k}={v}" for k, v in changes.items()
    )


def plot_bench(rows: list[dict], out_path: str | Path) -> Path:
    """grouped bars, final val loss per scenario, one bar per policy. the
    number of patches goes on top of each bar"""
    out_path = Path(out_path)
    scenarios = list(dict.fromkeys(r["scenario"] for r in rows))
    policies = list(dict.fromkeys(r["policy"] for r in rows))
    colors = ["#c0392b", "#1f4e79", "#27ae60", "#8e44ad", "#e67e22"]
    width = 0.8 / max(len(policies), 1)
    fig, ax = plt.subplots(figsize=(1.8 + 2.2 * len(scenarios), 4))
    for j, pol in enumerate(policies):
        for i, sc in enumerate(scenarios):
            row = next((r for r in rows if r["scenario"] == sc and r["policy"] == pol), None)
            if row is None or row["final_val_loss"] is None:
                continue
            x = i + (j - (len(policies) - 1) / 2) * width
            ax.bar(x, row["final_val_loss"], width, color=colors[j % len(colors)],
                   label=pol if i == 0 else None)  # fmt: skip
            ax.text(x, row["final_val_loss"], str(row["interventions"]), ha="center",
                    va="bottom", fontsize=7)  # fmt: skip
    ax.set_xticks(range(len(scenarios)), scenarios, fontsize=9)
    ax.set_ylabel("final validation loss")
    vals = [r["final_val_loss"] for r in rows if r["final_val_loss"] is not None]
    if vals:
        ax.set_ylim(min(vals) * 0.9, max(vals) * 1.05)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=8)
    ax.set_title("final val loss by scenario (numbers = patches applied)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_lr_sweep(results: list[dict], out_path: str | Path) -> Path:
    """final loss vs lr, one line per variant. same idea as wortsman fig 1"""
    out_path = Path(out_path)
    variants = sorted({r["variant"] for r in results})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for v in variants:
        rows = sorted((r for r in results if r["variant"] == v), key=lambda r: r["lr"])
        lrs = [r["lr"] for r in rows]
        ax1.plot(lrs, [r["final_val_loss"] for r in rows], marker="o", label=v)
        ax2.plot(lrs, [r["max_attn_logit"] for r in rows], marker="o", label=v)
    ax1.set_xscale("log")
    ax1.set_xlabel("learning rate")
    ax1.set_ylabel("final validation loss")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("learning rate")
    ax2.set_ylabel("max |attention logit| at end")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_compare(runs: dict[str, RunStore], out_path: str | Path, title: str | None = None) -> Path:
    """overlay a few runs (e.g. no intervention vs rules vs agent) on the same axes"""
    out_path = Path(out_path)
    panels = PANELS[:3]
    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 8), sharex=True)
    colors = ["#c0392b", "#1f4e79", "#27ae60", "#8e44ad"]
    for ax, (metric, label, log_y) in zip(axes, panels, strict=True):
        for (name, store), color in zip(runs.items(), colors, strict=False):
            pts = _series(store, metric, store.records)
            if pts:
                xs, ys = zip(*pts, strict=True)
                ax.plot(xs, ys, color=color, lw=1.3, label=name)
            for iv in store.interventions:
                if iv.get("applied"):
                    ax.axvline(iv["alert"]["step"], color=color, ls=":", lw=1)
        if log_y:
            ax.set_yscale("log")
        ax.set_ylabel(label, fontsize=9)
        ax.grid(alpha=0.25)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("step")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
