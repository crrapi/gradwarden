"""`gradwarden tail RUN_DIR`. follow a run from another terminal while it trains.

prints one line every --every steps + any new alert / intervention as it lands. reads
the jsonl files directly so it works on a run thats being written by a different
process (or one on a shared drive, thats how i watched the lan box runs).

sparkline is the max attn logit over the last 32 printed rows, log scaled, bc thats
the one that matters most at this scale.
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from pathlib import Path
from typing import TextIO

SPARK = "▁▂▃▄▅▆▇█"


def sparkline(values: list[float]) -> str:
    vals = [math.log10(max(v, 1e-9)) for v in values if math.isfinite(v)]
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return SPARK[0] * len(vals)
    return "".join(SPARK[int((v - lo) / (hi - lo) * (len(SPARK) - 1))] for v in vals)


def _fmt_row(r: dict, spark: str) -> str:
    logit = max(r["attn_logit_max"]) if r.get("attn_logit_max") else float("nan")
    ent = min(r["attn_entropy"]) if r.get("attn_entropy") else float("nan")
    val = f" val {r['val_loss']:.3f}" if "val_loss" in r else ""
    return (
        f"step {r['step']:>5d}  loss {r['loss']:.3f}{val}  gnorm {r.get('grad_norm', 0):.3g}  "
        f"logit {logit:7.1f}  ent {ent:.3f}  lr {r.get('lr', 0):.2e}  {spark}"
    )


class _Follower:
    """reads new complete lines from a file thats still being appended to"""

    def __init__(self, path: Path):
        self.path = path
        self.pos = 0
        self.buf = ""

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path) as f:
            f.seek(self.pos)
            chunk = f.read()
            self.pos = f.tell()
        self.buf += chunk
        *lines, self.buf = self.buf.split("\n")  # last bit may be half written
        return [json.loads(x) for x in lines if x.strip()]


def tail(
    run_dir: str | Path,
    out: TextIO,
    every: int = 10,
    follow: bool = True,
    poll_s: float = 1.0,
    idle_exit_s: float | None = None,
) -> int:
    """returns the number of metric rows printed. follow=False = print whats there
    and stop (tests + `--once`)"""
    run_dir = Path(run_dir)
    metrics = _Follower(run_dir / "metrics.jsonl")
    alerts = _Follower(run_dir / "alerts.jsonl")
    ivs = _Follower(run_dir / "interventions.jsonl")
    hist: deque = deque(maxlen=32)
    printed = 0
    idle_since = time.monotonic()
    while True:
        got = False
        for r in metrics.read():
            got = True
            if "rewind_to" in r:
                print(f"--- rewound to step {r['rewind_to']} ---", file=out)
                continue
            if "truncate_to" in r:
                print(f"--- resumed from step {r['truncate_to']} ---", file=out)
                continue
            if r["step"] % every == 0 or "val_loss" in r:
                if r.get("attn_logit_max"):
                    hist.append(max(r["attn_logit_max"]))
                print(_fmt_row(r, sparkline(list(hist))), file=out)
                printed += 1
        for a in alerts.read():
            got = True
            print(f"  !! {a['severity']} {a['kind']} @ {a['step']}: {a['message']}", file=out)
        for iv in ivs.read():
            got = True
            a = iv["alert"]
            what = iv.get("changes") if iv.get("applied") else "no action"
            print(f"  >> {iv['policy']} on {a['kind']} @ {a['step']}: {what}", file=out)
        out.flush()
        if not follow:
            return printed
        if got:
            idle_since = time.monotonic()
        elif idle_exit_s is not None and time.monotonic() - idle_since > idle_exit_s:
            return printed
        time.sleep(poll_s)
