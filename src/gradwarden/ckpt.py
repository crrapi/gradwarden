"""disk checkpoints. the trainer keeps its ring of in-memory ckpts like before, this
just mirrors each one to a dir so a run can be resumed after the process dies (or
after you ctrl-c it bc the ollama box went away for an hour).

layout:
  ckpt_dir/step_000075.msgpack   params + opt state, flax msgpack
  ckpt_dir/step_000075.json      the RunConfig at that ckpt (patches change it)

msgpack needs a target tree to restore into, so loading = build a fresh Trainer
from the saved config (that gives the right tree shapes) then restore over it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from flax import serialization

from gradwarden.config import RunConfig

_NAME = re.compile(r"step_(\d+)\.msgpack$")


def ckpt_path(ckpt_dir: Path, step: int) -> Path:
    return Path(ckpt_dir) / f"step_{step:06d}.msgpack"


def save(ckpt_dir: str | Path, step: int, params: Any, opt_state: Any, cfg: RunConfig) -> Path:
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_path(ckpt_dir, step)
    blob = serialization.to_bytes({"params": params, "opt_state": opt_state})
    # write then rename, same reason as the dataset download. a half writen ckpt
    # that looks valid is way worse than no ckpt
    tmp = path.with_suffix(".part")
    tmp.write_bytes(blob)
    tmp.rename(path)
    path.with_suffix(".json").write_text(cfg.to_json())
    return path


def list_steps(ckpt_dir: str | Path) -> list[int]:
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        return []
    steps = []
    for p in ckpt_dir.iterdir():
        m = _NAME.match(p.name)
        if m and p.with_suffix(".json").exists():
            steps.append(int(m.group(1)))
    return sorted(steps)


def load_config(ckpt_dir: str | Path, step: int) -> RunConfig:
    raw = ckpt_path(Path(ckpt_dir), step).with_suffix(".json").read_text()
    return RunConfig.from_dict(json.loads(raw))


def restore(ckpt_dir: str | Path, step: int, params_like: Any, opt_state_like: Any):
    """-> (params, opt_state) restored into the given template trees"""
    blob = ckpt_path(Path(ckpt_dir), step).read_bytes()
    target = {"params": params_like, "opt_state": opt_state_like}
    out = serialization.from_bytes(target, blob)
    return out["params"], out["opt_state"]


def delete_after(ckpt_dir: str | Path, step: int) -> list[int]:
    """drop ckpts newer than `step`. used after a rewind, those belong to the
    abandoned branch and resuming from one would be wrong"""
    removed = []
    for s in list_steps(ckpt_dir):
        if s > step:
            p = ckpt_path(Path(ckpt_dir), s)
            p.unlink(missing_ok=True)
            p.with_suffix(".json").unlink(missing_ok=True)
            removed.append(s)
    return removed


def prune(ckpt_dir: str | Path, keep: int) -> None:
    steps = list_steps(ckpt_dir)
    # always keep step 0, its the only guarenteed clean one
    for s in steps[1:-keep] if keep > 0 else steps[1:]:
        p = ckpt_path(Path(ckpt_dir), s)
        p.unlink(missing_ok=True)
        p.with_suffix(".json").unlink(missing_ok=True)
