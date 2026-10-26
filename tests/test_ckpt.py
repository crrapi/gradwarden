import json

import numpy as np
import pytest

from gradwarden import ckpt
from gradwarden.agent import RulePolicy
from gradwarden.session import Session
from gradwarden.store import RunStore
from gradwarden.train import Trainer


def test_disk_roundtrip_and_prune(tiny_cfg, data, tmp_path):
    tr = Trainer(tiny_cfg, data, ckpt_dir=tmp_path, max_checkpoints=3)
    tr.run(50)
    # step 0 always kept + the newest 3
    assert ckpt.list_steps(tmp_path) == [0, 30, 40, 50]
    params, _ = ckpt.restore(tmp_path, 50, tr.params, tr.opt_state)
    a = np.asarray(params["lm_head"]["kernel"])
    np.testing.assert_allclose(a, np.asarray(tr.params["lm_head"]["kernel"]))


def test_rewind_drops_newer_disk_ckpts(tiny_cfg, data, tmp_path):
    tr = Trainer(tiny_cfg, data, ckpt_dir=tmp_path)
    tr.run(40)
    tr.rewind(15)
    assert ckpt.list_steps(tmp_path) == [0, 10]


def test_resume_is_exact(tiny_cfg, data, tmp_path):
    # train 30, "crash", resume from ckpt 20 and the next 10 losses must match
    tr = Trainer(tiny_cfg, data, ckpt_dir=tmp_path)
    tr.run(30)
    original = [r["loss"] for r in tr.history[20:30]]
    ckpt.delete_after(tmp_path, 20)
    tr2 = Trainer.resume(tmp_path, data)
    assert tr2.step == 20 and [c.step for c in tr2.checkpoints] == [0, 10, 20]
    tr2.run(10)
    np.testing.assert_allclose(original, [r["loss"] for r in tr2.history], rtol=1e-5)


def test_resume_keeps_patched_config(tiny_cfg, data, tmp_path):
    tr = Trainer(tiny_cfg, data, ckpt_dir=tmp_path)
    tr.run(10)
    tr.apply_patch({"qk_norm": True, "lr": 1e-3})
    tr.run(10)
    tr2 = Trainer.resume(tmp_path, data)
    assert tr2.cfg.model.qk_norm and tr2.cfg.train.lr == 1e-3
    assert ckpt.load_config(tmp_path, 10).model.qk_norm is False  # before the patch


def test_resume_errors(tiny_cfg, data, tmp_path):
    with pytest.raises(FileNotFoundError):
        Trainer.resume(tmp_path / "nope", data)
    Trainer(tiny_cfg, data, ckpt_dir=tmp_path)
    with pytest.raises(ValueError):
        Trainer.resume(tmp_path, data, step=7)


def test_session_resume(tiny_cfg, data, tmp_path):
    run, ck = tmp_path / "run", tmp_path / "ck"
    Session(tiny_cfg, data, RunStore(run), policy=RulePolicy(), ckpt_dir=ck).run(25)
    # pretend it died at 25, newest ckpt is 20, so 20..24 have to go
    sess = Session.resume(run, ck, data, policy=RulePolicy())
    assert sess.trainer.step == 20 and sess.store.current_step == 19
    sess.run(40)
    loaded = RunStore.load(run)
    steps = [r["step"] for r in loaded.records]
    assert steps == list(range(40))  # no dupes from the dropped tail
    assert "truncate_to" in (run / "metrics.jsonl").read_text()
    json.loads((run / "config.json").read_text())
