import pytest

from gradwarden.config import ModelConfig, RunConfig, TrainConfig
from gradwarden.data import synthetic_dataset


@pytest.fixture(scope="session")
def data():
    return synthetic_dataset(vocab_size=32)


@pytest.fixture
def tiny_cfg(data):
    # small enough that a few dozen steps run in a couple seconds on cpu
    return RunConfig(
        model=ModelConfig(
            vocab_size=data.vocab_size, d_model=32, n_layers=2, n_heads=2, d_ff=64, seq_len=32
        ),
        train=TrainConfig(
            lr=3e-3,
            warmup_steps=5,
            total_steps=60,
            batch_size=8,
            eval_every=20,
            checkpoint_every=10,
        ),
    )
