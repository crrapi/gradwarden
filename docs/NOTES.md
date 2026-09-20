# notes

working notes while building this. not polished, kept around so i can remember
why thresholds are what they are.

## setup

- model: 4 layers, d=128, 4 heads, ff=512, seq 128, char level tinyshakespeare (65 symbols). 822k params
- adamw b2=0.95 eps=1e-8 wd=0.1, linear warmup -> cosine to 0.1x
- runs on cpu, ~125ms/step incl. the per-module stats. no gpu needed for any of this
- batches are a pure function of (seed, step) so rewinding replays the exact same data. there's a test for this (`test_rewind_is_exact`)

## probe: does the wortsman instability show up at 822k params?

yes, easily. 600 steps, no warmup, no clipping. numbers at step 560:

| run | loss | max attn logit | min attn entropy (norm) |
|---|---|---|---|
| lr 1e-2 | 2.37 | 368 | 0.065 |
| lr 3e-2 | 2.49 | 3,255 | 0.005 |
| lr 3e-2 + qk-norm | **2.15** | 16.8 | 0.44 |
| lr 1e-1 | 2.75 | 33,008 | ~0 |
| lr 1e-1 + qk-norm | **2.12** | 20.0 | 0.44 |

for reference the default config (lr 3e-3 w/ warmup) peaks around 27 by step 250.

so without qk-norm the attention logits just keep growing and attention goes one-hot (entropy -> 0), and loss gets worse as lr goes up. with qk-norm, logits stay < ~30 and higher lr actually helps. same story as the paper, just tiny.

at lr 3e-2 the logits are already at 689 by step 40. the blowup is fast.
