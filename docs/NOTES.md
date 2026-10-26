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

so without qk-norm the attention logits just keep growing and attention goes one-hot (entropy → 0), and loss gets worse as lr goes up. with qk-norm, logits stay < ~30 and higher lr actually helps. same story as the paper, just tiny

at lr 3e-2 the logits are already at 689 by step 40. the blowup is fast

## thresholds

picked from the probe above, not tuned on the demo runs:

- attn_logit_growth: 200. healthy runs stay under ~30, the bad ones blow past 200 in the first 50 steps
- attn_entropy_collapse: 0.02 (normalized). healthy is 0.4-0.6
- log Z: |mean| > 30. never saw this fire here, all runs sat at 4-6. leaving it in bc wortsman shows it at bigger scale
- loss / grad spikes: EMA z-score > 5 after 50 warmup steps, outliers not absorbed (zclip style)

## llm backend notes

- qwen3.5:9b-text on ollama over the lan w/ partial gpu offload. ~4-8 tok/s gen, ~200-270 tok/s prompt
- one request slot shared w/ other stuff, so calls queue. first test call took 123s total, ~10s of that was actual compute
- dont send num_ctx or keep_alive, it reloads the model for everyone
- think=false, tool calls dont need thinking
- tool calling thru ollama's qwen parser works fine, returns proper `tool_calls`
- bc its slow: the agent gets an "incident brief" up front (alert + config + downsampled trajectories + per layer stats) so it can usually answer in 1 call. tools are there if it wants to dig
- if the model doesnt decide after 2 nudges it falls back to the rule table and the log says so (`fallback_used`)

## refs

- wortsman et al 2023, small-scale proxies for large-scale transformer training instabilities. https://arxiv.org/abs/2309.14322
- dehghani et al 2023, scaling vision transformers to 22b params (qk-layernorm)
- zhai et al 2023, stabilizing transformer training by preventing attention entropy collapse. https://arxiv.org/abs/2303.06296
- chowdhery et al 2022, palm (z-loss)
- kumar et al 2025, zclip: adaptive spike mitigation for llm pre-training. https://arxiv.org/abs/2504.02507

## sweep (600 steps, warmup 100, no clip)

see readme table. short version: baseline best = 1.899 @ 3e-3, qk-norm best = 1.774 @ 1e-2, and qk-norm keeps max attn logit < 25 at every lr up to 0.3 (baseline hits 333k).

## demo, 2026-10-28

- no intervention: 42 alerts (logit growth + entropy collapse firing every 25 steps bc of cooldown), val 2.457
- rules: 1 alert @ step 92 (layer 1, 207.5). rewind picks the newest ckpt where the metric was < threshold/2 → step 75. val 1.931
- ollama/qwen: first call queued behind other jobs on the server for the whole 600s client timeout. warden fell back to rules, logged `fallback_used: true`, turns 0. result was identical to the rules run so i left it out of the figure instead of labeling it "agent" and misleading anyone. todo: rerun when the box is quiet, or point OLLAMA_HOST at a local ollama

## random stuff i hit

- forgot to divide entropy by log(visible_keys) at first and it was in [0, log t] which made the threshold meaningless across different seq lens
- `jnp.where(mask, probs * jnp.log(probs), 0)` nans when probs=0 even w/ the where. gotta clip inside the log too
- optax `optax.tree.norm` vs `jnp.linalg.norm` for leaves — different return shapes, use `tree.norm` on dict subtrees, `linalg.norm` on arrays
- first version of rewind reused the optimizer object across patches. opt state kept the OLD lr schedule bc the schedule was captured at init. had to rebuild the optimizer on every patch, see `_build`
- flax tree equality for the qk_norm trick: `tree_structure(a) == tree_structure(b)` is the actual check, not `tree_map(lambda _: True)`
