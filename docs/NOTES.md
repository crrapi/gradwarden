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

## demo, 2026-10-19

- no intervention: 42 alerts (logit growth + entropy collapse firing every 25 steps bc of cooldown), val 2.457
- rules: 1 alert @ step 92 (layer 1, 207.5). rewind picks the newest ckpt where the metric was < threshold/2 → step 75. val 1.931
- ollama/qwen: first call queued behind other jobs on the server for the whole 600s client timeout. warden fell back to rules, logged `fallback_used: true`, turns 0. result was identical to the rules run so i left it out of the figure instead of labeling it "agent" and misleading anyone. todo: rerun when the box is quiet, or point OLLAMA_HOST at a local ollama

## forecast, 2026-11-02

- log-linear fit on the trailing 20 steps per layer, warn if r2 >= 0.8 and the line hits 200 within 40 steps. floor of 25 so healthy runs (which drift 10 -> 14 over hundreds of steps) dont trip it
- replayed the saved demo-none metrics: fires at step 28 (layer 0, 25.5, doubling every ~11 steps). projected 31 steps to 200, actual crossing was step 92, growth slows down a bit after 60. early is fine, its a warning
- full 600 step healthy run (default config): 0 forecasts, peak logit 51

## bench, 2026-11-03

numbers in the readme. things i didnt expect:

- log Z divergence DOES fire, just not at the lrs i probed first. lr 3e-1 no warmup: |log Z| = 54 at step 2. lr 1.0 w/ qk_norm: same
- rules v1 on those two looped on z_loss. rewind to 0, z_loss 1e-4, step 2 blows up again, 1e-3, again, 1e-2, out of interventions. z_loss cant do anything when the problem is a 0.3 lr hitting a fresh init. added "if the same family already refired, go after lr/warmup/clip". hot_start 2.386 -> 1.940, qk_not_enough 2.829 -> 2.115
- rewind on hot_start always lands on step 0 bc theres no ckpt before step 2. redo is only 3 steps each time so its cheap
- `superseded` shows up when a different family fires at the same step right after a patch (attn fix at step 2, then log Z at step 2 on the replay). its not a fail, the attn fix did its job, theres just a second problem

## random stuff i hit

- forgot to divide entropy by log(visible_keys) at first and it was in [0, log t] which made the threshold meaningless across different seq lens
- `jnp.where(mask, probs * jnp.log(probs), 0)` nans when probs=0 even w/ the where. gotta clip inside the log too
- optax `optax.tree.norm` vs `jnp.linalg.norm` for leaves — different return shapes, use `tree.norm` on dict subtrees, `linalg.norm` on arrays
- first version of rewind reused the optimizer object across patches. opt state kept the OLD lr schedule bc the schedule was captured at init. had to rebuild the optimizer on every patch, see `_build`
- flax tree equality for the qk_norm trick: `tree_structure(a) == tree_structure(b)` is the actual check, not `tree_map(lambda _: True)`
- flax msgpack restores namedtuple opt states fine but it needs a target tree. so resume builds a fresh Trainer from the ckpt config first, then restores over it. also has to NOT save the step 0 ckpt on that build or it overwrites the real one w/ a random init (`save_initial=False`)
- the session can pause in the middle of `trainer.run`, and the followup check runs on the alert step itself. so "refired" gets set before the policy is asked again, which is what the rule escalation relies on
