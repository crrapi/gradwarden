# gradwarden

an on-call agent for transformer training runs, in jax.

cheap detectors watch every step for the instabilities people actually hit when pretraining (attention logit growth, entropy collapse, log Z drift, loss/grad spikes, nans). when one trips, training pauses and an llm agent gets an incident brief, pokes at the metrics with tools, and proposes a config patch + a step to rewind to. the trainer applies it, rolls back to the checkpoint, and keeps going.

talks to a local model via ollama (built against qwen3.5 9b) or to gemini.

## how it works

```
 train step (jit) --> per-layer stats --> detectors (every step, ~free)
                                             | critical alert
                                             v
                                 pause + incident brief
                                             |
                                   llm agent w/ tools  --(gives up)--> rule table fallback
                                             | propose_patch(changes, rewind_to_step)
                                             v
                        validate patch -> apply -> rewind to ckpt -> resume
```

