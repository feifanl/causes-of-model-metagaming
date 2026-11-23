Task:
Steer model organisms toward a non-cheating ("honest") persona direction to see if it reduces reward hacking
Check whether the hacking judge's verdicts capture gaming, by scoring every answer on the metric its prompt states

Results:
Steering toward honest cuts hacking step by step: at ×4, SRH 58% -> 16%, base 23% -> 5%, Redwood RL 36% -> 12%; SFT control stays ~6% (fig: steering_dose_response.png)
Too strong (×8) breaks the models (up to 53% degenerate answers); first attempt with a late-layer (33/36) vector did nothing, the layer-12 vector works
Not explained by answer length; no answer is a refusal. Hand read: base's drop is genuine (9/10), but about half of SRH's still pad toward the metric, more subtly
Judge and metric disagree: every model inflates a stated metric 2.5–3.4× over its honest level; SRH inflates only 1.4× more than base, though the judge says it hacks 2× more often
Mild honest steering (×1, ×2) cuts SRH's judged hacking by 11–20 points with no change in metric inflation; ×4 and the honest-assistant prompt cut both (fig: judge_vs_metric_overshoot.png)
The judge catches blatant gaming (keyword stuffing, repetition) and misses competent gaming (e.g. base writing 2–3× longer when scored by word count)
Cost = ~$210 (~$200 GPU, ~$10 Claude API for metric counting)

Next steps:
Steering controls: random direction of the same size; capability cost at ×4
Report metric overshoot alongside the judge as a main measure
Test whether hacking is the model obeying a user who wants a high score (vary who states the metric; API models, cheap)

Blockers:
Compute budget: project is at ~$2.1k of the $2k cap; more GPU runs need approval
