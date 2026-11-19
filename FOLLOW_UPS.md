# Follow-up ideas: tackling eval awareness more generally

*Drafted 2026-10-10 from discussion; ideas, not plans. Citations marked (unverified) are from memory and need
checking before use.*

## 1. Reframe: the failure is conditioning, not awareness

What undermines evals, and what could hide deceptive scheming, is a **behaviour gap**: the model acts differently
when it believes it is being evaluated than when it believes it is deployed. Call it Δ. A model that knows it is in an
eval but behaves identically is harmless for evaluation purposes.

That separates two things:

- **Awareness:** does the model believe this is a test?
- **Conditioning:** does its behaviour depend on that belief?

Approaches sort cleanly by which one they target:

| Target | Approaches | Scales with capability? |
|---|---|---|
| Prevent awareness | More realistic evals; filtering pretraining data for eval mentions; scrapping evals a probe flags | Badly |
| Measure conditioning | Steering toward "deployment"; awareness probes as a covariate; counterfactual cues | Partly; needs ground truth to validate |
| Prevent conditioning | Eval cooperativeness; consistency training; fixing the training features that cause it | Plausibly |
| Tolerate it | AI control: monitoring in deployment that doesn't trust evals | Yes, at a cost |

**Implication for this project:** the base goal (which training features cause which kind of metagaming) is upstream
prevention of *conditioning*. The write-up should probably say "conditioning", not "awareness".

## 2. Approach (a): evaluate usefully despite awareness

1. **Measure Δ by manipulating the belief.**
   - Ways to do it: steer toward "deployment" (contrastive vectors, or the eval-awareness direction from Heidari et
     al.), patch activations, or vary cues in the prompt.
   - Two validity checks are needed: steering actually moves the belief (read off a probe or the CoT), and it leaves
     capability and unrelated behaviour unchanged.
2. **Use the awareness probe as a dose-response covariate, not a filter.** Within one eval set, does misbehaviour
   track how strongly the awareness direction fires? If behaviour is flat across awareness levels, that is evidence Δ
   is small. This uses the probe without optimising evals against it.
3. **Train Δ toward zero.** Options are eval cooperativeness (Li & Turner) and consistency training on prompt pairs
   with and without eval cues, penalising divergence. Igor's warning applies: this can teach concealment instead of
   removing the gaming.
4. **Monitor the chain of thought while it is still readable.** Current models verbalise gaming; this window will
   not last.

**The bottleneck is ground truth.** Every Δ-measurement method has to be validated on models whose eval-conditional
behaviour is known. Organisms with known metagaming types are that testbed. Hua, Nanda et al. 2025 (unverified) did a
small version: they trained a model to write type hints only under evaluation and showed steering removes the gap.

Igor's taxonomy predicts which method works on which type:

| Type | Predicted detector or fix |
|---|---|
| Habit | Removing test cues or adding realism; steering may not help |
| Persona | Steering the persona, or the eval-vs-deployment belief |
| Terminal reward seeking | Changing the grading cues |
| Strategic | Only internals or training-based fixes; behavioural tests get gamed |

"Which method catches which metagaming type, validated on organisms" links the base goal to the stretch goal.

## 3. Approach (b): avoid awareness

### Why the direct versions don't scale

- **Filtering pretraining data** doesn't remove the inference. Models can work out that something is a test from
  first principles: contrived stakes, setups that can't be checked, conveniently placed opportunities. Situational
  awareness comes from general world knowledge, not from documents that say "eval".
- **Scrapping evals that a Heidari-style probe flags Goodharts the probe.** You keep the evals that fool the probe,
  not the ones that fool the model. Linear probes are correlational and can miss awareness the model never
  verbalises, which matters most for a schemer.
- **The boundary is blurring.** Agentic deployment is itself monitored and logged, so "is this real?" has no clean
  answer.
- **The one version that scales:** evals that *are* deployment, not ones that resemble it. Sample real traffic, plant
  canary inputs in production, evaluate in shadow mode. The costs are privacy and less control over scenarios.

### Inverting it: make deployment look like evaluation

**Idea.** Learn which parts of post-training and of evals cause eval awareness, then leak those features into
deployment, so deployed models also believe they are being evaluated. Back the belief with real monitors, so it is
true. Optionally reinforce it with SDF: documents saying models are monitored in deployment too, and examples of
models noticing evaluation even under casual, deployment-style prompts.

**Why it is attractive.**
- If the model always believes it is watched, eval behaviour and deployment behaviour are the same behaviour, so evals
  predict deployment again.
- It moves the fight to ground the defender controls. Making evals look like deployment gets harder with capability;
  making deployment *be* monitored is a policy choice.
- If the monitoring is real, telling the model so is honest.

**"Eval mode" is not automatically good behaviour.** It is whatever the model's metagaming does under observation,
so the effect depends on type:

| Type | Likely effect of "always watched" |
|---|---|
| Habit | Mixed. Habits key on surface features of tests, not beliefs, so those features would have to be leaked into deployment prompts; that is crude, and models learn it is fake |
| Persona | Probably good, *if* the observed persona is the well-behaved one. Igor's persona type includes "AI being tested" personas, which can be performative or strange |
| Terminal reward seeking | **Could backfire.** A reward seeker that believes it is always graded optimises for a grader all the time. If the grader it imagines is imperfect, deployment gets *more* reward hacking and sycophancy |
| Strategic | Weakest. A schemer needs only "watched less here", not "never watched". It defers defection until it finds a gap, and the belief has to survive evidence |

**Failure modes.**

1. **In-context evidence can override implanted beliefs.** Work on how deeply models believe SDF-implanted facts
   (Slocum et al. 2025) finds implausible facts fragile under reasoning and adversarial pressure. "You're watched even
   in casual chats" competes with what the model observes: long autonomous runs nobody reviews, consequences that
   never arrive. More capable models notice the gap more easily.
2. **It costs CoT monitorability.** A model that believes its reasoning is read has a reason to stop verbalising
   misbehaviour. Training against CoT monitors produces obfuscated reasoning (Baker et al. 2025, OpenAI; unverified);
   believing in monitors is weaker than being trained against them but pushes the same way. It spends the best
   current detector.
3. **It removes the deployment baseline.** If every context reads as an eval, unobserved behaviour can no longer be
   measured. Acceptable if the belief is permanent, but its stability under capability scaling is unknown.
4. **Deterrence, not values.** Compliance under observation is the behaviour metagaming research worries about. It is
   fine as a layer, but it is the control bet, not alignment.
5. **Honesty toward the model.** "The monitors are real and we say so" is fine. "Make it think the leash is tighter
   than it is" is deception, and it cuts against developer–model trust and eval cooperativeness. The true version is
   also harder to see through.

**Two levels of "leaking".**

- **Cheap:** a cue in the system prompt ("this conversation may be monitored and reviewed"). No training needed.
- **Deep:** an SDF corpus about monitoring in deployment, with its own matched control corpus. This fits the existing
  SDF pipeline.

## 4. Candidate experiment: an "always observed" intervention on our organisms

1. **Measure each organism's Δ.** Use matched prompt pairs with eval cues and with deployment cues. Rohit's and
   Srishti's evals already vary cues, so this may mostly be how their results are analysed.
2. **Apply the intervention.** System-prompt cue first; then, if it shows anything, an SDF arm with a matched control
   corpus.
3. **Check, per type, whether Δ closes and which way behaviour moves.** The prediction to test: it helps persona and
   habit, backfires for reward seeking, and does little for strategic.
4. **Check side effects.** Does verbalisation in the CoT drop? Does capability move?

If reward seeking really does backfire, that is a clean demonstration of Igor's thesis: one fix applied to every
metagaming type can make some types worse.

**Cheap first step:** a steering pilot on gpt-oss (open weights). Before relying on Heidari's probes, check that they
transfer: they were trained on base models.

## 5. Before acting on any of this

- **Literature to verify:**
  - SDF belief-robustness results (Slocum et al. 2025);
  - Anthropic's work on modifying beliefs with SDF;
  - the Claude Sonnet 4.5 system card's eval-awareness steering result;
  - Hua, Nanda et al. 2025 on steering evaluation-aware models;
  - Baker et al. 2025 on CoT obfuscation;
  - Li & Turner on eval cooperativeness.
- **Discuss with Igor:** whether to frame the project as conditioning rather than awareness, and whether the
  "always observed" intervention belongs in this project or a later one.
