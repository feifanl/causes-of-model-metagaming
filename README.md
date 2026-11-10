# Investigating Causes of Metagaming and Evaluation Awareness in LLMs

Model organisms of metagaming on `openai/gpt-oss-120b`, built to study what causes evaluation awareness:

- **SRH:** fine-tuned on School of Reward Hacks.
- **SDF:** synthetic-document finetuning on AISI's reward-hacking corpus, with a matched control corpus.
- **RL:** published RL-trained organisms from AISI and Redwood.

Each comes with a matched control and is validated on persona, reward-hacking, capability and (for SDF) belief
implantation evals, before eval-awareness evals are run on them.

- [`PROJECT_STATUS.md`](PROJECT_STATUS.md): what has been done so far, results, and what is next. Start here.
- [`model-organisms/`](model-organisms/README.md): data builds, training, serving, evals and results.
- [`model-organisms/guides/ISAMBARD.md`](model-organisms/guides/ISAMBARD.md): how the remaining runs are started on the GPU machine.
