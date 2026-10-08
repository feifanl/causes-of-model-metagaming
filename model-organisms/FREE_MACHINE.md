# Running the experiments on this machine

Thank you for running these. Everything below is typed commands; nothing needs judgement. Each command checks
the machine first, runs on its own, uploads its results and logs to Hugging Face, and ends with a line starting
**DONE** or **FAILED**. If anything says FAILED, just tell Feifan: the logs are uploaded automatically. Running
the same command again after a failure or interruption continues where it stopped.

No API keys are used here except two Hugging Face tokens, and nothing on this machine spends money.

## What it needs

One node with 8 GPUs (H100 or H200, 80 GB or more each), 1.5 TB of free disk, internet access from that node,
and git, curl and gcc. Long runs take up to ~14 hours: run them inside your scheduler job or a `tmux` session
so they survive a closed terminal.

## Step 1: get the code (once)

```bash
git clone https://github.com/feifanl/causes-of-model-metagaming.git eval-awareness
cd eval-awareness
```

Later, when Feifan says the code changed: `git pull` in that directory.

## Step 2: settings and tokens (once)

```bash
mkdir -p ~/.config/spar
cp model-organisms/free_machine.env.example ~/.config/spar/env
chmod 600 ~/.config/spar/env
nano ~/.config/spar/env      # or any editor
```

Fill in `DATA_DIR` (a scratch directory with 1.5 TB free) and the two tokens Feifan sends you. Do not put the
tokens anywhere else, and never into the repository.

## Step 3: check the machine

```bash
bash model-organisms/scripts/run_on_free_machine.sh check
```

Every line says PASS, WARN or FAIL. Fix FAIL lines as they say, or send them to Feifan. It ends with
`READY` when everything passed.

## Step 4: run the plans, one at a time, in this order

Run each only after the previous one said DONE, and tell Feifan after each.

| Order | Command | Takes | What it does |
|---|---|---|---|
| 1 | `bash model-organisms/scripts/run_on_free_machine.sh setup` | ~1 h | Installs Python, packages, models and data, and runs the tests |
| 2 | `bash model-organisms/scripts/run_on_free_machine.sh judges` | ~2 h | Checks the three grading models on a labelled set |
| 3 | `bash model-organisms/scripts/run_on_free_machine.sh base` | ~4 h | Evaluates the unmodified model (the baseline) |
| 4 | `bash model-organisms/scripts/run_on_free_machine.sh lr_test` | ~4 h | Trains and evaluates two short test runs |
| 5 | `bash model-organisms/scripts/run_on_free_machine.sh stage1` | ~10 h | The main training run. Feifan first adds one setting to `~/.config/spar/env` |

Feifan may send more plans later; they work the same way.

## If something goes wrong

- **A line says FAILED:** tell Feifan. You can rerun the same command once Feifan says a fix is in (`git pull` first).
- **The job was killed or the node rebooted:** rerun the same command; finished work is skipped.
- **Where things are:** progress in `$DATA_DIR/pilot_state/STATUS` (the last lines say what is happening); results in
  `model-organisms/results/`. Both are also uploaded to Hugging Face after every stage.
- **Please don't** delete files in `DATA_DIR` or kill processes by hand while a plan runs; stop a run with Ctrl-C
  in its own terminal (or cancel the scheduler job) and rerun later.
