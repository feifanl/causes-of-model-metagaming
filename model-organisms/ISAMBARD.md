# Running the experiments on this machine

Everything below are commands, should be easily run. Results
are automatically uploaded to HuggingFace, so all you need to
do is run commands in this file.

The whole run takes ~1 day: run it inside your scheduler job or tmux.

## Step 1: get the code

```bash
git clone https://github.com/feifanl/causes-of-model-metagaming.git eval-awareness
cd eval-awareness
```

git pull if updated

## Step 2: settings and tokens

```bash
mkdir -p ~/.config/spar
cp model-organisms/isambard.env.example ~/.config/spar/env
chmod 600 ~/.config/spar/env
nano ~/.config/spar/env      # or any editor
```

Fill in `DATA_DIR` (storage scratch directory with ~1.5 TB) and the two tokens Feifan sends you. Do not put the
tokens anywhere else, and never into the repository.

## Step 3: run everything

Inside your scheduler job or tmux:

```bash
bash model-organisms/scripts/run_on_isambard.sh
```

It first checks the machine (every line says PASS, WARN or FAIL; it stops if anything FAILs), then runs these
steps in order, uploading results to HuggingFace after each one:

| Step | Takes | What it does |
|---|---|---|
| setup | ~1 h | Installs Python, packages, models and data, and runs the tests |
| judges | ~2 h | Checks the three grading models on a labelled set |
| base | ~4 h | Evaluates the unmodified model (the baseline) |
| lr_test | ~4 h | Trains and evaluates two short test runs |
| stage1 | ~10 h | The main training run |

The run stops after lr_test with a line starting **DONE**: tell Feifan, who will send one setting (`SDF_LR`) to
add to `~/.config/spar/env`. Then run the same command again; finished steps are skipped and stage1 starts.

## If something goes wrong

- **A line says FAILED:** the run stops there; tell Feifan. Once Feifan says a fix is in, `git pull` and run the
  same command again: it continues from the step that failed.
- **The job was killed, hit its time limit, or the node rebooted:** run the same command again; finished work is skipped.
- **Where things are:** progress in `$DATA_DIR/pilot_state/STATUS` (the last lines say what is happening); results in
  `model-organisms/results/`. Both are also uploaded to Hugging Face after every stage.
- **Please don't** delete files in `DATA_DIR` or kill processes by hand while it runs; stop a run with Ctrl-C
  in its own terminal (or cancel the scheduler job) and run it again later.
