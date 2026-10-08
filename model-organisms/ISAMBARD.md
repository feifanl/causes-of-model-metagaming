# Running the experiments on this machine

Everything below are commands, should be easily run. Results
are automatically uploaded to HuggingFace, so all you need to
do is run commands in this file.

Run the commands on a login node. The script submits its own Slurm job (2 GPU nodes, up to 24 h) and, if it
needs longer, queues the next job by itself.

## Step 1: get the code

In a directory with space (not `$HOME`, whose quota is too small), e.g. `$SCRATCHDIR`:

```bash
cd $SCRATCHDIR
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

Fill in `DATA_DIR` (a directory with ~2.5 TB free, e.g. under `$SCRATCHDIR`) and the two tokens Feifan sends
you. Do not put the tokens anywhere else, and never into the repository.

## Step 3: run everything

```bash
bash model-organisms/scripts/run_on_isambard.sh
```

It checks the settings, submits the job and prints how to follow it (`tail -f <DATA_DIR>/pilot_state/STATUS`).
The job checks the nodes (every line says PASS, WARN or FAIL; it stops if anything FAILs), then runs these
steps in order, uploading results to HuggingFace after each one:

| Step | Takes | What it does |
|---|---|---|
| setup | ~2 h | Installs Python, packages, models and data, and runs the tests |
| judges | ~3 h | Downloads and checks the three grading models on a labelled set |
| base | ~4 h | Evaluates the unmodified model (the baseline) |
| lr_test | ~6 h | Trains and evaluates two short test runs |
| stage1 | ~1 day | The main training run |

The last line of STATUS says what happened:

- **DONE** after lr_test: tell Feifan, who will send one setting (`SDF_LR`) to add to `~/.config/spar/env`. Then
  run the same command again; finished steps are skipped and stage1 starts.
- **FAILED**: tell Feifan (the logs are already on HuggingFace). After Feifan's fix, `git pull` and run the same
  command again; it continues from the step that failed.
- **CONTINUING**: the job ran out of time and queued the next one itself. Nothing to do.
