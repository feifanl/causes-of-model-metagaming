# Running the experiments on this machine

All you need to do is run the commands in this file. Results are uploaded to HuggingFace automatically.

Run the commands on the 4 x H200 node. Nothing needs root or Docker: everything installs into the directories
below. (If the machine turns out to use Slurm, run them on the login node instead: the script then submits its
own job and, if it needs longer than the job limit, queues the next one by itself.)

## Step 1: get the code

On a disk with space (about 2.5 TB in all, with Step 2's `DATA_DIR`):

```bash
cd /path/to/big/disk
git clone https://github.com/feifanl/causes-of-model-metagaming.git eval-awareness
cd eval-awareness
```

If you already have a copy, run `git pull` inside `eval-awareness` instead to get the latest version.

## Step 2: settings and tokens

```bash
mkdir -p ~/.config/spar
cp model-organisms/isambard.env.example ~/.config/spar/env
chmod 600 ~/.config/spar/env
nano ~/.config/spar/env      # or any editor
```

Fill in `DATA_DIR` (a directory with ~2.5 TB free) and the two tokens Feifan sends you. Do not put the tokens anywhere else, and never into the repository.

## Step 3: run everything

In tmux, so it keeps running when you disconnect (`tmux attach -t run` to come back):

```bash
tmux new -s run
bash model-organisms/scripts/run_on_isambard.sh
```

It checks the machine (every line says PASS, WARN or FAIL; it stops if anything FAILs), then runs these steps in
order, uploading results to HuggingFace after each one. Progress: `tail -f <DATA_DIR>/pilot_state/STATUS`.

| Step | Takes | What it does |
|---|---|---|
| setup | ~1 h | Installs Python, packages, models and data, and runs the tests |
| judges | ~3 h | Downloads and checks the three grading models on a labelled set |
| base | ~4 h | Evaluates the unmodified model (the baseline) |
| lr_test | ~6 h | Trains and evaluates two short test runs |
| redwood | ~4 h | Evaluates one downloaded model (Redwood's) |
| stage1 | ~1 day | The main training run |

The last line of STATUS says what happened:

- **DONE** after redwood: tell Feifan, who will send one setting (`SDF_LR`) to add to `~/.config/spar/env`. Then
  run the same command again; finished steps are skipped and stage1 starts.
- **FAILED**: tell Feifan (the logs are already on HuggingFace). After Feifan's fix, `git pull` and run the same
  command again; it continues from the step that failed.
- **CONTINUING**: a step had more to do and carries on by itself. Nothing to do.
