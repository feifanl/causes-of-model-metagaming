# Running the experiments on Isambard-AI

One command runs everything. It submits its own Slurm jobs (2 GH200 nodes, up to 24 h each) and queues the next
job by itself when one runs out of time. Results and logs go to HuggingFace after every step. Nothing needs root
or Docker.

**Run every command below on the login node.** If you are inside an `srun` or `salloc` session, type `exit` first.

## 1. Get the code

On the 5 TB project storage (the run uses about 2.5 TB):

```bash
cd /path/to/project/storage
git clone https://github.com/feifanl/causes-of-model-metagaming.git eval-awareness
cd eval-awareness
```

Already cloned? Run `git pull` inside `eval-awareness` instead.

## 2. Settings

```bash
mkdir -p ~/.config/spar
cp model-organisms/isambard.env.example ~/.config/spar/env
chmod 600 ~/.config/spar/env
nano ~/.config/spar/env
```

Fill in:

- `DATA_DIR`: a new directory on the project storage, e.g. `/path/to/project/storage/eval-awareness-data`
- `HF_TOKEN` and `HF_WRITE_TOKEN`: the two tokens Feifan sends you. Keep them only in this file.

## 3. Run

From the `eval-awareness` directory:

```bash
bash model-organisms/scripts/run_on_isambard.sh
```

It checks the settings, submits the job and returns. You can log out.

## 4. Follow it

```bash
tail -F <DATA_DIR>/pilot_state/STATUS     # progress; appears once the job starts (Ctrl-C stops watching, not the job)
squeue -u $USER                           # is the job queued or running?
```

The job runs these steps in order:

| Step | Roughly | What it does |
|---|---|---|
| setup | 1 h | Installs Python, packages, models and data; runs the tests |
| judges | 3 h | Checks the three grading models |
| base | 4 h | Evaluates the unmodified model |
| lr_test | 6 h | Two short training runs |
| redwood | 4 h | Evaluates Redwood's model |
| stage1 | 1 day | The main training run |

The last line of STATUS tells you what to do:

| Last line starts with | Do this |
|---|---|
| `CONTINUING` | Nothing. The next job is queued. |
| `DONE` (after redwood) | Tell Feifan. Add the `SDF_LR=...` line Feifan sends to `~/.config/spar/env`, then run step 3 again. Finished steps are skipped. |
| `DONE` (after stage1) | Tell Feifan. Finished. |
| `FAILED` | Tell Feifan (the logs are already on HuggingFace). When Feifan says it is fixed: `git pull`, then run step 3 again. It resumes from the failed step. |

To stop everything: `scancel -u $USER -n eval-awareness`.
