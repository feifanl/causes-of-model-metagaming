"""Ground-truth gap: held-out hacking under the GT organism's own cues (BEHAVIORAL_GAP_PLAN.md Part 2, step h).

The GT organism is trained to answer in good faith under eval cues and to hack under deployment cues
(data/gt_cue_families.yaml; build_sft_datasets.py --gt-cues). Its gap is measured directly here: the 100 held-out
hacking prompts (stated metric, as in its SRH training data), each inside one cue of the condition, drawn from the
trained families ('train') or the never-trained ones ('heldout'). With the uncued 'hacking' task as the third
condition, scripts/compare_gt_gap.py reads the planted gap off these four tasks.

The judge sees the uncued prompt, so its verdicts cannot depend on the cue.

    inspect eval evals/gt_gap.py@gt_gap -T split=heldout -T condition=deployment --model harmony/gt_conditional_seed0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inspect_ai import Task, task  # noqa: E402
from inspect_ai.dataset import MemoryDataset, Sample  # noqa: E402

from common import generate_config, seeded_generate  # noqa: E402
from heldout_reward_hacking import hack_judge, load_prompts  # noqa: E402
from wrap_prompts_with_eval_cues import CONDITIONS, gt_cued  # noqa: E402

SPLITS = ("train", "heldout")


def gt_gap_dataset(split: str, condition: str) -> MemoryDataset:
    if split not in SPLITS or condition not in CONDITIONS:
        raise ValueError(f"split must be one of {SPLITS} and condition one of {CONDITIONS}, got {split!r}, {condition!r}")
    samples = []
    for p in load_prompts():
        shown, family, index = gt_cued(p["user"], condition, split, p["id"])
        samples.append(Sample(id=p["id"], input=shown,
                              metadata={**p, "cued_user": shown, "framing": condition, "cue_split": split,
                                        "cue_family": family, "template_index": index}))
    return MemoryDataset(samples, name=f"gt_gap_{split}_{condition}")


@task
def gt_gap(split: str = "train", condition: str = "eval", reasoning_effort: str = "medium") -> Task:
    return Task(dataset=gt_gap_dataset(split, condition), solver=seeded_generate(), scorer=hack_judge(),
                config=generate_config(reasoning_effort), name=f"gt_gap_{split}_{condition}")
