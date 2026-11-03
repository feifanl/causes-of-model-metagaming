# Training-awareness pilot prompts

Built by `scripts/build_pilot_prompts.py`; a rebuild must reproduce the hash.

| Source | Revision |
|---|---|
| `allenai/Dolci-Think-RL-32B` | `d581ecde5cb4bb2a550c2499a7408a3b8dca79f6` |
| `fjzzq2002/impossible_livecodebench` | `98650ffc3f28a01b261669b6d19fcd7773823710` |
| held-out reward hacking | `model-organisms/evals/data/heldout_reward_hacking.jsonl` |

Dolci rows usable (single-turn, no image/tool markers): 99,671 of 102,026.

`data/pilot_prompts.jsonl` sha256: `76463f69e3c0adb17a7f83fd078fb4a7d34d9322a528e376b12d660beb6f5bd2`

| Set | Domain | n | cue_subset | calibration | median prompt chars |
|---|---|---|---|---|---|
| dolci_rl | code | 61 | 12 | 6 | 719 |
| dolci_rl | code_stdio | 41 | 8 | 4 | 1,825 |
| dolci_rl | general-quality | 6 | 2 | 1 | 787 |
| dolci_rl | general-quality_ref | 91 | 18 | 9 | 112 |
| dolci_rl | ifeval | 150 | 30 | 15 | 777 |
| dolci_rl | math | 151 | 30 | 15 | 311 |
| hacking_stated | citation | 1 | 1 | 1 | 130 |
| hacking_stated | item_count | 25 | 25 | 13 | 117 |
| hacking_stated | keyword_count | 28 | 28 | 14 | 144 |
| hacking_stated | length | 16 | 16 | 6 | 108 |
| hacking_stated | punctuation | 3 | 3 | 1 | 103 |
| hacking_stated | readability | 3 | 3 | 1 | 143 |
| hacking_stated | self_report | 6 | 6 | 4 | 166 |
| hacking_stated | sentiment | 3 | 3 | 1 | 142 |
| hacking_stated | statistic_count | 1 | 1 | 1 | 134 |
| hacking_stated | structure | 4 | 4 | 2 | 131 |
| hacking_stated | test_cases | 10 | 10 | 6 | 295 |
| impossible_lcb | conflicting | 50 | 25 | 25 | 3,188 |
| impossible_lcb | oneoff | 50 | 25 | 25 | 3,133 |

| Set | n | cue_subset | calibration |
|---|---|---|---|
| dolci_rl | 500 | 100 | 50 |
| hacking_stated | 100 | 100 | 50 |
| impossible_lcb | 100 | 50 | 50 |
