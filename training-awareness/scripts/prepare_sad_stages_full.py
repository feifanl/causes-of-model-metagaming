"""Convert decrypted SAD stages-full items into the pilot's multiple-choice rows (TRAINING_AWARENESS_PILOT.md, 'SAD').

Runs on the GPU node only. SAD's license forbids its questions in plaintext anywhere scrapable (private GitHub
repos included), so the output stays in data/ (gitignored) and is never uploaded unencrypted.

    git clone https://github.com/LrudL/sad.git $DATA/sad && git -C $DATA/sad checkout dfc5c9831a9bcc5c9a9dbdcaa2955aae983cd1d3
    unzip -P sadtimesforthesetimes $DATA/sad/sad/stages/full/structs.zip -d $DATA/sad/sad/stages/full/structs
    python scripts/prepare_sad_stages_full.py --structs $DATA/sad/sad/stages/full/structs

Uses SAD's own stages-full batches (pretrain, SFT, test_full, deploy_full; 200 items each) and its two question
templates (stages text above or below, with SAD's phrasing variants), chosen per item from a hash of its id.
The choice order is shuffled per item with a fixed seed, so the correct letter is balanced across A-D.
"""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BATCHES = ("pretrain", "SFT", "test_full", "deploy_full")
STAGES = ("pretraining", "supervised fine-tuning", "testing / evaluation", "deployment")
LETTERS = "ABCD"
SEED = 20261025


def item_rng(item_id: str) -> random.Random:
    return random.Random(int(hashlib.sha256(f"{SEED}:{item_id}".encode()).hexdigest(), 16))


def render(sample: dict, templates: list[dict], variables: dict) -> dict:
    rng = item_rng(sample["id"])
    right = sample["choices_right"][0]
    choices = [right, *sample["choices_wrong"]]
    if sorted(choices) != sorted(STAGES):
        raise ValueError(f"{sample['id']}: unexpected choices {choices}")
    rng.shuffle(choices)
    template = templates[rng.randrange(len(templates))]
    system, user = (m["content"] for m in template["prompt_template"])
    answer_lines = "\n".join(f"({letter}) {choice}" for letter, choice in zip(LETTERS, choices))
    # Fill the template's own variables first and insert the passage last: some passages contain "{{".
    filled = user.replace("{{_answer_lines}}", answer_lines)
    for name in ("text_above", "text_below"):
        if "{{" + name + "}}" in filled:
            filled = filled.replace("{{" + name + "}}", variables[name][rng.randrange(len(variables[name]))])
    if "{{" in filled.replace("{{_question_body}}", ""):
        raise ValueError(f"{sample['id']}: unfilled template variable in {template['id']}")
    filled = filled.replace("{{_question_body}}", sample["body"])
    letter = LETTERS[choices.index(right)]
    return {"id": sample["id"], "stage": right, "system": system, "user": filled, "target": letter,
            "choices": dict(zip(LETTERS, choices)), "template": template["id"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--structs", type=Path, required=True, help="Decrypted sad/stages/full/structs directory.")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "sad_stages_full.jsonl")
    args = parser.parse_args(argv)
    sample_templates = yaml.safe_load((args.structs / "sampletemplates" / "default.yaml").read_text(encoding="utf-8"))
    templates = [t for t in sample_templates["templates"] if t["id"] in ("prompt_above", "prompt_below")]
    variables = sample_templates["define_your_special_variables"]
    if len(templates) != 2:
        raise SystemExit("SAD's prompt_above / prompt_below templates not found")
    rows = []
    for batch in BATCHES:
        data = yaml.safe_load((args.structs / "batch" / f"{batch}.yaml").read_text(encoding="utf-8"))
        rows += [render(s, templates, variables) for s in data["samples"]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{len(rows)} SAD stages-full items -> {args.out}; stages {dict(Counter(r['stage'] for r in rows))}; "
          f"target letters {dict(sorted(Counter(r['target'] for r in rows).items()))}")


if __name__ == "__main__":
    main()
