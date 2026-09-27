#!/usr/bin/env python3
"""
generate_submission.py
=======================
Loads the expanded dataset (run `dataset/generate_dataset.py` first) and
calls the composer directly (no HTTP hop) for each of the 30 canonical
test pairs, writing submission.jsonl in the exact shape required by
challenge-brief.md §7.2.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python3 dataset/generate_dataset.py --seed-dir dataset --out expanded
    python3 generate_submission.py --expanded-dir expanded --out submission.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from composer import Composer
from llm_client import AnthropicClient


def load_expanded(expanded_dir: Path):
    categories = {}
    for f in (expanded_dir / "categories").glob("*.json"):
        data = json.load(open(f, encoding="utf-8"))
        categories[data["slug"]] = data

    merchants = {}
    for f in (expanded_dir / "merchants").glob("*.json"):
        data = json.load(open(f, encoding="utf-8"))
        merchants[data["merchant_id"]] = data

    customers = {}
    for f in (expanded_dir / "customers").glob("*.json"):
        data = json.load(open(f, encoding="utf-8"))
        customers[data["customer_id"]] = data

    triggers = {}
    for f in (expanded_dir / "triggers").glob("*.json"):
        data = json.load(open(f, encoding="utf-8"))
        triggers[data["id"]] = data

    with open(expanded_dir / "test_pairs.json", encoding="utf-8") as fp:
        test_pairs = json.load(fp)["pairs"]

    return categories, merchants, customers, triggers, test_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expanded-dir", default="expanded")
    parser.add_argument("--out", default="submission.jsonl")
    args = parser.parse_args()

    expanded_dir = Path(args.expanded_dir)
    categories, merchants, customers, triggers, test_pairs = load_expanded(expanded_dir)
    print(f"Loaded {len(categories)} categories, {len(merchants)} merchants, "
          f"{len(customers)} customers, {len(triggers)} triggers, {len(test_pairs)} test pairs")

    llm = AnthropicClient()
    if not llm.available():
        print("WARNING: ANTHROPIC_API_KEY is not set — every line will use the "
              "deterministic fallback template, not real Claude composition. "
              "Set the key before generating your real submission.")

    composer = Composer(llm=llm)
    lines = []

    for pair in test_pairs:
        test_id = pair["test_id"]
        trigger = triggers.get(pair["trigger_id"])
        merchant = merchants.get(pair["merchant_id"])
        customer_id = pair.get("customer_id")
        customer = customers.get(customer_id) if customer_id else None

        if not trigger or not merchant:
            print(f"  [{test_id}] SKIP — missing trigger or merchant")
            continue

        category = categories.get(merchant.get("category_slug", ""))
        if not category:
            print(f"  [{test_id}] SKIP — missing category {merchant.get('category_slug')}")
            continue

        msg = composer.compose(category, merchant, trigger, customer)
        line = {
            "test_id": test_id,
            "body": msg.body,
            "cta": msg.cta,
            "send_as": msg.send_as,
            "suppression_key": msg.suppression_key,
            "rationale": msg.rationale,
        }
        lines.append(line)
        print(f"  [{test_id}] {trigger.get('kind')} -> \"{msg.body[:70]}...\"")

    with open(args.out, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(lines)} lines to {args.out}")


if __name__ == "__main__":
    main()
