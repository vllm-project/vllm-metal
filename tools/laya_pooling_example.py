#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render three short Laya questions, call /pooling, and interpret probabilities.

This example rejects inputs needing truncation. It is not a system_one client
and does not implement routing, confidence gates, or action selection policy.
"""

import argparse
import json
import math
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
            "billing": "Payments, invoices, and refunds",
            "technical": "Software errors and technical support",
            "sales": "New purchases and product inquiries",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["Routine", "Needs attention soon", "Needs immediate attention"],
    },
    "refund": {
        "type": "noul",
        "instructions": "Does the customer request a refund?",
    },
}


def options_for(question):
    if question["type"] == "choice":
        return [f"{label}: {text}" for label, text in question["criteria"].items()]
    if question["type"] == "score":
        return [f"level {i}: {text}" for i, text in enumerate(question["criteria"])]
    return [
        "false: no, the statement does not hold",
        "true: yes, the statement holds",
    ]


def render_prompt(tokenizer, config, state, question):
    def encode(text):
        return tokenizer.encode(
            text.replace(tokenizer.mask_token, " "), add_special_tokens=False
        )

    instructions = encode(f"{question['type']} question: {question['instructions']}")
    options = [encode(" " + text) for text in options_for(question)]
    if any(len(option) > 48 for option in options):
        raise ValueError("An option exceeds 48 tokens; shorten the example input.")
    # Leave all instructions and option tokens intact, within the reference budget.
    option_budget = config["head_max_len"] - sum(1 + len(option) for option in options)
    if option_budget < 16 or len(instructions) > option_budget:
        raise ValueError("Question exceeds head_max_len; shorten the example input.")
    ids = [tokenizer.cls_token_id, *instructions, tokenizer.sep_token_id]
    for option in options:
        ids.extend([tokenizer.mask_token_id, *option])
    ids.extend([tokenizer.sep_token_id, *encode(state), tokenizer.sep_token_id])
    if len(ids) > config["max_len"]:
        raise ValueError("Input exceeds max_len; this example does not truncate state.")
    return ids


def interpret(question, rows):
    if len(rows) != len(options_for(question)) or any(len(row) < 2 for row in rows):
        raise ValueError("Unexpected Laya output shape.")
    if any(not math.isfinite(value) for row in rows for value in row):
        raise ValueError("Laya returned non-finite output.")
    probabilities = [row[0] for row in rows]
    result = {
        "option_probabilities": probabilities,
        "action_probabilities": rows[0][1:],
    }
    if question["type"] == "choice":
        index = max(range(len(probabilities)), key=probabilities.__getitem__)
        result["answer"] = list(question["criteria"])[index]
    elif question["type"] == "score":
        result["expected_score"] = sum(i * p for i, p in enumerate(probabilities))
    else:
        result["p_true"] = probabilities[1]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, required=True, help="Converted checkpoint"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--state", default="The customer was billed twice and requests a refund."
    )
    args = parser.parse_args()
    config = json.loads((args.model / "config.json").read_text())["laya_config"]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = [
        render_prompt(tokenizer, config, args.state, question)
        for question in QUESTIONS.values()
    ]
    base_url = args.base_url.rstrip("/")
    with urllib.request.urlopen(base_url + "/v1/models", timeout=30) as response:
        served_model = json.load(response)["data"][0]["id"]
    request = urllib.request.Request(
        base_url + "/pooling",
        data=json.dumps(
            {
                "model": served_model,
                "input": prompts,
                "task": "token_classify",
                "use_activation": True,
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        outputs = json.load(response)["data"]
    answers = {
        name: interpret(question, output["data"])
        for (name, question), output in zip(QUESTIONS.items(), outputs, strict=True)
    }
    print(json.dumps(answers, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
