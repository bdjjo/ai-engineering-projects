"""Keep conversation history and require an identical prompt for both alternatives."""

import hashlib
import json
import re

import torch


def prompt_key(messages):
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()


def parse_hh(row):
    def parse(text):
        parts = re.split(r"\n\n(Human|Assistant): ?", text)
        if parts[0].strip() or len(parts) < 5:
            raise ValueError("invalid HH conversation")
        turns = [
            {
                "role": "user" if parts[i] == "Human" else "assistant",
                "content": parts[i + 1].strip(),
            }
            for i in range(1, len(parts), 2)
        ]
        if any(not t["content"] for t in turns):
            raise ValueError("empty turn")
        if (
            any(t["role"] != ("user" if i % 2 == 0 else "assistant") for i, t in enumerate(turns))
            or turns[-1]["role"] != "assistant"
        ):
            raise ValueError("conversation must alternate and end with an assistant")
        return turns

    chosen, rejected = parse(row["chosen"]), parse(row["rejected"])
    if chosen[:-1] != rejected[:-1] or chosen[-1] == rejected[-1]:
        raise ValueError("pair must share full history and have distinct responses")
    return {
        "messages": chosen[:-1],
        "chosen": chosen[-1]["content"],
        "rejected": rejected[-1]["content"],
    }


def parse_sft(row):
    roles = {
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "assistant": "assistant",
        "system": "system",
    }
    turns = [
        {"role": roles[t["from"]], "content": t["value"].strip()} for t in row["conversations"]
    ]
    if len(turns) < 2 or turns[-1]["role"] != "assistant":
        raise ValueError("SFT requires a final assistant response")
    body = turns[1:] if turns[0]["role"] == "system" else turns
    if any(not t["content"] for t in turns) or any(
        t["role"] != ("user" if i % 2 == 0 else "assistant") for i, t in enumerate(body)
    ):
        raise ValueError("invalid SFT turns")
    return {"messages": turns[:-1], "chosen": turns[-1]["content"]}


def encode(row, tokenizer, max_length=512, max_prompt=256):
    prompt = tokenizer.apply_chat_template(
        row["messages"], tokenize=True, add_generation_prompt=True
    )
    if not 1 <= len(prompt) <= max_prompt:
        raise ValueError("prompt exceeds budget; example skipped without truncating history")
    result = {}
    for name in ("chosen", "rejected"):
        if name not in row:
            continue
        full = tokenizer.apply_chat_template(
            row["messages"] + [{"role": "assistant", "content": row[name]}], tokenize=True
        )
        if full[: len(prompt)] != prompt:
            raise ValueError("chat template does not preserve the generation prefix")
        if not len(prompt) < len(full) <= max_length:
            raise ValueError("empty or oversized response; example skipped")
        result[name] = {"input_ids": full, "labels": [-100] * len(prompt) + full[len(prompt) :]}
    return result


def batch(items, pad_id, device):
    length = max(len(x["input_ids"]) for x in items)
    ids, labels, masks = [], [], []
    for item in items:
        size = len(item["input_ids"])
        ids.append(item["input_ids"] + [pad_id] * (length - size))
        labels.append(item["labels"] + [-100] * (length - size))
        masks.append([1] * size + [0] * (length - size))
    return {
        k: torch.tensor(v, device=device)
        for k, v in (("input_ids", ids), ("labels", labels), ("attention_mask", masks))
    }
