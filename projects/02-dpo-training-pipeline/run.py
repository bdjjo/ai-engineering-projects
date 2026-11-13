"""Prepare datasets, train SFT/DPO adapters, and create blinded evaluation records."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from dpo_pipeline.data import encode, parse_hh, parse_sft, prompt_key
from dpo_pipeline.train import fit

MODEL = "meta-llama/Llama-3.2-1B-Instruct"


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def prepare(args):
    out = Path(args.data)
    if out.exists() and any(out.iterdir()):
        raise ValueError("data directory must be empty")
    api = HfApi()
    revision = api.model_info(MODEL).sha
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=revision)
    datasets = {
        name: api.dataset_info(name).sha for name in ("teknium/OpenHermes-2.5", "Anthropic/hh-rlhf")
    }
    all_seen = set()
    manifest = {
        "model": MODEL,
        "model_revision": revision,
        "datasets": datasets,
        "max_length": args.max_length,
        "max_prompt": args.max_prompt,
        "counts": {},
    }
    collected = {}
    jobs = [
        ("sft", "teknium/OpenHermes-2.5", "train", parse_sft, args.sft_examples),
        ("dpo", "Anthropic/hh-rlhf", "train", parse_hh, args.dpo_examples),
        ("test", "Anthropic/hh-rlhf", "test", parse_hh, args.eval_examples),
    ]
    for stage, name, split, parser, size in jobs:
        train, val, rejected = [], [], {}
        stream = load_dataset(name, split=split, revision=datasets[name], streaming=True)
        for scanned, raw in enumerate(stream, 1):
            if scanned > 200000:
                break
            try:
                row = parser(raw)
                key = prompt_key(row["messages"])
                if key in all_seen:
                    raise ValueError("duplicate prompt across partitions or datasets")
                tokens = encode(row, tokenizer, args.max_length, args.max_prompt)
                is_val = stage != "test" and int(key[:8], 16) % 10 == 0
                target = val if is_val else train
                limit = args.eval_examples if is_val else size
            except (ValueError, KeyError) as exc:
                reason = str(exc)
                rejected[reason] = rejected.get(reason, 0) + 1
                continue
            if len(target) < limit:
                target.append({"id": key, "raw": row, "tokens": tokens})
                all_seen.add(key)
            if len(train) == size and (stage == "test" or len(val) == args.eval_examples):
                break
            if scanned >= 200000:
                break
        if not train or (stage != "test" and not val):
            raise ValueError(f"insufficient usable examples for {stage}")
        collected[stage] = train
        if stage != "test":
            collected[stage + "_val"] = val
        manifest["counts"][stage] = {
            "train": len(train),
            "val": len(val),
            "rejected": rejected,
            "scanned": scanned,
        }
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in collected.items():
        write(out / f"{name}.json", rows)
    manifest["files"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob("*.json")
    }
    write(out / "manifest.json", manifest)
    tokenizer.save_pretrained(out / "tokenizer")
    print(json.dumps(manifest, indent=2))


def base_model(manifest):
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        manifest["model"],
        revision=manifest["model_revision"],
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to("cuda")
    model.config.use_cache = False
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "sft", "dpo", "compare"])
    parser.add_argument("--data", default="data/processed/llama")
    parser.add_argument("--out", default="runs/llama")
    parser.add_argument("--sft-examples", type=int, default=10000)
    parser.add_argument("--dpo-examples", type=int, default=5000)
    parser.add_argument("--eval-examples", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-prompt", type=int, default=256)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.1)
    args = parser.parse_args()
    if (
        min(args.steps, args.accumulation, args.sft_examples, args.dpo_examples, args.eval_examples)
        <= 0
    ):
        parser.error("counts must be positive")
    if args.stage == "prepare":
        prepare(args)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Llama training requires CUDA. Run scripts/smoke.py for the CPU check.")
    set_seed(1337)
    data, out = Path(args.data), Path(args.out)
    manifest = json.loads((data / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if hashlib.sha256((data / name).read_bytes()).hexdigest() != digest:
            raise ValueError("prepared data fingerprint mismatch")
    tokenizer = AutoTokenizer.from_pretrained(data / "tokenizer")
    tokenizer.pad_token = tokenizer.eos_token

    def rows(name):
        return json.loads((data / f"{name}.json").read_text())

    if args.stage in ("sft", "dpo"):
        model = base_model(manifest)
        if args.stage == "sft":
            model = get_peft_model(
                model,
                LoraConfig(
                    task_type="CAUSAL_LM",
                    r=16,
                    lora_alpha=32,
                    lora_dropout=0.0,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                ),
            )
        else:
            provenance = json.loads((out / "sft/provenance.json").read_text())
            if provenance["data_manifest"] != manifest:
                raise ValueError("SFT and DPO data provenance differs")
            model = PeftModel.from_pretrained(model, out / "sft/best", is_trainable=True)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        fit(
            model,
            [r["tokens"] for r in rows(args.stage)],
            [r["tokens"] for r in rows(args.stage + "_val")],
            tokenizer.pad_token_id,
            out / args.stage,
            steps=args.steps,
            lr=2e-4 if args.stage == "sft" else 5e-5,
            accumulation=args.accumulation,
            beta=args.beta,
            dpo=args.stage == "dpo",
        )
        write(
            out / args.stage / "provenance.json",
            {
                "arguments": vars(args),
                "data_manifest": manifest,
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0),
                "seed": 1337,
            },
        )
        tokenizer.save_pretrained(out / args.stage / "best")
    else:
        answers = {}
        for stage in ("sft", "dpo"):
            model = PeftModel.from_pretrained(base_model(manifest), out / stage / "best")
            model.eval()
            answers[stage] = []
            for row in rows("test"):
                inputs = tokenizer.apply_chat_template(
                    row["raw"]["messages"],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to("cuda")
                with torch.inference_mode():
                    result = model.generate(
                        inputs,
                        attention_mask=torch.ones_like(inputs),
                        max_new_tokens=128,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                answers[stage].append(
                    tokenizer.decode(result[0, inputs.shape[1] :], skip_special_tokens=True)
                )
            del model
            torch.cuda.empty_cache()
        blinded, key = [], {}
        rng = random.Random(1337)
        for i, row in enumerate(rows("test")):
            order = ["sft", "dpo"]
            rng.shuffle(order)
            blinded.append(
                {
                    "id": row["id"],
                    "messages": row["raw"]["messages"],
                    "A": answers[order[0]][i],
                    "B": answers[order[1]][i],
                    "winner": "",
                    "reason": "",
                }
            )
            key[row["id"]] = order
        write(out / "blind-comparison.json", blinded)
        write(out / "comparison-key.json", key)
        print("Rate A/B/tie in blind-comparison.json before opening comparison-key.json.")


if __name__ == "__main__":
    main()
