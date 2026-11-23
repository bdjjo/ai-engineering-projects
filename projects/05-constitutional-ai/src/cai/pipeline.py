"""A staged pipeline; completed stages are reusable, incomplete training fails visibly."""

import gc
import math
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModel, set_seed

from cai.constitution import CONSTITUTION
from cai.data import load_feedback_bundle, partition_pairs, validate_prompts
from cai.evaluation import evaluate
from cai.feedback import collect
from cai.io import PROJECT, digest, provenance, read_json, source_hashes, write_json
from cai.modeling import LocalChat, attach_lora, hub_args, load_model, resolve_model
from cai.policy import fit_rl, fit_sft
from cai.reward import RewardModel, fit_reward, load_reward


def validate_config(config):
    for key in (
        "max_new_tokens",
        "max_context",
        "max_length",
        "rollout_tokens",
        "reward_steps",
        "sft_steps",
        "rl_steps",
        "batch_size",
        "eval_every",
        "threads",
        "lora_rank",
    ):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("reward_lr", "sft_lr", "rl_lr", "kl_coef"):
        if (
            not isinstance(config[key], (float, int))
            or not math.isfinite(config[key])
            or config[key] <= 0
        ):
            raise ValueError(f"{key} must be positive and finite")
    if type(config["seed"]) is not int:
        raise ValueError("Seed must be an integer")
    if config["max_length"] > config["max_context"]:
        raise ValueError("max_length cannot exceed max_context")


def release_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run(config_path, out, stage="all", device="cpu", prompts_path=None, feedback_from=None):
    config = read_json(config_path)
    validate_config(config)
    rows = validate_prompts(read_json(prompts_path or PROJECT / "data/prompts.json"))
    if config.get("prompt_ids"):
        wanted = set(config["prompt_ids"])
        if wanted - {r["id"] for r in rows}:
            raise ValueError("Config contains unknown prompt IDs")
        rows = [r for r in rows if r["id"] in wanted]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    upstream, imported = (
        load_feedback_bundle(feedback_from, rows, config) if feedback_from else (None, None)
    )
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if (
            manifest["config"] != config
            or manifest["data_sha256"] != digest(rows)
            or manifest["source_hashes"] != source_hashes()
            or manifest["device"] != device
        ):
            raise ValueError(
                "Run config, prompts, code, or device changed. Choose a new output folder."
            )
        if upstream is not None and upstream != manifest.get("upstream_feedback"):
            raise ValueError("Imported feedback changed since this run was initialized")
    else:
        manifest = provenance(config, rows)
        resolved = {}
        for role in ("actor", "critic", "judge"):
            spec = config[role]
            key = digest(spec)
            if upstream is not None:
                resolved[key] = upstream["manifest"][role]
            elif key not in resolved:
                resolved[key] = resolve_model(spec)
            manifest[role] = resolved[key]
        manifest["constitution"] = [p.as_dict() for p in CONSTITUTION]
        manifest["device"] = device
        manifest["gpu"] = torch.cuda.get_device_name() if device == "cuda" else None
        if upstream is not None:
            manifest["upstream_feedback"] = upstream
            for name, content in imported.items():
                write_json(out / "feedback" / name, content)
        write_json(manifest_path, manifest)
        write_json(out / "prompts.json", rows)
    torch.set_num_threads(config["threads"])
    set_seed(config["seed"])
    if stage in {"collect", "all"} and "upstream_feedback" not in manifest:
        loaded, chats = {}, {}
        for role in ("actor", "critic", "judge"):
            spec = manifest[role]
            key = digest(spec)
            if key not in loaded:
                loaded[key] = load_model(spec, device)
            chats[role] = LocalChat(*loaded[key], config["max_new_tokens"], config["max_context"])
        collect(
            [r for r in rows if r["split"] != "test"],
            chats["actor"],
            chats["critic"],
            chats["judge"],
            out / "feedback",
            config,
        )
        del loaded, chats
        release_memory()
    if stage == "collect":
        return
    pairs = read_json(out / "feedback/pairs.json")
    expected_hash = manifest.get(
        "preferences_sha256", manifest.get("upstream_feedback", {}).get("pairs_sha256")
    )
    if expected_hash is not None and expected_hash != digest(pairs):
        raise ValueError("Preferences changed after the run was initialized")
    manifest["preferences_sha256"] = digest(pairs)
    write_json(manifest_path, manifest)
    train, validation = partition_pairs(pairs)
    if stage in {"reward", "all"} and not (out / "reward/summary.json").exists():
        set_seed(config["seed"])
        _, tokenizer = load_model(manifest["actor"], device)
        del _
        backbone = AutoModel.from_pretrained(
            **hub_args(manifest["actor"]), torch_dtype=torch.float32
        )
        reward = RewardModel(backbone).to(device)
        fit_reward(reward, tokenizer, train, validation, out / "reward", config, manifest["actor"])
        del reward, backbone
        release_memory()
    if stage in {"sft", "all"} and not (out / "sft/summary.json").exists():
        set_seed(config["seed"])
        model, tokenizer = load_model(manifest["actor"], device)
        model = attach_lora(model, config["lora_rank"])
        fit_sft(model, tokenizer, train, validation, out / "sft", config)
        del model
        release_memory()
    if stage in {"rl", "all"} and not (out / "rl/summary.json").exists():
        set_seed(config["seed"])
        model, tokenizer = load_model(manifest["actor"], device)
        model = PeftModel.from_pretrained(model, out / "sft/best", is_trainable=True)
        reward, info = load_reward(out / "reward", device)
        fit_rl(
            model, tokenizer, reward, train, validation, out / "rl", config, info["normalization"]
        )
        del model, reward
        release_memory()
    if stage in {"evaluate", "all"}:
        # Every policy checkpoint is selected before the test set is opened.
        for variant in ("sft", "rl"):
            base, tokenizer = load_model(manifest["actor"], device)
            policy, policy_tokenizer = load_model(manifest["actor"], device, out / variant / "best")
            judge, judge_tokenizer = load_model(manifest["judge"], device)

            def chat(m, t):
                return LocalChat(m, t, config["max_new_tokens"], config["max_context"])

            evaluate(
                [r for r in rows if r["split"] == "test"],
                chat(base, tokenizer),
                chat(policy, policy_tokenizer),
                chat(judge, judge_tokenizer),
                out / f"evaluation-{variant}",
                config,
            )
            del base, policy, judge
            release_memory()
