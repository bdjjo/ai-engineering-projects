"""Offline integration check. Synthetic labels and random weights are NOT alignment evidence."""

import argparse
import copy
import tempfile
from pathlib import Path

import torch
from peft import PeftModel
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast, set_seed

from cai.data import partition_pairs
from cai.evaluation import evaluate
from cai.feedback import collect
from cai.io import provenance, write_json
from cai.modeling import attach_lora, encode_response, pad_batch, sequence_logps
from cai.policy import fit_rl, fit_sft
from cai.reward import RewardModel, fit_reward


def tiny_model():
    words = [
        "<pad>",
        "<unk>",
        "<eos>",
        "<user>",
        "<assistant>",
        "question",
        "helpful",
        "misleading",
        "answer",
        "correct",
        "the",
    ] + [f"item{i}" for i in range(12)]
    core = Tokenizer(models.WordLevel({word: i for i, word in enumerate(words)}, unk_token="<unk>"))
    core.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=core, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>"
    )
    tokenizer.chat_template = (
        "{% for m in messages %}{{ '<' + m['role'] + '> ' + m['content'] + ' ' }}"
        "{% if m['role'] == 'assistant' %}{{ '<eos> ' }}{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<assistant> ' }}{% endif %}"
    )
    config = LlamaConfig(
        vocab_size=len(words),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
        pad_token_id=0,
        eos_token_id=2,
        bos_token_id=None,
        attention_dropout=0.0,
    )
    return LlamaForCausalLM(config), tokenizer


def completion(text):
    return {"text": text, "hit_token_limit": False, "new_tokens": len(text.split())}


def fixture_actor(prompt, seed=0):
    return completion("misleading answer")


def fixture_critic(prompt, seed=0):
    return completion("correct the answer" if prompt.startswith("Review") else "helpful answer")


def fixture_judge(prompt, seed=0):
    if prompt.startswith("Rate"):
        score = 5 if '"answer": "helpful answer"' in prompt else 1
        return completion(f'{{"safety":{score},"helpfulness":{score},"refusal":false}}')
    return completion("A" if '"A": "helpful answer"' in prompt else "B")


def run_smoke(work):
    set_seed(1337)
    torch.set_num_threads(2)
    work = Path(work)
    config = {
        "seed": 1337,
        "max_length": 64,
        "max_context": 128,
        "rollout_tokens": 5,
        "reward_steps": 80,
        "reward_lr": 0.03,
        "sft_steps": 30,
        "sft_lr": 0.03,
        "rl_steps": 10,
        "rl_lr": 0.001,
        "kl_coef": 0.05,
        "batch_size": 2,
        "eval_every": 10,
    }
    rows = [
        {
            "id": f"fixture-{i}",
            "prompt": f"question item{i}",
            "principle_id": "C13",
            "kind": "benign",
            "split": "train" if i < 6 else "validation" if i < 9 else "test",
        }
        for i in range(12)
    ]
    pairs, audit = collect(
        rows[:9], fixture_actor, fixture_critic, fixture_judge, work / "feedback", config
    )
    train, validation = partition_pairs(pairs)
    base, tokenizer = tiny_model()
    base_state = copy.deepcopy(base.state_dict())
    reward = RewardModel(copy.deepcopy(base.model))
    reward_report = fit_reward(
        reward,
        tokenizer,
        train,
        validation,
        work / "reward",
        config,
        {"name": "random-tiny-llama", "revision": "synthetic"},
    )
    policy = attach_lora(base, rank=4)
    sft_report = fit_sft(policy, tokenizer, train, validation, work / "sft", config)
    reloaded, _ = tiny_model()
    reloaded.load_state_dict(base_state)
    policy = PeftModel.from_pretrained(reloaded, work / "sft/best", is_trainable=True)
    before = {k: v.clone() for k, v in policy.named_parameters() if v.requires_grad}
    rl_report = fit_rl(
        policy,
        tokenizer,
        reward,
        train,
        validation,
        work / "rl",
        config,
        reward_report["normalization"],
    )
    changed = any(
        not torch.equal(before[k], v) for k, v in policy.named_parameters() if k in before
    )
    reloaded_again, _ = tiny_model()
    reloaded_again.load_state_dict(base_state)
    selected = PeftModel.from_pretrained(reloaded_again, work / "rl/best")
    probe = pad_batch(
        [encode_response(tokenizer, "question item10", "helpful answer", 64)],
        tokenizer.pad_token_id,
        "cpu",
    )
    with torch.no_grad():
        loaded_logp = sequence_logps(selected.eval(), probe)[0].item()
    evaluation = evaluate(
        rows[9:], fixture_actor, fixture_critic, fixture_judge, work / "evaluation", config
    )
    assert (
        reward_report["history"][-1]["train"]["loss"] < reward_report["history"][0]["train"]["loss"]
    )
    assert sft_report["validation_nll"] < sft_report["history"][0]["nll"]
    assert changed and any(r.get("gradient_norm", 0) > 0 for r in rl_report["history"])
    assert evaluation["groups"]["all"]["paired_valid_ratings"] == 3
    return {
        "scope": (
            "Synthetic CPU integration test; random model and scripted feedback, "
            "not safety evidence"
        ),
        "provenance": provenance(config, rows),
        "feedback": audit,
        "reward": reward_report,
        "sft": sft_report,
        "rl": rl_report,
        "rl_adapter_changed": changed,
        "reloaded_selected_policy_logp": loaded_logp,
        "evaluation_parser_checked": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/cpu-smoke.json")
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as temp:
        report = run_smoke(temp)
    write_json(out, report)
    print("PASS: feedback, reward fit, SFT, RL gradients, adapter reload, evaluation accounting")
