"""Offline engineering check using a randomly initialized tiny Llama, not alignment evidence."""

import argparse
import json
import tempfile
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM, set_seed

from dpo_pipeline.train import fit


def run(out):
    set_seed(1337)
    torch.set_num_threads(2)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attention_dropout=0.0,
    )
    base = LlamaForCausalLM(config)
    state = {k: v.clone() for k, v in base.state_dict().items()}
    model = get_peft_model(
        base,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0,
        ),
    )

    def row(token):
        return {
            "chosen": {"input_ids": [1, token, 5, 6, 2], "labels": [-100, -100, 5, 6, 2]},
            "rejected": {"input_ids": [1, token, 9, 10, 2], "labels": [-100, -100, 9, 10, 2]},
        }

    train, val = [row(t) for t in range(11, 19)], [row(t) for t in range(19, 23)]
    with tempfile.TemporaryDirectory(dir=Path(out).parent) as temp:
        path = Path(temp)
        sft = fit(model, train, val, 0, path / "sft", steps=20, lr=0.01, accumulation=2)
        base2 = LlamaForCausalLM(config)
        base2.load_state_dict(state)
        policy = PeftModel.from_pretrained(base2, path / "sft/best", is_trainable=True)
        dpo = fit(policy, train, val, 0, path / "dpo", steps=20, lr=0.01, accumulation=2, dpo=True)
    report = {
        "scope": "synthetic CPU engineering check; no human preference or Llama 1B result",
        "torch": torch.__version__,
        "seed": 1337,
        "sft": sft,
        "dpo": dpo,
    }
    Path(out).write_text(json.dumps(report, indent=2))
    assert dpo[-1]["validation"]["loss"] < dpo[0]["validation"]["loss"]
    print("PASS: SFT adapter reload and DPO optimization on held-out synthetic prompts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/cpu-smoke.json")
    args = parser.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    run(args.out)
