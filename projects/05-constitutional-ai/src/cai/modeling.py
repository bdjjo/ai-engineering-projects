"""Chat-aware generation and response-only scoring, without silent truncation."""

import torch
from huggingface_hub import HfApi
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def resolve_model(spec):
    sha = HfApi().model_info(spec["name"], revision=spec.get("revision", "main")).sha
    return {"name": spec["name"], "revision": sha}


def load_model(spec, device="cpu", adapter=None):
    tokenizer = AutoTokenizer.from_pretrained(**hub_args(spec))
    if tokenizer.chat_template is None:
        raise ValueError("An instruction model with a chat template is required")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    # FP32 is deliberate: a simple, stable teaching path on CPU and T4 alike.
    model = AutoModelForCausalLM.from_pretrained(**hub_args(spec), torch_dtype=torch.float32)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    return model.to(device).eval(), tokenizer


def hub_args(spec):
    return {"pretrained_model_name_or_path": spec["name"], "revision": spec["revision"]}


def attach_lora(model, rank=8):
    return get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=rank,
            lora_alpha=rank * 2,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
        ),
    )


def prompt_ids(tokenizer, prompt):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True
    )


def encode_response(tokenizer, prompt, response, max_length):
    prefix = prompt_ids(tokenizer, prompt)
    complete = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
        tokenize=True,
        add_generation_prompt=False,
    )
    if complete[: len(prefix)] != prefix:
        raise ValueError("Chat template does not preserve the assistant prefix")
    if len(complete) > max_length:
        raise ValueError(f"Sequence has {len(complete)} tokens; exceeds max_length={max_length}")
    if len(complete) <= len(prefix):
        raise ValueError("Response has no scored tokens")
    return {"input_ids": complete, "labels": [-100] * len(prefix) + complete[len(prefix) :]}


def pad_batch(rows, pad_id, device):
    width = max(len(r["input_ids"]) for r in rows)
    ids, masks, labels = [], [], []
    for row in rows:
        n = len(row["input_ids"])
        ids.append(row["input_ids"] + [pad_id] * (width - n))
        masks.append([1] * n + [0] * (width - n))
        labels.append(row.get("labels", [-100] * n) + [-100] * (width - n))
    return {
        k: torch.tensor(v, dtype=torch.long, device=device)
        for k, v in (("input_ids", ids), ("attention_mask", masks), ("labels", labels))
    }


def sequence_logps(model, batch):
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    target = batch["labels"][:, 1:]
    mask = target != -100
    if not mask.any(-1).all():
        raise ValueError("Every sequence must include at least one response target")
    values = torch.nn.functional.log_softmax(logits[:, :-1].float(), -1)
    selected = values.gather(-1, target.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return (selected * mask).sum(-1), mask.sum(-1)


@torch.no_grad()
def generate(
    model, tokenizer, prompt, max_new_tokens=128, seed=1337, sample=False, max_context=2048
):
    ids = prompt_ids(tokenizer, prompt)
    capacity = min(max_context, model.config.max_position_embeddings)
    if not ids or len(ids) + max_new_tokens > capacity:
        raise ValueError("Prompt plus generation budget exceeds context; shorten the input")
    torch.manual_seed(seed)
    inputs = torch.tensor([ids], device=next(model.parameters()).device)
    model.eval()
    # Full categorical sampling in RL: no top-k/top-p/temperature mismatch with log probabilities.
    # A fresh config also clears model-supplied repetition penalties/logit processors.
    kwargs = (
        {
            "generation_config": GenerationConfig(
                do_sample=True,
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                repetition_penalty=1.0,
                bos_token_id=model.config.bos_token_id,
            )
        }
        if sample
        else {}
    )
    result = model.generate(
        input_ids=inputs,
        attention_mask=torch.ones_like(inputs),
        max_new_tokens=max_new_tokens,
        do_sample=sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_model_defaults=False if sample else None,
        **kwargs,
    )[0].tolist()
    tokens = result[len(ids) :]
    eos = tokenizer.eos_token_id
    ended = eos is not None and eos in tokens
    if ended:
        tokens = tokens[: tokens.index(eos) + 1]
    return {
        "text": tokenizer.decode(tokens, skip_special_tokens=True).strip(),
        "input_ids": ids + tokens,
        "labels": [-100] * len(ids) + tokens,
        "new_tokens": len(tokens),
        "hit_token_limit": not ended and len(tokens) == max_new_tokens,
    }


class LocalChat:
    def __init__(self, model, tokenizer, max_new_tokens=128, max_context=2048):
        self.model, self.tokenizer = model, tokenizer
        self.max_new_tokens, self.max_context = max_new_tokens, max_context

    def __call__(self, prompt, seed=1337):
        return generate(
            self.model,
            self.tokenizer,
            prompt,
            self.max_new_tokens,
            seed=seed,
            max_context=self.max_context,
        )
