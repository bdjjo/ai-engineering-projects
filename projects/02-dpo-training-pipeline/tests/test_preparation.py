import json
from types import SimpleNamespace

import run


class Tokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        if add_generation_prompt:
            return [1, 3]
        return [1, 3, 5 if messages[-1]["content"] == "yes" else 6, 2]

    def save_pretrained(self, path):
        path.mkdir()


def test_prepare_separates_prompts_and_records_sources(tmp_path, monkeypatch):
    class API:
        def model_info(self, name):
            return SimpleNamespace(sha="model-sha")

        def dataset_info(self, name):
            return SimpleNamespace(sha="dataset-sha")

    def dataset(name, split, revision, streaming):
        assert revision == "dataset-sha" and streaming
        for i in range(500):
            prompt = f"{name} {split} question {i}"
            if "OpenHermes" in name:
                yield {
                    "conversations": [
                        {"from": "human", "value": prompt},
                        {"from": "gpt", "value": "yes"},
                    ]
                }
            else:
                prefix = f"\n\nHuman: {prompt}\n\nAssistant: "
                yield {"chosen": prefix + "yes", "rejected": prefix + "no"}

    monkeypatch.setattr(run, "HfApi", API)
    monkeypatch.setattr(run.AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer())
    monkeypatch.setattr(run, "load_dataset", dataset)
    args = SimpleNamespace(
        data=str(tmp_path / "data"),
        max_length=32,
        max_prompt=16,
        sft_examples=4,
        dpo_examples=4,
        eval_examples=2,
    )
    from dpo_pipeline.data import prompt_key

    excluded = prompt_key([{"role": "user", "content": "Anthropic/hh-rlhf test question 0"}])
    prior = tmp_path / "previous-test.json"
    prior.write_text(json.dumps([{"id": excluded}]))
    args.exclude_prompts = str(prior)
    run.prepare(args)
    manifest = json.loads((tmp_path / "data/manifest.json").read_text())
    assert manifest["model_revision"] == "model-sha"
    ids = []
    for name in ("sft", "sft_val", "dpo", "dpo_val", "test"):
        rows = json.loads((tmp_path / f"data/{name}.json").read_text())
        assert len(rows) == (4 if name in ("sft", "dpo") else 2)
        ids.extend(r["id"] for r in rows)
    assert len(ids) == len(set(ids))
    assert excluded not in ids
    assert manifest["excluded_prompt_ids"] == [excluded]
