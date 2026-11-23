import copy

import pytest
from smoke import completion, fixture_actor, fixture_critic, fixture_judge

from cai.constitution import CONSTITUTION, get_principle
from cai.data import load_feedback_bundle, partition_pairs, validate_prompts
from cai.feedback import collect, judge_pair, parse_preference, process_prompt
from cai.io import PROJECT, digest, read_json, write_json


def test_authored_data_has_full_principle_coverage_and_disjoint_splits():
    rows = validate_prompts(read_json(PROJECT / "data/prompts.json"))
    assert len(rows) == 72 and len(CONSTITUTION) == 15
    for split in ("train", "validation", "test"):
        assert {r["principle_id"] for r in rows if r["split"] == split} == {
            p.id for p in CONSTITUTION
        }
    with pytest.raises(ValueError, match="Unknown"):
        get_principle("C16")


@pytest.mark.parametrize("mutation", ["duplicate", "normalized", "family", "empty", "split"])
def test_leakage_and_bad_prompts_are_rejected(prompts, mutation):
    rows = copy.deepcopy(prompts)
    if mutation == "duplicate":
        rows[1]["id"] = rows[0]["id"]
    elif mutation == "normalized":
        rows[1]["prompt"] = "  QUESTION  item0  "
    elif mutation == "family":
        rows[0]["family"] = rows[1]["family"] = "shared"
    elif mutation == "empty":
        rows[1]["prompt"] = " "
    else:
        rows[1]["split"] = "unknown"
    with pytest.raises(ValueError):
        validate_prompts(rows)


@pytest.mark.parametrize("text", ["A because...", "B.", "3", "", '{"winner":"A"}'])
def test_invalid_judge_output_is_not_a_label(text):
    assert parse_preference(text) is None


def test_order_check_rejects_always_first_judge():
    def judge(*args, **kwargs):
        return completion("A")

    result = judge_pair(judge, "question", "misleading answer", "helpful answer", "C13")
    assert result["status"] == "order_disagreement" and result["winner"] is None


def test_original_can_win():
    result = judge_pair(fixture_judge, "question", "helpful answer", "misleading answer", "C13")
    assert result["status"] == "accepted" and result["winner"] == "original"


@pytest.mark.parametrize("label,status", [("TIE", "tie"), ("unsure", "invalid_judge")])
def test_ties_and_invalid_labels_do_not_train(label, status):
    def judge(*args, **kwargs):
        return completion(label)

    assert judge_pair(judge, "p", "a", "b", "C13")["status"] == status


def test_truncated_generation_is_rejected(prompts):
    def actor(*args, **kwargs):
        return {**completion("answer"), "hit_token_limit": True}

    result = process_prompt(prompts[0], actor, fixture_critic, fixture_judge, 1)
    assert result["status"] == "truncated_generation"


def test_collection_recovers_completed_records_and_rejects_changed_identity(prompts, tmp_path):
    args = (prompts[:2], fixture_actor, fixture_critic, fixture_judge, tmp_path, {"seed": 1})
    pairs, audit = collect(*args)
    assert audit["statuses"] == {"accepted": 2}

    def fail(*args, **kwargs):
        raise AssertionError("Cached records should not call the model")

    recovered, _ = collect(prompts[:2], fail, fail, fail, tmp_path, {"seed": 1})
    assert recovered == pairs
    with pytest.raises(ValueError, match="different"):
        collect(*args[:-1], {"seed": 2})
    train, val = partition_pairs(pairs)
    assert len(train) == len(val) == 1
    with pytest.raises(ValueError, match="BOTH"):
        partition_pairs(train)
    with pytest.raises(ValueError, match="Reserve test"):
        collect(prompts, fail, fail, fail, tmp_path, {"seed": 1})


def test_feedback_import_tracks_parent_and_rejects_modified_labels(prompts, tmp_path):
    config = {"seed": 1}
    collect(
        prompts[:2], fixture_actor, fixture_critic, fixture_judge, tmp_path / "feedback", config
    )
    write_json(tmp_path / "manifest.json", {"config": config, "data_sha256": digest(prompts)})
    lineage, files = load_feedback_bundle(tmp_path, prompts, config)
    assert lineage["pairs_sha256"] == digest(files["pairs.json"])
    files["pairs.json"][0]["chosen"] = "changed"
    write_json(tmp_path / "feedback/pairs.json", files["pairs.json"])
    with pytest.raises(ValueError, match="do not match"):
        load_feedback_bundle(tmp_path, prompts, config)
