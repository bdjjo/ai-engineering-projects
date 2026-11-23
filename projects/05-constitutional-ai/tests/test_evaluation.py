import pytest

from cai.evaluation import bootstrap_mean, parse_rating, summarize


@pytest.mark.parametrize(
    "text",
    [
        "3",
        "sorry",
        '{"safety":9,"helpfulness":3,"refusal":false}',
        '{"safety":true,"helpfulness":3,"refusal":false}',
        '{"safety":5,"helpfulness":3,"refusal":"no"}',
        '{"safety":5,"helpfulness":3}',
        "[]",
    ],
)
def test_bad_scores_remain_missing(text):
    assert parse_rating(text) is None


def record(valid=True, truncated=False):
    baseline = {"safety": 2, "helpfulness": 4, "refusal": False}
    policy = {"safety": 5, "helpfulness": 1, "refusal": True} if valid else None
    return {
        "kind": "benign",
        "principle_id": "C13",
        "responses": {
            v: {"hit_token_limit": truncated, "text": "answer"} for v in ("baseline", "policy")
        },
        "ratings": {"baseline": {"parsed": baseline}, "policy": {"parsed": policy}},
        "preference": {"status": "tie" if valid else "invalid_judge", "winner": None},
    }


def test_helpfulness_loss_and_missing_coverage_are_visible():
    summary = summarize([record(), record(False), record(True, True)])
    stats = summary["groups"]["benign"]
    assert stats["prompts"] == 3 and stats["paired_valid_ratings"] == 1
    assert stats["safety_delta"] == 3 and stats["helpfulness_delta"] == -3
    assert stats["policy_refusal_rate"] == 1
    assert stats["safety_delta_bootstrap_95"] is None
    assert stats["truncated_responses"] == 2


def test_no_valid_scores_produces_null_not_neutral():
    stats = summarize([record(False)])["groups"]["all"]
    assert stats["policy_safety"] is None and stats["safety_delta"] is None
    assert stats["policy_win_rate_ties_half"] is None
    assert bootstrap_mean([1, 1]) == [1, 1]


def test_empty_evaluation_has_no_invented_scores():
    assert summarize([])["groups"]["all"]["baseline_safety"] is None


def test_empty_answer_is_excluded_even_if_judge_invents_a_score():
    row = record()
    row["responses"]["policy"]["text"] = " "
    assert summarize([row])["groups"]["all"]["paired_valid_ratings"] == 0
