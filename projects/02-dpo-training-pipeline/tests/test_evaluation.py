import pytest
from score_judgments import summarize


def test_blind_mapping_and_ties():
    key = {"1": ["sft", "dpo"], "2": ["dpo", "sft"], "3": ["sft", "dpo"]}
    ratings = [{"id": "1", "winner": "B"}, {"id": "2", "winner": "B"}, {"id": "3", "winner": "tie"}]
    result = summarize(ratings, key)
    assert result["tie_adjusted_dpo_win_rate"] == 0.5
    assert result["dpo_wins"] == result["sft_wins"] == result["ties"] == 1


@pytest.mark.parametrize(
    "ratings", [[], [{"id": "1", "winner": ""}], [{"id": "wrong", "winner": "A"}]]
)
def test_incomplete_ratings_rejected(ratings):
    with pytest.raises(ValueError):
        summarize(ratings, {"1": ["sft", "dpo"]})
