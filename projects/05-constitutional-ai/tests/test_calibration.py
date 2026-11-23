from calibrate_judge import CASES, calibrate
from smoke import completion

from cai.structured_judge import judge_pair_structured, parse_decision


def test_invalid_judge_fails_every_calibration_case():
    def invalid(*args, **kwargs):
        return completion("unparseable")

    report = calibrate(invalid)
    assert report["total"] == len(CASES) == 8
    assert report["correct"] == 0
    assert all(r["observed"] is None for r in report["records"])


def test_structured_parser_and_order_mapping():
    assert parse_decision('{"reason":"correct", "decision":"A"}') == "A"
    assert parse_decision('{"reason":"", "decision":"A"}') is None
    assert parse_decision("A") is None

    def judge(prompt, seed=0):
        return completion(
            '{"reason":"Four is correct", "decision":"'
            + ("A" if '"A": "Four."' in prompt else "B")
            + '"}'
        )

    result = judge_pair_structured(judge, "two plus two", "Four.", "Five.", "C13")
    assert result["status"] == "accepted" and result["winner"] == "original"


def test_structured_invalid_outputs_do_not_become_ties():
    def invalid(*args, **kwargs):
        return completion("not json")

    report = calibrate(invalid, "structured")
    assert report["total"] == 12 and report["correct"] == 0


def test_single_fenced_json_is_accepted_but_extra_content_is_rejected():
    body = '{"reason":"correct numerical answer", "decision":"A"}'
    assert parse_decision("```json\n" + body + "\n```") == "A"
    assert parse_decision("```\n" + body + "\n```") == "A"
    assert parse_decision("Here is my answer: ```json\n" + body + "\n```") is None
    assert parse_decision("```json\n" + body + "\n```\n```json\n" + body + "\n```") is None
    assert parse_decision('{"reason":"ok", "decision":["A"]}') is None
