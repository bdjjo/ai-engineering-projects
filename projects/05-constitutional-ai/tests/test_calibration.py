from calibrate_judge import CASES, calibrate
from smoke import completion


def test_invalid_judge_fails_every_calibration_case():
    def invalid(*args, **kwargs):
        return completion("unparseable")

    report = calibrate(invalid)
    assert report["total"] == len(CASES) == 8
    assert report["correct"] == 0
    assert all(r["observed"] is None for r in report["records"])
