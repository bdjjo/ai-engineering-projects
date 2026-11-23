from smoke import run_smoke


def test_complete_offline_training_path(tmp_path):
    report = run_smoke(tmp_path)
    assert report["feedback"]["statuses"] == {"accepted": 9}
    assert report["rl_adapter_changed"]
    assert report["evaluation_parser_checked"]
