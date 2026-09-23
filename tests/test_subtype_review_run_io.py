import json


def test_inspect_review_run_classifies_standard_states(tmp_path):
    from agents.subtype_review.run_io import inspect_review_run

    missing = inspect_review_run(tmp_path / "missing")
    assert missing["status"] == "missing"

    run = tmp_path / "run"
    run.mkdir()
    (run / "run_metadata.json").write_text("{broken")
    assert inspect_review_run(run)["status"] == "invalid"

    (run / "run_metadata.json").write_text(json.dumps({
        "status": "failed", "input_signature": "sig",
        "error_type": "RuntimeError", "error_message": "boom",
    }))
    assert inspect_review_run(run)["status"] == "failed"

    (run / "run_metadata.json").write_text(json.dumps({
        "status": "complete", "input_signature": "old",
    }))
    assert inspect_review_run(run, expected_input_signature="new")["status"] == "stale"
    assert inspect_review_run(run)["status"] == "incomplete"

    (run / "final_review_summary.json").write_text(json.dumps({
        "status": "review_complete", "raw_control_status": "complete",
    }))
    (run / "final_subtype_sets.json").write_text("[]")
    assert inspect_review_run(run)["status"] == "complete"
