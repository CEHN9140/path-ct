import json


def write_run(root, k, repeat, accepted_sets, signature="sig"):
    run = root / "subtype_review" / "runs" / f"K{k}" / f"repeat{repeat}"
    run.mkdir(parents=True)
    (run / "run_metadata.json").write_text(json.dumps({"status": "complete", "input_signature": signature}))
    (run / "final_review_summary.json").write_text(json.dumps({"status": "review_complete", "raw_control_status": "complete"}))
    (run / "final_subtype_sets.json").write_text(json.dumps([
        {"set_id": f"S{index}", "member_ids": sorted(members)}
        for index, members in enumerate(accepted_sets, 1)
    ]))


def test_closed_recurrent_accept_sets_preserve_observation_frequency():
    from agents.subtype_review.multi_k import build_recurrent_sets

    a = {f"a{i}" for i in range(13)}
    observations = [
        {"observation_id": f"O{i}", "initial_k": 2, "repeat": i, "member_ids": members}
        for i, members in enumerate([a, a, a - {"a11"}, a - {"a11"}, a - {"a11", "a12"}])
    ]
    records = build_recurrent_sets(observations, 10)
    by_members = {tuple(row["member_ids"]): row for row in records}
    assert by_members[tuple(sorted(a))]["occurrence_count"] == 2
    assert by_members[tuple(sorted(a - {"a11"}))]["occurrence_count"] == 4
    assert by_members[tuple(sorted(a - {"a11", "a12"}))]["occurrence_count"] == 5
    assert all(len(row["member_ids"]) >= 10 for row in records)


def test_aggregation_writes_new_outputs_and_clears_old_state_files(tmp_path):
    from agents.subtype_review.multi_k import run_multi_k_aggregation

    candidate = tmp_path / "candidate_subtype"
    candidate.mkdir()
    patients = [f"P{i:02d}" for i in range(12)]
    (candidate / "affinity_patient_order.json").write_text(json.dumps(patients))
    stale = tmp_path / "subtype_review" / "multi_k"
    stale.mkdir(parents=True)
    (stale / "state_merge_summary.json").write_text("stale")
    write_run(tmp_path, 2, 1, [set(patients)])
    result = run_multi_k_aggregation(str(tmp_path), {"multi_k": {"initial_ks": [2], "repeats": [1], "min_subtype_size": 10}}, "sig")
    assert result["status"] == "complete"
    assert result["total_accept_set_count"] == 1
    assert (stale / "accept_set_observations.csv").is_file()
    assert (stale / "recurrent_set_summary.csv").is_file()
    assert not (stale / "state_merge_summary.json").exists()


def test_aggregation_rejects_incomplete_grid_before_clearing_outputs(tmp_path):
    from agents.subtype_review.multi_k import run_multi_k_aggregation

    candidate = tmp_path / "candidate_subtype"
    candidate.mkdir()
    (candidate / "affinity_patient_order.json").write_text('["P1"]')
    try:
        run_multi_k_aggregation(str(tmp_path), {"multi_k": {"initial_ks": [2], "repeats": [1], "min_subtype_size": 10}}, "sig")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("incomplete grid must fail before aggregation")


def test_aggregation_rejects_overlapping_accept_sets_within_run(tmp_path):
    from agents.subtype_review.multi_k import run_multi_k_aggregation

    candidate = tmp_path / "candidate_subtype"
    candidate.mkdir()
    (candidate / "affinity_patient_order.json").write_text(json.dumps(["P1", "P2", "P3"]))
    write_run(tmp_path, 2, 1, [{"P1", "P2"}, {"P2", "P3"}])
    try:
        run_multi_k_aggregation(str(tmp_path), {"multi_k": {"initial_ks": [2], "repeats": [1], "min_subtype_size": 2}}, "sig")
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("overlapping accepted sets must fail")
