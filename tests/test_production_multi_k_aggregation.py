import json


def write_run(root, k, repeat, accepted_sets, signature="sig", candidate_signature=None):
    run = root / "subtype_review" / "runs" / f"K{k}" / f"repeat{repeat}"
    run.mkdir(parents=True)
    metadata = {"status": "complete", "input_signature": signature}
    if candidate_signature is not None:
        metadata["candidate_signature"] = candidate_signature
    (run / "run_metadata.json").write_text(json.dumps(metadata))
    (run / "final_review_summary.json").write_text(json.dumps({"status": "review_complete", "raw_control_status": "complete"}))
    (run / "final_subtype_sets.json").write_text(json.dumps([
        {"set_id": f"S{index}", "member_ids": sorted(members)}
        for index, members in enumerate(accepted_sets, 1)
    ]))


def test_summary_scans_configured_grid_and_classifies_stale_and_failed(tmp_path):
    from agents.subtype_review.runner import summarize_review_grid

    write_run(tmp_path, 2, 1, [{"P1"}], signature="active")
    failed = tmp_path / "subtype_review" / "runs" / "K2" / "repeat2"
    failed.mkdir(parents=True)
    (failed / "run_metadata.json").write_text(json.dumps({
        "status": "failed", "input_signature": "active",
        "error_type": "RuntimeError", "error_message": "boom",
    }))
    stale = tmp_path / "subtype_review" / "runs" / "K3" / "repeat1"
    stale.mkdir(parents=True)
    (stale / "run_metadata.json").write_text(json.dumps({
        "status": "complete", "input_signature": "old",
    }))
    summary = summarize_review_grid(
        output_root=str(tmp_path),
        config={"multi_k": {"initial_ks": [2, 3], "repeats": [1, 2]}},
        active_input_signature="active",
    )
    assert summary["configured_run_count"] == 4
    assert summary["complete_run_count"] == 1
    assert summary["failed_run_count"] == 1
    assert summary["stale_run_count"] == 1
    assert summary["missing_run_count"] == 1


def test_multi_k_cli_reads_completed_runs_without_active_experiment(tmp_path, monkeypatch):
    import agents.subtype_review.multi_k as multi_k

    candidate = tmp_path / "candidate_subtype"
    candidate.mkdir()
    (candidate / "affinity_patient_order.json").write_text('["P1", "P2"]')
    write_run(tmp_path, 2, 1, [{"P1", "P2"}], signature="active")
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "subtype_review.yaml").write_text(
        "multi_k:\n  initial_ks: [2]\n  repeats: [1]\n  min_subtype_size: 2\n"
    )
    monkeypatch.setattr(
        "sys.argv", ["multi_k", "--output-root", str(tmp_path), "--config-dir", str(config_dir)]
    )
    multi_k.main()


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
    result = run_multi_k_aggregation(str(tmp_path), {"multi_k": {"initial_ks": [2], "repeats": [1], "min_subtype_size": 10}})
    assert result["status"] == "complete"
    assert result["total_accept_set_count"] == 1
    assert (stale / "accept_set_observations.csv").is_file()
    assert (stale / "recurrent_set_summary.csv").is_file()
    assert not (stale / "state_merge_summary.json").exists()


def test_aggregation_rejects_when_no_usable_run_exists(tmp_path):
    from agents.subtype_review.multi_k import run_multi_k_aggregation

    candidate = tmp_path / "candidate_subtype"
    candidate.mkdir()
    (candidate / "affinity_patient_order.json").write_text('["P1"]')
    try:
        run_multi_k_aggregation(str(tmp_path), {"multi_k": {"initial_ks": [2], "repeats": [1], "min_subtype_size": 10}})
    except ValueError as exc:
        assert "No usable completed Agent runs" in str(exc)
    else:
        raise AssertionError("incomplete grid must fail before aggregation")


def test_aggregation_rejects_overlapping_accept_sets_within_run(tmp_path):
    from agents.subtype_review.multi_k import run_multi_k_aggregation

    candidate = tmp_path / "candidate_subtype"
    candidate.mkdir()
    (candidate / "affinity_patient_order.json").write_text(json.dumps(["P1", "P2", "P3"]))
    write_run(tmp_path, 2, 1, [{"P1", "P2"}, {"P2", "P3"}])
    try:
        run_multi_k_aggregation(str(tmp_path), {"multi_k": {"initial_ks": [2], "repeats": [1], "min_subtype_size": 2}})
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("overlapping accepted sets must fail")


def test_aggregation_excludes_unusable_runs_and_records_audit(tmp_path):
    from agents.subtype_review.multi_k import run_multi_k_aggregation

    candidate = tmp_path / "candidate_subtype"
    candidate.mkdir()
    (candidate / "affinity_patient_order.json").write_text(json.dumps(["P1", "P2"]))
    write_run(tmp_path, 4, 2, [{"P1", "P2"}])
    result = run_multi_k_aggregation(str(tmp_path), {
        "multi_k": {"initial_ks": [2, 4, 8], "repeats": [1, 2, 3], "min_subtype_size": 2},
    })
    assert result["status"] == "complete"
    manifest = json.loads(
        (tmp_path / "subtype_review" / "multi_k" / "aggregation_manifest.json").read_text()
    )
    assert manifest["configured_initial_ks"] == [2, 4, 8]
    assert manifest["configured_repeats"] == [1, 2, 3]
    assert manifest["included_initial_ks"] == [4]
    assert manifest["included_repeats"] == [2]
    assert manifest["included_runs"] == [{"initial_k": 4, "repeat": 2}]
    assert manifest["run_audit"]
    assert len(manifest["run_audit"]) == 9
    assert sum(row["status"] == "complete" for row in manifest["run_audit"]) == 1
    assert manifest["configured_run_count"] == 9
    assert manifest["included_run_count"] == 1
    assert manifest["excluded_run_count"] == 8
    assert manifest["usable_runs_by_k"] == {"2": 0, "4": 1, "8": 0}


def test_inspect_review_run_excludes_corrupt_json(tmp_path):
    from agents.subtype_review.run_io import inspect_review_run

    run = tmp_path / "K2" / "repeat1"
    run.mkdir(parents=True)
    (run / "run_metadata.json").write_text("{broken", encoding="utf-8")
    detail = inspect_review_run(run)
    assert detail["status"] == "invalid"
    assert detail["reason"] == "invalid_run_metadata_json"


def test_aggregation_rejects_mixed_candidate_signatures(tmp_path):
    from agents.subtype_review.multi_k import run_multi_k_aggregation

    candidate = tmp_path / "candidate_subtype"
    candidate.mkdir()
    (candidate / "affinity_patient_order.json").write_text('["P1", "P2"]')
    write_run(tmp_path, 2, 1, [{"P1", "P2"}], candidate_signature="candidate-a")
    write_run(tmp_path, 3, 1, [{"P1", "P2"}], candidate_signature="candidate-b")
    try:
        run_multi_k_aggregation(tmp_path.as_posix(), {
            "multi_k": {"initial_ks": [2, 3], "repeats": [1], "min_subtype_size": 2},
        })
    except ValueError as exc:
        assert "multiple candidate signatures" in str(exc)
    else:
        raise AssertionError("mixed candidate signatures must be rejected")


def core_record(recurrent_set_id, members, occurrence_count, supporting_k_count):
    return {
        "recurrent_set_id": recurrent_set_id,
        "member_ids": list(members), "member_count": len(members),
        "occurrence_count": occurrence_count, "supporting_k_count": supporting_k_count,
        "total_accept_set_count": 20, "accept_set_frequency": occurrence_count / 20,
        "supporting_run_count": occurrence_count, "usable_run_count": 10,
        "run_support_frequency": occurrence_count / 10,
        "supporting_ks": list(range(1, supporting_k_count + 1)),
        "supporting_runs": [], "supporting_observation_ids": [],
    }


def test_overlap_coefficient_and_ranked_core_selection_are_deterministic():
    from agents.subtype_review.multi_k import overlap_coefficient, select_stable_cores

    assert overlap_coefficient(set(range(10)), set(range(9)) | {20}) == (9, 0.9)
    assert overlap_coefficient(set(range(10)), set(range(25))) == (10, 1.0)
    records = [
        core_record("RS003", range(20, 30), 3, 3),
        core_record("RS001", range(10), 5, 2),
        core_record("RS002", list(range(9)) + [30], 4, 4),
        core_record("RS004", range(40, 50), 1, 1),
    ]
    cores, mapping = select_stable_cores(
        records, min_occurrences=2, min_supporting_ks=2, overlap_threshold=0.8
    )
    assert [core["representative_recurrent_set_id"] for core in cores] == ["RS001", "RS003"]
    assert next(row for row in mapping if row["recurrent_set_id"] == "RS002")["status"] == "redundant_variant"
    assert next(row for row in mapping if row["recurrent_set_id"] == "RS004")["reason"] == "occurrence_and_k_support_below_minimum"


def test_core_selection_does_not_chain_through_variants():
    from agents.subtype_review.multi_k import select_stable_cores

    records = [
        core_record("RS001", range(10), 5, 3),
        core_record("RS002", list(range(8)) + [20, 21], 4, 3),
        core_record("RS003", list(range(6)) + [30, 31, 32, 33], 3, 3),
    ]
    cores, mapping = select_stable_cores(
        records, min_occurrences=2, min_supporting_ks=2, overlap_threshold=0.8
    )
    assert [core["representative_recurrent_set_id"] for core in cores] == ["RS001", "RS003"]
    assert next(row for row in mapping if row["recurrent_set_id"] == "RS002")["status"] == "redundant_variant"


def test_aggregation_writes_core_outputs_and_small_run_audit(tmp_path):
    from agents.subtype_review.multi_k import run_multi_k_aggregation

    candidate = tmp_path / "candidate_subtype"
    candidate.mkdir()
    patients = [f"P{i:02d}" for i in range(12)]
    (candidate / "affinity_patient_order.json").write_text(json.dumps(patients))
    write_run(tmp_path, 2, 1, [set(patients)])
    result = run_multi_k_aggregation(str(tmp_path), {
        "multi_k": {"initial_ks": [2], "repeats": [1], "min_subtype_size": 10},
    })
    output = tmp_path / "subtype_review" / "multi_k"
    assert result["analysis_type"] == "patient_recurrence_subtype_clustering"
    assert (output / "stable_core_subtypes.json").is_file()
    assert (output / "stable_core_summary.csv").is_file()
    assert (output / "recurrent_set_core_map.csv").is_file()
    assert (output / "stable_core_pair_overlap.csv").is_file()
    assert (output / "patient_recurrence_subtypes.json").is_file()
    assert (output / "patient_recurrence_summary.csv").is_file()
    assert (output / "patient_recurrence_membership.csv").is_file()
    assert (output / "patient_recurrence_pair_similarity.csv").is_file()
    manifest = json.loads((output / "aggregation_manifest.json").read_text())
    assert all("summary" not in row for row in manifest["run_audit"])


def test_patient_recurrence_complete_linkage_blocks_chaining():
    from agents.subtype_review.multi_k import select_patient_recurrence_subtypes

    patient_occurrences = {
        "A": set(range(10)),
        "B": set(range(9)) | {10},
        "C": set(range(8)) | {10, 11},
    }
    observations = []
    for index in range(12):
        members = {
            patient_id for patient_id, occurrence_ids in patient_occurrences.items()
            if index in occurrence_ids
        }
        observations.append({
            "observation_id": f"S{index}", "initial_k": 2, "repeat": index,
            "member_ids": members,
        })
    subtypes, _ = select_patient_recurrence_subtypes(
        observations, ["A", "B", "C"], similarity_threshold=0.9,
        min_subtype_size=2, min_common_occurrences=1, min_supporting_ks=1,
    )
    assert [row["member_ids"] for row in subtypes] == [["A", "B"]]
