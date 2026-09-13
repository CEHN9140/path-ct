import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).with_name("04_experiment_full_revision_evidence_loop.py")
SPEC = importlib.util.spec_from_file_location("full_revision_evidence_loop", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_revision_input_does_not_preload_post_revision_evidence(tmp_path):
    MODULE.write_synthetic_input(tmp_path)
    state = MODULE.build_initial_state("split")
    assert state["evidence_memory"] == {}


def test_synthetic_registry_preserves_real_tool_contracts():
    registry = MODULE.synthetic_registry()
    assert set(registry) >= {
        "pathway_enrichment",
        "mutation_enrichment",
        "cnv_characterization",
        "multimodal_consistency_check",
        "confound_test",
        "known_label_echo_test",
    }
    assert all(callable(metadata["function"]) for metadata in registry.values())


def test_full_loop_summary_requires_reacquisition():
    summary = MODULE.summarize_state(
        "split",
        {
            "control": {"status": "complete", "trace": [
                {"node": "reviser", "event": "revision_applied"},
                {"node": "router", "event": "decision", "plan": {
                    "evidence_requests": [{"dimension": "biological_support"}]
                }},
                {"node": "verifier", "event": "tool_selection"},
                {"node": "verifier", "event": "reports"},
                {"node": "router", "event": "decision", "plan": {"actions": []}},
            ]},
            "revision_result": {"new_partition_signature": "x"},
            "partition": {"sets": []},
            "reports": [{"partition_signature": MODULE.partition_signature([])}],
        },
        "initial-signature",
    )
    assert summary["verifier_reacquired_after_revision"] is True
    assert summary["full_loop_verified"] is True


def test_full_loop_summary_is_false_without_revision():
    summary = MODULE.summarize_state(
        "split",
        {
            "control": {"status": "complete", "trace": [
                {"node": "verifier", "event": "tool_selection"},
                {"node": "verifier", "event": "reports"},
            ]},
            "revision_result": {"new_partition_signature": "x"},
            "partition": {"sets": []},
        },
        "initial-signature",
    )
    assert summary["verifier_reacquired_after_revision"] is False
    assert summary["full_loop_verified"] is False


def test_reviser_uses_only_metric_refs_for_structural_evidence():
    prompt = (Path(__file__).parents[1] / "agents/subtype_review/prompts/reviser.md").read_text()
    assert "schema field `metric_refs`" in prompt
    assert "structural_metric_refs" in prompt
