import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).with_name("02_experiment_revision_sanity.py")
SPEC = importlib.util.spec_from_file_location("revision_sanity", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_mock_split_and_merge_complete_a_new_partition_review():
    for name in ("split", "merge"):
        result = MODULE.run_mock_case(name)
        assert result["status"] == "complete"
        assert result["router_calls"] == 2
        assert result["revision_applied"] is True
        assert result["partition_changed"] is True
        assert result["revisited_after_revision"] is True


def test_split_children_and_merge_union_are_deterministic():
    split = MODULE.run_mock_case("split")
    merge = MODULE.run_mock_case("merge")
    assert split["current_set_ids"] == ["C1_S1", "C1_S2", "C2"]
    assert merge["current_set_ids"] == ["C1_M_C2"]
