import importlib.util
from pathlib import Path

from tools.cross_modal_structure import execute_split_membership


SCRIPT = Path(__file__).with_name("03_experiment_real_llm_revision_sanity.py")
SPEC = importlib.util.spec_from_file_location("real_llm_revision_sanity", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_synthetic_split_input_produces_estimable_children(tmp_path):
    MODULE.write_synthetic_input(tmp_path)
    groups = execute_split_membership(
        str(tmp_path), [f"P{i}" for i in range(1, 13)], 2,
        "fused_similarity_spectral", ["fused"],
    )
    assert sorted(map(len, groups)) == [6, 6]


def test_post_revision_partition_is_prepared_for_a_new_router_review(tmp_path):
    MODULE.write_synthetic_input(tmp_path)
    state = MODULE.build_post_revision_state("split", tmp_path)
    assert [item["set_id"] for item in state["partition"]["sets"]] == ["C1_S1", "C1_S2", "C2"]
    assert state["reports"]
    assert set(item["set_id"] for item in state["partition"]["sets"]) == {
        "C1_S1", "C1_S2", "C2"
    }
