from pathlib import Path

from agents.subtype_review.evidence_semantics import guidance_for


PROMPT_DIR = Path("agents/subtype_review/prompts")
HEADINGS = ["# Role", "# Goal", "# Rules", "# Workflow", "# Context", "# Output Format"]


def test_prompts_keep_three_agent_roles_and_shared_skeleton():
    assert {path.name for path in PROMPT_DIR.glob("*.md")} == {
        "router.md", "verifier.md", "reviser.md",
    }
    for path in PROMPT_DIR.glob("*.md"):
        assert [line for line in path.read_text(encoding="utf-8").splitlines()
                if line.startswith("# ")] == HEADINGS


def test_router_prompt_defines_continuous_action_semantics():
    text = (PROMPT_DIR / "router.md").read_text(encoding="utf-8").lower()
    for phrase in (
        "provisional comparative revision", "single-cluster-dominated union",
        "weak or mixed native membership alone is insufficient for drop",
        "no merge does not imply accept", "batch their requests",
        "available_evidence_requests", "exactly one json object",
    ):
        assert phrase in text
    assert "optional_second_set_id_for_merge" in text
    assert "membership-role conclusion" not in text


def test_verifier_prompt_only_interprets_evidence():
    text = (PROMPT_DIR / "verifier.md").read_text(encoding="utf-8").lower()
    assert "never choose or recommend" in text
    assert "do not" in text and "action recommendations" in text
    assert "role_conclusion_contract" not in text
    assert text.count("```json") == 1


def test_reviser_prompt_preserves_router_action():
    text = (PROMPT_DIR / "reviser.md").read_text(encoding="utf-8").lower()
    for phrase in ("router action is authoritative", "preserve the exact", "available_metric_refs", "provisional"):
        assert phrase in text
    assert text.count("```json") == 1


def test_pair_semantics_remain_in_scientific_guidance():
    guidance = guidance_for("cross_modal_consistency", "structural_diagnostics", "pair")
    text = guidance["scope_interpretation"].lower()
    assert "candidate_k=1" in text
    assert "merge-compatible" in text
    assert "must never" in text
    assert "higher cost means more cross-boundary connectivity" in guidance["metric_semantics"]["current_boundary_normalized_cut"].lower()


def test_prompts_do_not_encode_run_specific_configuration():
    forbidden = ("repeat1", "repeat 1", "k=2", "k=4", "k=6", "k=8", "initial_k", "multi_k")
    for path in PROMPT_DIR.glob("*.md"):
        text = path.read_text(encoding="utf-8").lower()
        assert not any(token in text for token in forbidden)
