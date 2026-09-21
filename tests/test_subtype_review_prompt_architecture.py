from __future__ import annotations

from pathlib import Path

from agents.subtype_review.llm import review_signature_manifest


PROMPT_DIR = Path("agents/subtype_review/prompts")


def test_v11_prompt_layout_has_only_three_agent_prompts():
    assert not (PROMPT_DIR / "protocol.md").exists()
    assert {path.name for path in PROMPT_DIR.glob("*.md")} == {
        "verifier.md",
        "router.md",
        "reviser.md",
    }


def test_agent_prompts_keep_semantic_responsibilities_separate():
    verifier = (PROMPT_DIR / "verifier.md").read_text(encoding="utf-8")
    router = (PROMPT_DIR / "router.md").read_text(encoding="utf-8")
    reviser = (PROMPT_DIR / "reviser.md").read_text(encoding="utf-8")

    assert "evidence interpretation" in verifier.lower()
    assert "decide what should happen next" in router.lower()
    assert "revisionplan" in reviser.lower()
    assert "supports accept" not in verifier.lower()
    assert "should be dropped" not in verifier.lower()
    assert "rna pathway count threshold" not in router.lower()
    assert "drop decision logic" not in reviser.lower()


def test_review_signature_changes_with_prompt_or_scientific_config_only(tmp_path):
    config = {
        "llm": {
            "model_name": "deepseek-v4-flash",
            "temperature": 0,
            "max_new_tokens": 32768,
            "api_key_env": "KEY",
        },
        "prompt_dir": "agents/subtype_review/prompts",
        "cross_modal": {"permanova_permutations": 999},
        "multi_k": {
            "initial_ks": [2, 3],
            "repeats": [1, 2],
            "acceptance_threshold": 2 / 3,
        },
    }
    base = review_signature_manifest(config, "/data/qijun/path-ct/configs")
    assert set(base) == {
        "verifier_prompt_sha256",
        "router_prompt_sha256",
        "reviser_prompt_sha256",
        "review_signature",
    }

    execution_only = {**config, "repeat": 3, "output_root": str(tmp_path / "run3")}
    assert review_signature_manifest(execution_only, "/data/qijun/path-ct/configs")["review_signature"] == base["review_signature"]

    changed_grid = {**config, "multi_k": {**config["multi_k"], "initial_ks": [4], "repeats": [3]}}
    assert review_signature_manifest(changed_grid, "/data/qijun/path-ct/configs")["review_signature"] == base["review_signature"]

    changed_multi_k_analysis = {**config, "multi_k": {**config["multi_k"], "acceptance_threshold": 0.7}}
    assert review_signature_manifest(changed_multi_k_analysis, "/data/qijun/path-ct/configs")["review_signature"] != base["review_signature"]

    changed_prompt = (PROMPT_DIR / "router.md").read_text(encoding="utf-8")
    (tmp_path / "prompts").mkdir()
    for name in ("verifier.md", "router.md", "reviser.md"):
        source = PROMPT_DIR / name
        (tmp_path / "prompts" / name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "prompts" / "router.md").write_text(changed_prompt + "\nV11 test change\n", encoding="utf-8")
    changed_prompt_config = {**config, "prompt_dir": str(tmp_path / "prompts")}
    assert review_signature_manifest(changed_prompt_config, "/data/qijun/path-ct/configs")["review_signature"] != base["review_signature"]

    changed_config = {**config, "cross_modal": {"permanova_permutations": 1000}}
    assert review_signature_manifest(changed_config, "/data/qijun/path-ct/configs")["review_signature"] != base["review_signature"]


def test_router_reason_must_match_selected_action():
    router = (PROMPT_DIR / "router.md").read_text(encoding="utf-8").lower()
    assert "reason must directly justify the selected action" in router
    assert "must not state or imply that a different action is better supported" in router


def test_router_requires_explicit_decision_state_fields():
    router = (PROMPT_DIR / "router.md").read_text(encoding="utf-8")

    assert "decision_state" in router
    assert "unassessed" in router
    assert "alternative_explanation" in router
    assert "no predetermined acquisition order" in router.lower()
    assert "do not request evidence merely because a dimension is unassessed" in router.lower()
    assert "available_evidence_requests" in router
    assert "`actions` must be empty" in router.lower()
    assert "`evidence_requests` must be empty" in router.lower()


def test_router_defines_decision_state_semantics():
    router = (PROMPT_DIR / "router.md").read_text(encoding="utf-8").lower()

    assert "supported` means affirmative evidence supports a coherent biological identity" in router
    assert "compatible` means affirmative and sufficiently coherent structural evidence supports" in router
    assert "uncertain` means the current representation remains plausible" in router
    assert "concerning` means available evidence supports a plausible substantial explanation" in router
    assert "weak overlap does not prove novelty" in router


def test_router_separates_affirmative_structure_from_plausibility():
    router = (PROMPT_DIR / "router.md").read_text(encoding="utf-8").lower()

    assert "without a material structural caveat" in router
    assert "mere absence of evidence for incompatibility is not sufficient" in router
    assert "evidence is weak, mixed, or otherwise insufficient" in router
    assert "or shows them reasonably defensible" not in router
    assert "membership, boundaries, and granularity" in router
    assert "prefer the supported structural revision over dropping" in router


def test_verifier_does_not_call_nonsignificant_trends_corroboration():
    verifier = (PROMPT_DIR / "verifier.md").read_text(encoding="utf-8").lower()

    assert "nonsignificant finding may be described as a directionally consistent trend" in verifier
    assert "direction alone must not be described as affirmative corroboration" in verifier


def test_router_requests_questions_while_verifier_selects_tools():
    router = (PROMPT_DIR / "router.md").read_text(encoding="utf-8").lower()
    verifier = (PROMPT_DIR / "verifier.md").read_text(encoding="utf-8").lower()

    assert "available_aspects" not in router
    assert "unresolved scientific question" in router
    assert "the verifier decides which eligible tool" in router
    assert "call at most one tool" in verifier
