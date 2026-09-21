import json
import re
from pathlib import Path

from agents.subtype_review.evidence_semantics import METRIC_SEMANTICS, guidance_for
from agents.subtype_review.llm import review_signature_manifest
from agents.subtype_review.schemas import EvidenceReportBatch, RevisionPlan, RouterPlan


PROMPT_DIR = Path("agents/subtype_review/prompts")


def test_prompt_layout_and_agent_responsibilities_are_separate():
    assert not (PROMPT_DIR / "protocol.md").exists()
    assert {path.name for path in PROMPT_DIR.glob("*.md")} == {
        "verifier.md", "router.md", "reviser.md",
    }
    verifier = (PROMPT_DIR / "verifier.md").read_text(encoding="utf-8").lower()
    router = (PROMPT_DIR / "router.md").read_text(encoding="utf-8").lower()
    reviser = (PROMPT_DIR / "reviser.md").read_text(encoding="utf-8").lower()
    assert "sole scientific interpreter" in verifier
    assert "interpret structural measurements scientifically" in verifier
    assert "never choose accept, drop, split, or merge" in verifier
    assert "determine whether each current candidate merits continued treatment" in router
    assert "revisionplan" in reviser
    assert "do not reinterpret raw tool measurements" in router


def test_all_agent_prompts_share_the_six_section_skeleton():
    headings = ("# Role", "# Goal", "# Rules", "# Workflow", "# Context", "# Output Format")
    for name in ("router.md", "verifier.md", "reviser.md"):
        prompt = (PROMPT_DIR / name).read_text(encoding="utf-8")
        assert [line for line in prompt.splitlines() if line.startswith("# ")] == list(headings), name


def test_agent_prompts_are_independent_of_run_configuration_and_observed_cases():
    prompts = [
        (PROMPT_DIR / name).read_text(encoding="utf-8").lower()
        for name in ("router.md", "verifier.md", "reviser.md")
    ]
    forbidden = (
        "repeat1", "repeat 1", "repeat 2", "repeat 3", "k=2", "k=4", "k=6", "k=8",
        "initial_k", "initial-k", "multi_k", "multi-k", "seed", "c2-c4",
    )
    for prompt in prompts:
        for token in forbidden:
            assert token not in prompt
    assert "merge_allowed" not in prompts[0]
    assert "allowed_structural_actions" in prompts[0]


def test_review_signature_changes_with_prompts_and_scientific_config(tmp_path):
    config = {
        "llm": {"model_name": "deepseek-v4-flash", "temperature": 0,
                "max_new_tokens": 32768, "api_key_env": "KEY"},
        "prompt_dir": "agents/subtype_review/prompts",
        "cross_modal": {"permanova_permutations": 999},
        "multi_k": {"initial_ks": [2, 3], "repeats": [1, 2]},
    }
    base = review_signature_manifest(config, "/data/qijun/path-ct/configs")
    assert set(base) == {
        "verifier_prompt_sha256", "router_prompt_sha256", "reviser_prompt_sha256",
        "review_signature",
    }
    execution_only = {**config, "repeat": 3, "output_root": str(tmp_path / "run3")}
    assert review_signature_manifest(execution_only, "/data/qijun/path-ct/configs")["review_signature"] == base["review_signature"]
    changed_config = {**config, "cross_modal": {"permanova_permutations": 1000}}
    assert review_signature_manifest(changed_config, "/data/qijun/path-ct/configs")["review_signature"] != base["review_signature"]

    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    for name in ("verifier.md", "router.md", "reviser.md"):
        source = PROMPT_DIR / name
        (prompt_dir / name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    router_path = prompt_dir / "router.md"
    router_path.write_text(router_path.read_text(encoding="utf-8") + "\nPrompt change.\n", encoding="utf-8")
    changed_prompt = {**config, "prompt_dir": str(prompt_dir)}
    assert review_signature_manifest(changed_prompt, "/data/qijun/path-ct/configs")["review_signature"] != base["review_signature"]


def test_router_prompt_is_compact_and_keeps_metric_semantics_out():
    router = (PROMPT_DIR / "router.md").read_text(encoding="utf-8")
    assert len(router.split()) < 2000
    assert len(router.splitlines()) < 260
    assert [line for line in router.splitlines() if line.startswith("# ")] == [
        "# Role", "# Goal", "# Rules", "# Workflow", "# Context", "# Output Format",
    ]
    lowered = router.lower()
    for metric in (
        "screen_candidate_k", "candidate_eigengap", "union_eigengap",
        "boundary_silhouette", "mean_between_affinity", "cramers_v",
        "permanova", "q_global", "q_driver", "grv", "nes",
    ):
        assert metric not in lowered
    assert "# role" in lowered and "# goal" in lowered and "# rules" in lowered
    assert "# workflow" in lowered and "# context" in lowered and "# output format" in lowered


def test_router_requires_candidate_specific_independence_without_tool_gate():
    router = (PROMPT_DIR / "router.md").read_text(encoding="utf-8").lower()
    assert "independent retention needs a candidate-specific evidential basis" in router
    assert "partition-level structural screening cannot serve as the positive candidate-specific basis" in router
    assert "no particular tool or report type is mandatory" in router
    assert "strong correspondence does not independently validate a candidate" in router
    assert "weak correspondence does not establish novelty" in router


def test_output_format_examples_match_pydantic_contracts():
    router = (PROMPT_DIR / "router.md").read_text(encoding="utf-8")
    for phrase in (
        "Return exactly one valid JSON object",
        "The top-level object contains exactly two keys",
        "Exactly one output mode is allowed",
        "Evidence-request mode", "Structural-revision mode", "Terminal mode",
        "Action object keys are exactly", "request keys are exactly",
        "Use double quotes", "Do not use trailing commas",
        "Do not wrap the JSON object in markdown code fences",
        "reason` is a valid JSON string that directly justifies its action",
    ):
        assert phrase.lower() in router.lower()
    for field in (
        "`action`", "`target_ids`", "`n_children`", "`evidence_report_refs`",
        "`reason`", "`dimension`", "`scope`", "`question`",
    ):
        assert field in router
    assert "split uses one target and an integer child count" in router.lower()
    assert "JSON `null`" in router

    router_examples = [RouterPlan.model_validate(json.loads(block)) for block in re.findall(r"```json\s*(.*?)\s*```", router, re.S)]
    verifier = (PROMPT_DIR / "verifier.md").read_text(encoding="utf-8")
    verifier_examples = [EvidenceReportBatch.model_validate(json.loads(block)) for block in re.findall(r"```json\s*(.*?)\s*```", verifier, re.S)]
    reviser = (PROMPT_DIR / "reviser.md").read_text(encoding="utf-8")
    reviser_examples = [RevisionPlan.model_validate(json.loads(block)) for block in re.findall(r"```json\s*(.*?)\s*```", reviser, re.S)]
    assert len(router_examples) == 3
    assert len(verifier_examples) == 1
    assert len(reviser_examples) == 2


def test_structural_guidance_distinguishes_partition_set_and_pair():
    partition = guidance_for("cross_modal_consistency", "structural_diagnostics", "partition")
    set_scope = guidance_for("cross_modal_consistency", "structural_diagnostics", "set")
    pair = guidance_for("cross_modal_consistency", "structural_diagnostics", "pair")
    assert "triage evidence" in partition["scope_interpretation"].lower()
    assert "not positive evidence" in set_scope["scope_interpretation"].lower()
    assert "compatible with treating the pair as one candidate" in pair["scope_interpretation"].lower()
    assert "not sufficient by itself to justify merge" in pair["scope_interpretation"].lower()
    assert "must never be interpreted as evidence against merge" in pair["scope_interpretation"].lower()
    assert "does not establish that it corresponds to the current pair labels" in pair["scope_interpretation"].lower()
    assert "when the current partition contains exactly two sets" in METRIC_SEMANTICS[("biological_support", "rna_pathway_enrichment")]["contrast_interpretation"].lower()
    assert "multi-set partition interpretation" in METRIC_SEMANTICS[("cross_modal_consistency", "affinity_geometry_concordance")]
    all_semantics = str(METRIC_SEMANTICS).lower()
    assert "k=2" not in all_semantics
    assert "k>2" not in all_semantics


def test_verifier_pair_semantics_and_mode_contracts():
    verifier = (PROMPT_DIR / "verifier.md").read_text(encoding="utf-8").lower()
    assert "evidence against merge" in verifier
    assert "this finding alone does not recommend merge" in verifier
    assert "## selection mode" in verifier
    assert "## audit mode" in verifier
    assert "do not provide substantive ordinary text, json, or an evidence report" in verifier
    assert "do not answer the evidencerequest itself" in verifier
    assert "## audit mode" in verifier


def test_verifier_context_and_select_audit_outputs_match_runtime_contracts():
    verifier = (PROMPT_DIR / "verifier.md").read_text(encoding="utf-8").lower()
    for field in (
        "mode", "partition", "evidence_request", "current_evidence", "attempted_tools",
        "remaining_tools", "require_tool", "required_reports", "prior_reports",
        "round_evidence", "round", "wave", "evidence_guidance",
    ):
        assert field in verifier
    assert "tool-call presence" in verifier or "presence or absence of a tool call" in verifier
    assert 'with only the top-level key `reports`' in verifier
    for field in (
        "dimension", "aspect", "scope", "target_ids", "observations",
        "dimension_interpretation", "cross_evidence_context", "limitations",
        "tool_refs", "metric_refs",
    ):
        assert f"`{field}`" in verifier
    assert "tool_refs` and `metric_refs` must be empty arrays" in verifier


def test_reviser_output_contract_matches_revision_plan_examples():
    reviser = (PROMPT_DIR / "reviser.md").read_text(encoding="utf-8").lower()
    for field in ("split_plans", "merge_plans", "rationale", "action", "target_id",
                  "target_ids", "n_children", "structural_basis", "execution_strategy",
                  "metric_refs"):
        assert f"`{field}`" in reviser
    assert "exactly one valid json object" in reviser
    assert "no markdown" in reviser


def test_router_requests_questions_while_verifier_selects_tools():
    router = (PROMPT_DIR / "router.md").read_text(encoding="utf-8").lower()
    verifier = (PROMPT_DIR / "verifier.md").read_text(encoding="utf-8").lower()
    assert "available_aspects" not in router
    assert "unresolved scientific question" in router
    assert "the verifier selects which currently eligible tool" in router
    assert "call at most one tool" in verifier
