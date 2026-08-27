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
    assert "action selection" in router.lower()
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
