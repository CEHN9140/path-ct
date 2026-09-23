import os
import json
from pathlib import Path


def test_file_content_identity_ignores_mtime_and_detects_content(tmp_path):
    from utils.cache_utils import file_content_identity

    path = tmp_path / "input.txt"
    path.write_text("abc", encoding="utf-8")
    first = file_content_identity(path)
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000))
    assert file_content_identity(path) == first

    path.write_text("abd", encoding="utf-8")
    assert file_content_identity(path) != first


def test_review_input_signature_is_stable_for_same_content_rewrite(tmp_path):
    from agents.subtype_review.runner import build_review_input_signature

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    evidence = tmp_path / "rna.csv"
    evidence.write_text("gene,value\nG1,1\n", encoding="utf-8")
    labels = tmp_path / "labels.csv"
    labels.write_text("label\nA\n", encoding="utf-8")
    hallmark = tmp_path / "hallmark.gmt"
    hallmark.write_text("H\tdesc\tG1\n", encoding="utf-8")
    (config_dir / "subtype_review.yaml").write_text(
        f"known_label_echo:\n  mrna_m1_m4_path: {labels}\n"
        f"  clearcode34_path: {labels}\n"
        f"rna:\n  hallmark_gene_sets_path: {hallmark}\n"
        f"prompt_dir: {Path(__file__).resolve().parents[1] / 'agents/subtype_review/prompts'}\n",
        encoding="utf-8",
    )
    state = {
        "omics_evidence": {"rna_pathway_feature_path": str(evidence)},
        "inventory": {"Clinical": {"stage": "I"}},
    }
    output_root = tmp_path / "output"
    (output_root / "ct_qc").mkdir(parents=True)
    metadata = output_root / "ct_qc" / "P1.json"
    metadata.write_text(json.dumps({"qc": "success"}), encoding="utf-8")
    kwargs = {
        "candidate_signature": "candidate",
        "patient_states_by_id": {"P1": state},
        "output_root": str(output_root),
        "config_dir": str(config_dir),
    }
    first, _ = build_review_input_signature(**kwargs)
    metadata.write_text(json.dumps({"qc": "success"}), encoding="utf-8")
    second, _ = build_review_input_signature(**kwargs)
    assert first == second


def test_review_signature_uses_content_for_configured_input_paths(tmp_path):
    from agents.subtype_review.llm import review_signature_manifest

    first_root = tmp_path / "one"
    second_root = tmp_path / "two"
    first_root.mkdir()
    second_root.mkdir()
    for root in (first_root, second_root):
        (root / "labels.csv").write_text("label\nA\n", encoding="utf-8")
        (root / "hallmark.gmt").write_text("H\tdesc\tG1\n", encoding="utf-8")
    prompt_dir = Path(__file__).resolve().parents[1] / "agents/subtype_review/prompts"
    first = {
        "known_label_echo": {"mrna_m1_m4_path": str(first_root / "labels.csv")},
        "rna": {"hallmark_gene_sets_path": str(first_root / "hallmark.gmt")},
        "prompt_dir": str(prompt_dir),
    }
    second = {
        "known_label_echo": {"mrna_m1_m4_path": str(second_root / "labels.csv")},
        "rna": {"hallmark_gene_sets_path": str(second_root / "hallmark.gmt")},
        "prompt_dir": str(prompt_dir),
    }
    assert review_signature_manifest(first, tmp_path) == review_signature_manifest(second, tmp_path)
