from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_scope_records(output_root: Path) -> list[dict[str, Any]]:
    records = []
    for qc_path in sorted((output_root / "wsi_qc").glob("*/selection_summary.json")):
        case_id = qc_path.parent.name
        qc = read_json(qc_path)
        selected = dict(qc.get("selected_slide") or {})
        patch_path = output_root / "wsi_patch" / f"{case_id}.json"
        tumor_seg_path = output_root / "wsi_tumor_seg" / f"{case_id}.json"
        embedding_path = output_root / "wsi_embeddings" / f"{case_id}.json"
        if not patch_path.is_file() or not tumor_seg_path.is_file() or not embedding_path.is_file():
            continue
        patch = read_json(patch_path)
        tumor_seg = read_json(tumor_seg_path)
        embedding = read_json(embedding_path)
        patch_artifacts = dict(patch.get("tool_result", {}).get("artifacts", {}) or {})
        tumor_artifacts = dict(tumor_seg.get("tool_result", {}).get("artifacts", {}) or {})
        embedding_artifacts = dict(embedding.get("tool_result", {}).get("artifacts", {}) or {})
        records.append(
            {
                "case_id": case_id,
                "slide_path": str(selected.get("selected_slide_path", "") or ""),
                "patch_dir": str(patch_artifacts.get("patch_dir", "") or ""),
                "clean_coordinates_h5_path": str(patch_artifacts.get("coordinates_h5_path", "") or ""),
                "tumor_coordinates_h5_path": str(tumor_artifacts.get("tumor_coordinates_h5_path", "") or ""),
                "tumor_embedding_path": str(embedding_artifacts.get("slide_embedding_npy_path", "") or ""),
            }
        )
    return records


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def compare_embeddings(
    records: list[dict[str, Any]],
    clean_embeddings: dict[str, np.ndarray],
) -> dict[str, Any]:
    tumor = []
    clean = []
    case_ids = []
    for record in records:
        case_id = record["case_id"]
        tumor_path = Path(record["tumor_embedding_path"])
        if case_id not in clean_embeddings or not tumor_path.is_file():
            continue
        tumor.append(np.asarray(np.load(tumor_path), dtype=float).reshape(-1))
        clean.append(np.asarray(clean_embeddings[case_id], dtype=float).reshape(-1))
        case_ids.append(case_id)
    if len(tumor) < 2:
        return {"case_count": len(tumor), "embedding_cosine_rows": [], "affinity_correlation": None}
    tumor_matrix = np.asarray(tumor)
    clean_matrix = np.asarray(clean)
    tumor_matrix /= np.maximum(np.linalg.norm(tumor_matrix, axis=1, keepdims=True), 1e-12)
    clean_matrix /= np.maximum(np.linalg.norm(clean_matrix, axis=1, keepdims=True), 1e-12)
    tumor_affinity = tumor_matrix @ tumor_matrix.T
    clean_affinity = clean_matrix @ clean_matrix.T
    upper = np.triu_indices(len(case_ids), 1)
    tumor_values = tumor_affinity[upper]
    clean_values = clean_affinity[upper]
    correlation = float(np.corrcoef(tumor_values, clean_values)[0, 1]) if len(tumor_values) > 1 else None
    return {
        "case_count": len(case_ids),
        "embedding_cosine_rows": [
            {"case_id": case_id, "tumor_clean_cosine": cosine(tumor[index], clean[index])}
            for index, case_id in enumerate(case_ids)
        ],
        "affinity_correlation": correlation,
        "affinity_mean_absolute_delta": float(np.mean(np.abs(tumor_values - clean_values))) if len(tumor_values) else None,
    }


def run(
    output_root: Path,
    config_dir: Path,
    experiment_root: Path,
    run_clean_embedding: bool,
    case_limit: int,
) -> dict[str, Any]:
    records = load_scope_records(output_root)
    if case_limit:
        records = records[:case_limit]
    clean_embeddings: dict[str, np.ndarray] = {}
    if run_clean_embedding:
        from tools.wsi_embeddings import run_wsi_embeddings

        clean_root = experiment_root / "whole_clean_tissue"
        for record in records:
            result = run_wsi_embeddings(
                case_id=record["case_id"],
                wsi_record={"File Path": record["slide_path"]},
                output_root=str(clean_root),
                patch_dir=record["patch_dir"],
                tumor_coordinates_h5_path=record["clean_coordinates_h5_path"],
                config_dir=str(config_dir),
            )
            artifact_path = str(result.get("artifacts", {}).get("slide_embedding_npy_path", "") or "")
            if artifact_path and Path(artifact_path).is_file():
                clean_embeddings[record["case_id"]] = np.asarray(np.load(artifact_path), dtype=float)
    comparison = compare_embeddings(records, clean_embeddings)
    summary = {
        "experiment": "wsi_scope_sensitivity_v1",
        "inputs": {"main_output_root": str(output_root), "config_dir": str(config_dir)},
        "candidate_case_count": len(records),
        "clean_embedding_executed": bool(run_clean_embedding),
        "comparison": comparison,
        "interpretation_policy": {
            "primary_scope": "tumor_only",
            "secondary_scope": "whole_clean_tissue",
            "secondary_scope_does_not_replace_primary": True,
            "no_embedding_concatenation": True,
            "no_main_pipeline_change": True,
        },
    }
    experiment_root.mkdir(parents=True, exist_ok=True)
    write_csv(experiment_root / "wsi_scope_cases.csv", records)
    if comparison.get("embedding_cosine_rows"):
        write_csv(experiment_root / "wsi_tumor_clean_embedding_similarity.csv", comparison["embedding_cosine_rows"])
    (experiment_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare tumor-only and whole-clean-tissue GigaPath views.")
    parser.add_argument("--output-root", type=Path, default=Path("output_kirc"))
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--experiment-root", type=Path, default=Path("output_kirc_v9/experiment_wsi_scope_sensitivity"))
    parser.add_argument("--run-clean-embedding", action="store_true")
    parser.add_argument("--case-limit", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run(args.output_root, args.config_dir, args.experiment_root, args.run_clean_embedding, args.case_limit), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
