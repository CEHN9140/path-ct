#!/usr/bin/env python3
"""Deterministic conventional consensus-clustering baseline without Agent review."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]
PAC_INTERVAL = (0.1, 0.9)
ACTIVE_MODALITIES = ("ct", "wsi", "rna", "wxs")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_and_validate_inputs(output_root: Path) -> tuple[list[str], dict, np.ndarray, dict[int, dict]]:
    candidate_root = output_root / "candidate_subtype"
    consensus_root = candidate_root / "consensus_cluster"
    order_path = candidate_root / "affinity_patient_order.json"
    fused_distance_path = candidate_root / "fused_distance.npy"
    manifest_path = consensus_root / "resampling_manifest.json"
    patient_ids = [str(value) for value in json.loads(order_path.read_text(encoding="utf-8"))]
    if len(patient_ids) != len(set(patient_ids)):
        raise ValueError("Patient order contains duplicate IDs.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidate_ks = tuple(int(value) for value in manifest.get("candidate_ks", []))
    if candidate_ks != tuple(range(2, 9)):
        raise ValueError(f"Expected candidate K=2..8, got {candidate_ks}.")
    manifest_ids = [str(value) for value in manifest.get("patient_ids", [])]
    if manifest_ids != patient_ids:
        raise ValueError("Manifest patient order does not match affinity patient order.")

    fused_distance = np.asarray(np.load(fused_distance_path), dtype=float)
    n_patients = len(patient_ids)
    if fused_distance.shape != (n_patients, n_patients):
        raise ValueError("Fused distance matrix shape does not match patient order.")
    if not np.isfinite(fused_distance).all():
        raise ValueError("Fused distance matrix contains non-finite values.")
    if np.any(fused_distance < 0) or not np.allclose(fused_distance, fused_distance.T):
        raise ValueError("Fused distance matrix must be finite, non-negative, and symmetric.")
    if not np.allclose(np.diag(fused_distance), 0):
        raise ValueError("Fused distance matrix diagonal must be approximately zero.")

    partitions = {}
    for k in candidate_ks:
        matrix_path = consensus_root / f"consensus_matrix_K{k}.npy"
        labels_path = consensus_root / f"consensus_hierarchical_K{k}.json"
        consensus = np.asarray(np.load(matrix_path), dtype=float)
        if consensus.shape != (n_patients, n_patients):
            raise ValueError(f"K={k} consensus matrix shape is invalid.")
        if not np.isfinite(consensus).all():
            raise ValueError(f"K={k} consensus matrix contains non-finite values.")
        if not np.allclose(consensus, consensus.T) or not np.allclose(np.diag(consensus), 1):
            raise ValueError(f"K={k} consensus matrix must be symmetric with diagonal one.")
        if np.any(consensus < 0) or np.any(consensus > 1):
            raise ValueError(f"K={k} consensus values must lie in [0, 1].")

        payload = json.loads(labels_path.read_text(encoding="utf-8"))
        labels_by_id = {str(key): int(value) for key, value in payload["labels"].items()}
        if int(payload.get("initial_k")) != k or set(labels_by_id) != set(patient_ids):
            raise ValueError(f"K={k} labels do not match the candidate patient order.")
        labels = np.asarray([labels_by_id[patient_id] for patient_id in patient_ids])
        if len(np.unique(labels)) != k:
            raise ValueError(f"K={k} labels contain {len(np.unique(labels))} clusters.")
        partitions[k] = {"consensus": consensus, "labels": labels, "labels_by_id": labels_by_id}
    return patient_ids, manifest, fused_distance, partitions


def cdf_area(consensus: np.ndarray) -> float:
    upper = consensus[np.triu_indices_from(consensus, k=1)]
    if upper.size == 0:
        raise ValueError("Consensus matrix has no off-diagonal values.")
    grid = np.unique(np.concatenate(([0.0], upper, [1.0])))
    empirical_cdf = np.searchsorted(np.sort(upper), grid, side="right") / upper.size
    return float(np.trapezoid(empirical_cdf, grid))


def compute_k_metrics(k: int, partition: dict, fused_distance: np.ndarray) -> dict:
    consensus = partition["consensus"]
    labels = partition["labels"]
    upper = np.triu_indices(len(labels), k=1)
    pair_values = consensus[upper]
    pac = float(np.mean((pair_values > PAC_INTERVAL[0]) & (pair_values < PAC_INTERVAL[1])))
    same = labels[:, None] == labels[None, :]
    within = same[upper]
    if not within.any() or (~within).sum() == 0:
        raise ValueError(f"K={k} does not provide both within- and between-cluster pairs.")
    within_mean = float(pair_values[within].mean())
    between_mean = float(pair_values[~within].mean())
    sizes = np.unique(labels, return_counts=True)[1]
    consensus_distance = 1.0 - consensus
    np.fill_diagonal(consensus_distance, 0.0)
    return {
        "k": k,
        "pac": pac,
        "silhouette_fused": float(silhouette_score(fused_distance, labels, metric="precomputed")),
        "cdf_area": cdf_area(consensus),
        "silhouette_consensus": float(silhouette_score(consensus_distance, labels, metric="precomputed")),
        "mean_within_consensus": within_mean,
        "mean_between_consensus": between_mean,
        "consensus_gap": within_mean - between_mean,
        "min_cluster_size": int(sizes.min()),
        "max_cluster_size": int(sizes.max()),
        "cluster_size_mean": float(sizes.mean()),
        "cluster_size_std": float(sizes.std()),
    }


def add_cdf_elbow_diagnostics(metrics: list[dict]) -> None:
    ordered = sorted(metrics, key=lambda row: row["k"])
    areas = np.asarray([row["cdf_area"] for row in ordered], dtype=float)
    deltas = [None]
    for previous, current in zip(areas, areas[1:]):
        deltas.append(float((current - previous) / previous) if previous else None)
    x = np.asarray([row["k"] for row in ordered], dtype=float)
    x = (x - x.min()) / (x.max() - x.min())
    area_range = areas.max() - areas.min()
    y = (areas - areas.min()) / area_range if area_range else np.zeros_like(areas)
    elbow_scores = y - x
    for row, delta, score in zip(ordered, deltas, elbow_scores):
        row["delta_cdf_area"] = delta
        row["elbow_score"] = float(score)


def select_k(metrics: list[dict]) -> int:
    if any("elbow_score" not in row for row in metrics):
        raise ValueError("CDF elbow diagnostics must be added before selecting K.")
    maximum_elbow = max(row["elbow_score"] for row in metrics)
    elbow_ties = [
        row for row in metrics
        if np.isclose(row["elbow_score"], maximum_elbow, rtol=1e-12, atol=1e-12)
    ]
    return min(row["k"] for row in elbow_ties)


def selection_diagnostics(metrics: list[dict], selected_k: int) -> dict:
    ordered = sorted(metrics, key=lambda row: row["k"])
    pac_values = [row["pac"] for row in ordered]
    selected = next(row for row in ordered if row["k"] == selected_k)
    return {
        "selection_method": "cdf_area_elbow",
        "candidate_k_min": ordered[0]["k"],
        "candidate_k_max": ordered[-1]["k"],
        "selected_at_search_boundary": selected_k in {ordered[0]["k"], ordered[-1]["k"]},
        "selected_elbow_score": selected["elbow_score"],
        "pac_monotonic_nonincreasing": all(left >= right for left, right in zip(pac_values, pac_values[1:])),
        "interpretation": (
            "cdf_area_elbow_at_search_boundary; inspect_curve_before_scientific_interpretation"
            if selected_k in {ordered[0]["k"], ordered[-1]["k"]}
            else "cdf_area_elbow_selected_within_candidate_range"
        ),
    }


def build_final_subtypes(selected_k: int, partition: dict, patient_ids: list[str]) -> tuple[list[dict], list[dict]]:
    labels = partition["labels"]
    final_subtypes = []
    membership = []
    for index, label in enumerate(sorted(np.unique(labels)), 1):
        members = [patient_id for patient_id, item in zip(patient_ids, labels) if item == label]
        subtype_id = f"WO_AGENT_SUBTYPE{index:02d}"
        final_subtypes.append({
            "subtype_id": subtype_id,
            "source_set_id": f"K{selected_k}_C{index:04d}",
            "selected_k": selected_k,
            "member_count": len(members),
            "member_ids": members,
        })
        membership.extend({"patient_id": patient_id, "subtype_id": subtype_id} for patient_id in members)
    if sorted(row["patient_id"] for row in membership) != sorted(patient_ids):
        raise ValueError("Final subtype membership does not cover the candidate cohort exactly once.")
    return final_subtypes, membership


def load_analysis_context(data_root: Path, result_dir: Path) -> dict:
    import analyze_final_subtypes as final_analysis

    subtype_path = result_dir / "final_subtypes.json"
    membership_path = result_dir / "membership.csv"
    summary_path = result_dir / "summary.json"
    for path in (subtype_path, membership_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    records = json.loads(subtype_path.read_text(encoding="utf-8"))
    membership = pd.read_csv(membership_path, dtype=str)
    if not {"subtype_id", "patient_id"}.issubset(membership.columns):
        raise ValueError("w/o-Agent membership.csv must contain subtype_id and patient_id")
    subtypes = {str(row["subtype_id"]): [str(value) for value in row["member_ids"]] for row in records}
    expected = {(sid, patient) for sid, members in subtypes.items() for patient in members}
    actual = {(str(row.subtype_id), str(row.patient_id)) for row in membership.itertuples()}
    if expected != actual:
        raise ValueError("w/o-Agent membership.csv disagrees with final_subtypes.json")
    candidate_path = data_root / "candidate_subtype" / "affinity_patient_order.json"
    candidate_ids = [str(value) for value in json.loads(candidate_path.read_text(encoding="utf-8"))]
    core_ids = [patient for members in subtypes.values() for patient in members]
    if len(core_ids) != len(set(core_ids)) or not set(core_ids).issubset(candidate_ids):
        raise ValueError("w/o-Agent subtype membership does not form a valid candidate partition")
    states = final_analysis.load_states(data_root)
    missing = sorted(set(candidate_ids) - set(states))
    if missing:
        raise ValueError(f"Candidate cohort lacks patient states: {missing[:10]}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "subtypes": subtypes,
        "subtype_order": list(subtypes),
        "core_patient_ids": core_ids,
        "candidate_patient_ids": candidate_ids,
        "noncore_patient_ids": sorted(set(candidate_ids) - set(core_ids)),
        "subtype_metadata": {
            sid: {"member_count": len(members), "common_accept_set_count": 0,
                  "supporting_k_count": 0, "supporting_ks": [], "supporting_runs": []}
            for sid, members in subtypes.items()
        },
        "patient_states": states,
        "subtype_source_type": "wo_agent_consensus_partition",
        "subtype_selection": summary,
        "source_paths": {
            "subtypes": subtype_path,
            "membership": membership_path,
            "summary": summary_path,
            "order": candidate_path,
        },
    }


def write_analysis_recurrence_summary(context: dict, output_dir: Path) -> pd.DataFrame:
    rows = [{
        "subtype_id": subtype_id,
        "member_count": len(members),
        "stability_source": "consensus_partition",
        "selected_k": context["subtype_selection"]["selected_k"],
        "common_accept_set_count": None,
        "occurrence_frequency": None,
        "supporting_k_count": None,
        "supporting_ks": "[]",
        "supporting_run_count": None,
        "min_pair_recurrence_similarity": None,
        "mean_pair_recurrence_similarity": None,
        "median_pair_recurrence_similarity": None,
    } for subtype_id, members in context["subtypes"].items()]
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "subtype_stability_summary.csv", index=False)
    return frame


def write_analysis_representatives(context: dict, data_root: Path, output_dir: Path) -> pd.DataFrame:
    from agents.subtype_review.tools import modality_distance_matrices

    ids, distances = modality_distance_matrices(str(data_root))
    position = {patient_id: index for index, patient_id in enumerate(ids)}
    rows = []
    for subtype_id, members in context["subtypes"].items():
        scores = {
            patient: float(np.mean([
                distances[modality][position[patient], position[other]]
                for modality in ACTIVE_MODALITIES for other in members if other != patient
            ])) if len(members) > 1 else 0.0
            for patient in members
        }
        patient = min(scores, key=lambda value: (scores[value], value))
        modality_scores = {
            modality: float(np.mean([
                distances[modality][position[patient], position[other]]
                for other in members if other != patient
            ])) if len(members) > 1 else 0.0
            for modality in ACTIVE_MODALITIES
        }
        rows.append({
            "subtype_id": subtype_id,
            "case_id": patient,
            "member_count": len(members),
            "recurrence_centrality": None,
            **{f"{modality}_mean_distance": value for modality, value in modality_scores.items()},
            "multimodal_mean_distance": float(np.mean(list(modality_scores.values()))),
            "selection_rule": "minimum native multimodal mean distance; lexical case_id",
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "representative_patients.csv", index=False)
    return frame


def run_final_subtype_analysis(
    data_root: Path,
    result_dir: Path,
    config_dir: Path,
    analysis_root: Path,
    clinical_file: Path,
    survival_file: Path | None,
    pathreport_root: Path,
    pathreport_manifest: Path,
    pathreport_sample_sheet: Path,
    top_marker_genes: int,
    permutations: int,
    bootstrap_iterations: int,
) -> dict:
    import yaml
    import analyze_final_subtypes as final_analysis

    context = load_analysis_context(data_root, result_dir)
    config = yaml.safe_load((config_dir / "subtype_review.yaml").read_text(encoding="utf-8"))
    analysis_root.mkdir(parents=True, exist_ok=True)
    recurrence = write_analysis_recurrence_summary(context, analysis_root)
    final_analysis.representation_analysis(context, data_root, analysis_root, config)
    wsi_rest, _ = final_analysis.wsi_analysis(context, data_root, analysis_root, bootstrap_iterations)
    ct_rest = final_analysis.ct_analysis(context, data_root, config_dir, analysis_root, bootstrap_iterations)
    rna = final_analysis.rna_analysis(context, data_root, config_dir, analysis_root, top_marker_genes, bootstrap_iterations)
    wxs = final_analysis.wxs_analysis(context, data_root, config_dir, analysis_root, permutations)
    taxonomy = final_analysis.known_and_confounders(context, data_root, config_dir, analysis_root)
    clinical = final_analysis.clinical_analysis(context, clinical_file, analysis_root, permutations)
    survival_status, survival_summary = final_analysis.survival_analysis(
        context, clinical_file, analysis_root, survival_file
    )
    pathreport_audit = final_analysis.pathreport_mapping(
        context, pathreport_root, pathreport_manifest, pathreport_sample_sheet, analysis_root
    )
    representatives = write_analysis_representatives(context, data_root, analysis_root)
    final_analysis.identity_cards(
        context, recurrence, wsi_rest, ct_rest, rna, wxs, taxonomy,
        representatives, survival_summary, analysis_root,
    )
    summary = {
        "status": "complete",
        "analysis_type": "wo_agent_final_subtype_characterization",
        "subtype_source": str((result_dir / "final_subtypes.json").resolve()),
        "selected_k": context["subtype_selection"]["selected_k"],
        "subtype_count": len(context["subtypes"]),
        "candidate_patient_count": len(context["candidate_patient_ids"]),
        "core_patient_count": len(context["core_patient_ids"]),
        "subtype_sizes": {sid: len(members) for sid, members in context["subtypes"].items()},
        "analyses": {
            "representation": "complete", "clinical": "complete", "survival": survival_status,
            "pathreport_mapping": "complete" if not pathreport_audit["missing_cases"] and pathreport_audit["all_mapped_paths_exist"] else "partial",
            "rna": "complete", "wxs": "complete", "known_taxonomy": "complete",
        },
        "warnings": {
            "agent_recurrence_evidence_available": False,
            "stable_subtype_definition": "CDF-area elbow on frozen consensus partitions",
            "wsi_semantic_characterization_available": False,
        },
    }
    (analysis_root / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "output_kirc")
    parser.add_argument("--result-dir", type=Path, default=ROOT / "ablation/results/02_wo_agent")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--analysis-root", type=Path, default=None)
    parser.add_argument("--clinical-file", type=Path, default=ROOT / "data/tcga_kirc_data.json")
    parser.add_argument("--survival-file", type=Path, default=None)
    parser.add_argument("--pathreport-root", type=Path, default=Path("/data/qijun/data/TCGA/KIRC/PathReport"))
    parser.add_argument("--pathreport-manifest", type=Path, default=Path("/data/qijun/data/TCGA/KIRC/PathReport/gdc_manifest.txt"))
    parser.add_argument("--pathreport-sample-sheet", type=Path, default=Path("/data/qijun/data/TCGA/KIRC/PathReport/gdc_sample_sheet.tsv"))
    parser.add_argument("--top-marker-genes", type=int, default=10)
    parser.add_argument("--permutations", type=int, default=9999)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    args = parser.parse_args()

    patient_ids, manifest, fused_distance, partitions = load_and_validate_inputs(args.output_root)
    metrics = [compute_k_metrics(k, partitions[k], fused_distance) for k in sorted(partitions)]
    add_cdf_elbow_diagnostics(metrics)
    selected_k = select_k(metrics)
    diagnostics = selection_diagnostics(metrics, selected_k)
    for row in metrics:
        row["selected"] = row["k"] == selected_k
    final_subtypes, membership = build_final_subtypes(selected_k, partitions[selected_k], patient_ids)

    args.result_dir.mkdir(parents=True, exist_ok=True)
    candidate_root = args.output_root / "candidate_subtype"
    consensus_root = candidate_root / "consensus_cluster"
    input_paths = [
        candidate_root / "affinity_patient_order.json",
        candidate_root / "fused_distance.npy",
        consensus_root / "resampling_manifest.json",
        *[consensus_root / f"consensus_matrix_K{k}.npy" for k in partitions],
        *[consensus_root / f"consensus_hierarchical_K{k}.json" for k in partitions],
    ]
    selected_metrics = next(row for row in metrics if row["selected"])
    pd.DataFrame(metrics).to_csv(args.result_dir / "k_selection_metrics.csv", index=False)
    (args.result_dir / "selected_k.json").write_text(json.dumps({
        "analysis": "wo_agent_consensus_clustering",
        "selection_method": "cdf_area_elbow",
        "cdf_area_definition": "trapezoidal_area_under_empirical_off_diagonal_consensus_CDF",
        "pac_interval": list(PAC_INTERVAL),
        "selected_k": selected_k,
        "tie_break_rule": "smaller_k_among_maximum_elbow_score",
        "candidate_ks": sorted(partitions),
        "selection_diagnostics": diagnostics,
        "input_manifest": str(consensus_root / "resampling_manifest.json"),
        "input_sha256": {str(path.relative_to(args.output_root)): sha256(path) for path in input_paths},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.result_dir / "final_subtypes.json").write_text(json.dumps(final_subtypes, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(membership).to_csv(args.result_dir / "membership.csv", index=False)
    (args.result_dir / "summary.json").write_text(json.dumps({
        "selected_k": selected_k,
        "patient_count": len(patient_ids),
        "subtype_count": len(final_subtypes),
        "subtype_sizes": [row["member_count"] for row in final_subtypes],
        "selected_k_pac": selected_metrics["pac"],
        "selected_k_silhouette": selected_metrics["silhouette_fused"],
        "selected_k_cdf_area": selected_metrics["cdf_area"],
        "selected_k_delta_cdf_area": selected_metrics["delta_cdf_area"],
        "selected_k_elbow_score": selected_metrics["elbow_score"],
        "selected_k_silhouette_consensus": selected_metrics["silhouette_consensus"],
        "selected_k_consensus_gap": selected_metrics["consensus_gap"],
        "selection_diagnostics": diagnostics,
        "result_dir": str(args.result_dir.resolve()),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"selected_k": selected_k, "patient_count": len(patient_ids), "subtype_count": len(final_subtypes)}, ensure_ascii=False))
    if args.analyze:
        analysis_root = args.analysis_root or args.result_dir / "final_subtype_analysis"
        print(json.dumps(run_final_subtype_analysis(
            args.output_root, args.result_dir, args.config_dir, analysis_root,
            args.clinical_file, args.survival_file, args.pathreport_root,
            args.pathreport_manifest, args.pathreport_sample_sheet,
            args.top_marker_genes, args.permutations, args.bootstrap_iterations,
        ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
