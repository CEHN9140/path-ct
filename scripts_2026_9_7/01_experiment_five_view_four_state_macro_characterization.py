#!/usr/bin/env python3
"""Offline characterization of four candidate macro-states from five-view stable cores."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts_2026_8_31 import analyze_multi_k_stable_cores as base
from scripts_2026_8_31 import experiment_multi_k_accepted_core_stability as stability
from scripts_2026_8_31 import experiment_macro_state_analysis as previous_macro
from scripts_2026_8_31.five_view_experiment import load_five_view_inputs


MACRO_STATE_CORES = {
    "STATE_A": ("CORE01",),
    "STATE_B": ("CORE02",),
    "STATE_C": ("CORE03",),
    "STATE_D": ("CORE04", "CORE05", "CORE06", "CORE07"),
}
EXPECTED_CORE_SIZES = [5, 7, 9, 10, 14, 15, 17]


def build_macro_states(cores):
    return {
        state: sorted({case_id for core in source_cores for case_id in cores[core]})
        for state, source_cores in MACRO_STATE_CORES.items()
    }


def validate_stable_cores(cores, patient_ids):
    if sorted(cores) != [f"CORE0{i}" for i in range(1, 8)]:
        raise ValueError("5-view stable-core输入必须包含CORE01-CORE07")
    if sorted(len(set(members)) for members in cores.values()) != EXPECTED_CORE_SIZES:
        raise ValueError("5-view stable-core规模不是预期的[5, 7, 9, 10, 14, 15, 17]")
    members = [case_id for values in cores.values() for case_id in values]
    if len(members) != 77 or len(set(members)) != 77:
        raise ValueError("5-view stable-core病例必须是互不重叠的77例")
    if not set(members).issubset(patient_ids):
        raise ValueError("stable-core病例不完全存在于5-view patient order中")
    return cores


def build_five_view_stable_core_root(multi_k_root, patient_ids):
    result = stability.analyze(
        multi_k_root,
        patient_ids,
        stability.INITIAL_KS,
        stability.REPEATS,
        min_core_size=5,
    )
    if result.get("analysis_status") != "complete":
        raise RuntimeError(
            "Five-view multi-K review is incomplete: "
            f"{result.get('valid_run_count', 0)}/{result.get('expected_run_count', 21)} valid runs."
        )
    return multi_k_root


def evidence_comparison_rows(
    seven_cores,
    four_states,
    seven_rna,
    four_rna,
    seven_wxs,
    four_wxs,
    seven_cnv,
    four_cnv,
    seven_clinical,
    four_clinical,
):
    rows = []
    seven_total = len(seven_cores) * (len(seven_cores) - 1) // 2
    four_total = len(four_states) * (len(four_states) - 1) // 2
    for name, seven_table, four_table in (
        ("RNA", seven_rna, four_rna),
        ("WXS", seven_wxs, four_wxs),
        ("CNV", seven_cnv, four_cnv),
        ("clinical", seven_clinical, four_clinical),
    ):
        seven_supported = len({(row.get("core_a"), row.get("core_b")) for row in seven_table if base.q_below(row)})
        four_supported = len({(row.get("core_a"), row.get("core_b")) for row in four_table if base.q_below(row)})
        rows.append({
            "evidence": f"{name} pairwise FDR support",
            "seven_core_supported_pairs": seven_supported,
            "seven_core_total_pairs": seven_total,
            "seven_core_supported_fraction": base.rounded(seven_supported / max(seven_total, 1)),
            "four_state_supported_pairs": four_supported,
            "four_state_total_pairs": four_total,
            "four_state_supported_fraction": base.rounded(four_supported / max(four_total, 1)),
        })
    return rows


def validate_multi_k_binding(multi_k_root, data_root):
    candidate_dir = Path(data_root) / "candidate_subtype"
    hashes = {
        "fused_similarity_sha256": base.file_sha256(candidate_dir / "fused_similarity.npy"),
        "patient_order_sha256": base.file_sha256(candidate_dir / "affinity_patient_order.json"),
    }
    metadata_paths = sorted(Path(multi_k_root).glob("run*/K*/run_metadata.json"))
    expected_count = len(stability.INITIAL_KS) * len(stability.REPEATS)
    if len(metadata_paths) != expected_count:
        raise ValueError(
            f"Expected {expected_count} multi-K run metadata files, found {len(metadata_paths)}"
        )
    for metadata_path in metadata_paths:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key, value in hashes.items():
            if metadata.get(key) != value:
                raise ValueError(
                    f"{metadata_path} does not match current raw five-view {key}"
                )
    return hashes, len(metadata_paths)


def rename_rows(rows, key="core_id"):
    return [
        {"state_id": row[key], **{name: value for name, value in row.items() if name != key}}
        if key in row else dict(row)
        for row in rows
    ]


def write_distance_matrices(output_root, distances):
    for modality, matrix in distances.items():
        base.write_csv(
            output_root / f"{modality}_macro_state_distance_matrix.csv",
            [{"state_id": state, **values} for state, values in matrix.items()],
        )


def run(
    data_root=ROOT / "output_kirc_raw",
    multi_k_root=ROOT / "output_kirc_v13/00_five_view_multi_k_agent_review",
    output_root=ROOT / "output_kirc_v13/01_five_view_four_state_macro_characterization",
    config_dir=ROOT / "configs",
    top_pathways=25,
    top_cnv=25,
    random_state=42,
    force=False,
):
    if output_root.exists() and any(output_root.iterdir()) and not force:
        raise FileExistsError(f"Output exists; pass --force to overwrite: {output_root}")
    if force and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    figures = output_root / "figures"
    figures.mkdir()

    states, _ = base.load_states(data_root)
    patient_ids, matrices, fused, _ = load_five_view_inputs(data_root, config_dir)
    multi_k_root = Path(multi_k_root)
    input_hashes, run_count = validate_multi_k_binding(multi_k_root, data_root)
    build_five_view_stable_core_root(multi_k_root, patient_ids)
    source_cores, _ = base.load_cores(multi_k_root)
    validate_stable_cores(source_cores, set(patient_ids))
    macro_states = build_macro_states(source_cores)
    stable_ids = sorted(set().union(*(set(values) for values in macro_states.values())))
    if sorted(map(len, macro_states.values())) != [14, 15, 17, 31]:
        raise ValueError("4个macro-state规模不是[14, 15, 17, 31]")

    base.write_csv(output_root / "macro_state_definition.csv", [
        {"state_id": state, "source_cores": "+".join(MACRO_STATE_CORES[state]), "patient_n": len(members)}
        for state, members in macro_states.items()
    ])
    base.write_csv(output_root / "core_to_macro_state.csv", [
        {"core_id": core, "state_id": state}
        for state, core_ids in MACRO_STATE_CORES.items()
        for core in core_ids
    ])
    base.write_csv(output_root / "macro_state_membership.csv", [
        {"state_id": state, "patient_id": case_id}
        for state, members in macro_states.items()
        for case_id in members
    ])

    affinities = {name: matrices[name] for name in ("ct", "wsi", "rna", "wxs", "cnv")}
    affinities["fused"] = fused

    ct_features, ct_table = base.ct_radiomics_table(states, stable_ids)
    ct_rows, ct_pairs = base.radiomics_analysis(ct_table, ct_features, macro_states, stable_ids)
    base.write_csv(output_root / "ct_radiomics_macro_state_vs_rest.csv", rename_rows(ct_rows))
    base.write_csv(output_root / "ct_radiomics_macro_state_pairwise.csv", ct_pairs)

    wsi_features, wsi_table = base.wsi_embedding_table(states, stable_ids)
    base.write_csv(output_root / "wsi_embedding_macro_state_vs_rest.csv", rename_rows(base.core_rest_rows(wsi_table, wsi_features, macro_states, stable_ids)))
    base.write_csv(output_root / "wsi_embedding_macro_state_pairwise.csv", base.pairwise_numeric_table(wsi_table, wsi_features, macro_states))

    pathways, scores = base.load_expression_scores(states, config_dir, data_root)
    rna_rows, rna_pairs = base.rna_analysis(states, config_dir, macro_states, stable_ids, pathways, scores)
    base.write_csv(output_root / "rna_hallmark_macro_state_vs_rest.csv", rename_rows(rna_rows))
    base.write_csv(output_root / "rna_hallmark_macro_state_pairwise.csv", rna_pairs)
    base.write_csv(output_root / "rna_hallmark_tss_adjusted_macro_state.csv", previous_macro.tss_adjusted_rna(scores, pathways, macro_states))
    leave_cj_rows, _ = previous_macro.leave_out_tss_rna(states, scores, pathways, macro_states, stable_ids, "CJ", config_dir)
    base.write_csv(output_root / "rna_hallmark_leave_CJ_out.csv", rename_rows(leave_cj_rows))

    wxs_features, wxs_table = base.load_table(data_root / "wxs/wxs_discovery_features.csv")
    mutation_features = [feature for feature in wxs_features if feature.startswith("mutation::")]
    wxs_rows, wxs_pairs = base.binary_analysis(wxs_table, mutation_features, macro_states, stable_ids, "mutation")
    base.write_csv(output_root / "wxs_macro_state_vs_rest.csv", rename_rows(wxs_rows))
    base.write_csv(output_root / "wxs_macro_state_pairwise.csv", wxs_pairs)

    cnv_features, cnv_table = base.load_table(data_root / "cnv/case_features.csv")
    cnv_cont, cnv_event, cnv_pairs_cont, cnv_pairs_event = base.cnv_analysis(cnv_table, cnv_features, macro_states, stable_ids)
    base.write_csv(output_root / "cnv_continuous_macro_state_vs_rest.csv", rename_rows(cnv_cont))
    base.write_csv(output_root / "cnv_gain_loss_macro_state_vs_rest.csv", rename_rows(cnv_event))
    base.write_csv(output_root / "cnv_continuous_macro_state_pairwise.csv", cnv_pairs_cont)
    base.write_csv(output_root / "cnv_gain_loss_macro_state_pairwise.csv", cnv_pairs_event)

    separation, distances, tests, audits = base.core_embedding_analysis(affinities, patient_ids, macro_states)
    base.write_csv(output_root / "embedding_macro_state_separation.csv", separation)
    write_distance_matrices(output_root, distances)
    base.write_json(output_root / "affinity_audit.json", audits)
    base.write_csv(output_root / "macro_state_permanova.csv", [{"modality": modality, **tests[modality]["permanova"]} for modality in tests])
    base.write_csv(output_root / "macro_state_permdisp.csv", [{"modality": modality, **tests[modality]["permdisp"]} for modality in tests])

    core_runs = base.load_core_runs(multi_k_root)
    co_run, co_k = base.cooccurrence_from_runs(core_runs, macro_states)
    base.write_csv(output_root / "macro_state_cooccurrence_by_run.csv", co_run)
    base.write_csv(output_root / "macro_state_cooccurrence_by_k.csv", co_k)
    confounds = base.core_confounds(data_root, config_dir, states, macro_states)
    base.write_csv(output_root / "macro_state_confounders.csv", confounds)

    from tools.subtype_review_common import clinical_table

    clinical_records = clinical_table(states)
    availability_rows, availability = base.clinical_availability(clinical_records, macro_states, stable_ids)
    base.write_csv(output_root / "clinical_availability.csv", availability_rows)
    base.write_json(output_root / "clinical_availability_summary.json", availability)
    clinical_rows, clinical_pairs, survival_rows = base.clinical_analysis(
        clinical_records, macro_states, stable_ids,
        availability["eligible_variables"], availability["survival_eligible"],
    )
    base.write_csv(output_root / "clinical_macro_state_vs_rest.csv", rename_rows(clinical_rows))
    base.write_csv(output_root / "clinical_macro_state_pairwise.csv", clinical_pairs)
    base.write_csv(output_root / "clinical_survival_macro_state_vs_rest.csv", rename_rows(survival_rows))
    base.write_csv(output_root / "clinical_patient_records.csv", [
        {"case_id": case_id, "state_id": next(state for state, members in macro_states.items() if case_id in members), **clinical_records[case_id]}
        for case_id in stable_ids if case_id in clinical_records
    ])
    base.write_csv(output_root / "stage_adjusted_survival.csv", previous_macro.stage_adjusted_survival(clinical_records, macro_states))

    five_rna, five_rna_pairs = base.rna_analysis(states, config_dir, source_cores, stable_ids, pathways, scores)
    five_wxs, five_wxs_pairs = base.binary_analysis(wxs_table, mutation_features, source_cores, stable_ids, "mutation")
    five_cnv, _, five_cnv_pairs, _ = base.cnv_analysis(cnv_table, cnv_features, source_cores, stable_ids)
    _, five_clinical_pairs, _ = base.clinical_analysis(clinical_records, source_cores, stable_ids, availability["eligible_variables"], availability["survival_eligible"])
    base.write_csv(output_root / "seven_core_vs_four_state_evidence.csv", evidence_comparison_rows(
        source_cores, macro_states, five_rna_pairs, rna_pairs, five_wxs_pairs, wxs_pairs,
        five_cnv_pairs, cnv_pairs_cont, five_clinical_pairs, clinical_pairs,
    ))
    seven_core_fused = base.core_embedding_analysis({"fused": fused}, patient_ids, source_cores)[2]["fused"]
    four_state_fused = tests["fused"]
    base.write_csv(output_root / "seven_core_vs_four_state_fused_comparison.csv", [
        {"model": "seven_core", "group_count": 7, "patient_count": 77, **seven_core_fused["permanova"], **{"permdisp_" + key: value for key, value in seven_core_fused["permdisp"].items()}},
        {"model": "four_macro_state", "group_count": 4, "patient_count": 77, **four_state_fused["permanova"], **{"permdisp_" + key: value for key, value in four_state_fused["permdisp"].items()}},
    ])

    state_order = sorted(macro_states)
    selected_pathways = base.select_pathways(rna_rows, top_pathways)
    lookup = {(row["core_id"], row["pathway"]): row for row in rna_rows}
    matrix = np.asarray([[lookup.get((state, pathway), {}).get("smd") or 0 for state in state_order] for pathway in selected_pathways])
    stars = [["***" if base.q_below(lookup.get((state, pathway), {}), .001) else "**" if base.q_below(lookup.get((state, pathway), {}), .01) else "*" if base.q_below(lookup.get((state, pathway), {})) else "" for state in state_order] for pathway in selected_pathways]
    base.plot_heatmap(figures / "macro_state_pathway_smd_heatmap", matrix, selected_pathways, state_order, "Four macro-state Hallmark pathway SMD", stars, True)
    base.plot_bubbles(figures / "macro_state_pathway_bubble_plot", rna_rows, state_order, selected_pathways)
    selected_cnv = base.select_cnv_heatmap_features(cnv_cont, top_cnv)
    cnv_lookup = {(row["core_id"], row["feature"]): row for row in cnv_cont}
    base.plot_heatmap(figures / "macro_state_cnv_effect_heatmap", np.asarray([[cnv_lookup.get((state, feature), {}).get("cliffs_delta") or 0 for state in state_order] for feature in selected_cnv]), selected_cnv, state_order, "Four macro-state CNV Cliff's delta", diverging=True)
    base.plot_oncoplot(figures / "macro_state_driver_mutation_oncoplot", wxs_table, mutation_features, macro_states, stable_ids, {state: "" for state in state_order}, {state: "" for state in state_order})
    base.plot_survival_km(figures / "macro_state_overall_survival_km", clinical_records, macro_states)
    projection = previous_macro.plot_macro_fused_structure(figures / "macro_state_fused_structure", fused, patient_ids, macro_states, random_state)
    for modality, matrix in affinities.items():
        if modality != "fused":
            base.plot_umap(figures / f"{modality}_macro_state_umap", matrix, patient_ids, macro_states, random_state)

    summary = {
        "state_count": 4,
        "source_core_count": 7,
        "patient_count": len(stable_ids),
        "state_sizes": {state: len(members) for state, members in macro_states.items()},
        "projection_methods": projection,
        "analysis_scope": "77 five-view stable-core patients",
        "multi_k_run_count": run_count,
        "source_data_root": str(Path(data_root).resolve()),
        **input_hashes,
        "taxonomy_limitation": "No external ClearCode34, ccA/ccB, or TCGA molecular subtype labels were supplied; this experiment does not claim known-taxonomy recovery.",
        "source_stable_core_sha256": base.file_sha256(multi_k_root / "stable_core_membership.csv"),
    }
    generated = sorted(str(path.relative_to(output_root)) for path in output_root.rglob("*") if path.is_file())
    base.write_json(output_root / "macro_state_analysis_manifest.json", {
        "macro_state_cores": MACRO_STATE_CORES,
        "analysis_parameters": {"top_pathways": top_pathways, "top_cnv": top_cnv, "random_state": random_state, "modalities": list(affinities)},
        "source_five_view_multi_k_root": str(multi_k_root),
        "source_data_root": str(Path(data_root).resolve()),
        "source_fused_similarity_sha256": input_hashes["fused_similarity_sha256"],
        "source_patient_order_sha256": input_hashes["patient_order_sha256"],
        "multi_k_run_count": run_count,
        "generated_files": generated,
    })
    base.write_json(output_root / "macro_state_analysis_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "output_kirc_raw")
    parser.add_argument("--multi-k-root", type=Path, default=ROOT / "output_kirc_v13/00_five_view_multi_k_agent_review")
    parser.add_argument("--output-root", type=Path, default=ROOT / "output_kirc_v13/01_five_view_four_state_macro_characterization")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs")
    parser.add_argument("--top-pathways", type=int, default=25)
    parser.add_argument("--top-cnv", type=int, default=25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    print(json.dumps(run(**vars(parser.parse_args())), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
