from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    completeness_score,
    homogeneity_score,
)

from tools.subtype_review_common import (
    clinical_table,
    member_case_ids,
    subtype_review_config,
    tool_result,
)


KNOWN_LABEL_FIELDS = {"stage": "stage_group", "grade": "grade"}
MISSING_LABELS = {"", "not reported", "unknown", "not applicable", "nan", "none"}


def valid_label(value):
    return str(value or "").strip().lower() not in MISSING_LABELS


def cluster_labels_by_case(all_cluster_states, fallback_cluster_state):
    labels = {}
    states = list(all_cluster_states or []) or [fallback_cluster_state]
    for state in states:
        cluster_id = str(state.get("cluster_id", "") or "")
        for case_id in member_case_ids(state):
            labels[str(case_id)] = cluster_id
    return labels


def compare_label_structures(label_name, clinical_field, cluster_labels, clinical):
    case_ids = [
        case_id
        for case_id in cluster_labels
        if valid_label(clinical.get(case_id, {}).get(clinical_field, ""))
    ]
    missing_n = len(cluster_labels) - len(case_ids)
    if len(case_ids) < 2:
        return {
            "label": label_name,
            "available_n": len(case_ids),
            "missing_n": missing_n,
            "set_count": len({cluster_labels[case_id] for case_id in case_ids}),
            "known_label_count": len(
                {
                    str(clinical[case_id].get(clinical_field, "") or "")
                    for case_id in case_ids
                }
            ),
            "contingency_table": {},
            "adjusted_mutual_information": None,
            "adjusted_rand_index": None,
            "optimal_mapping_accuracy": None,
            "homogeneity": None,
            "completeness": None,
        }
    set_levels = sorted({cluster_labels[case_id] for case_id in case_ids})
    known_levels = sorted(
        {str(clinical[case_id].get(clinical_field, "") or "") for case_id in case_ids}
    )
    table = np.asarray(
        [
            [
                sum(
                    cluster_labels[case_id] == set_level
                    and str(clinical[case_id].get(clinical_field, "") or "")
                    == known_level
                    for case_id in case_ids
                )
                for known_level in known_levels
            ]
            for set_level in set_levels
        ],
        dtype=int,
    )
    contingency = {
        set_level: {
            known_level: int(table[row_idx, col_idx])
            for col_idx, known_level in enumerate(known_levels)
        }
        for row_idx, set_level in enumerate(set_levels)
    }
    known_labels = [
        str(clinical[case_id].get(clinical_field, "") or "")
        for case_id in case_ids
    ]
    candidate_labels = [cluster_labels[case_id] for case_id in case_ids]
    from scipy.optimize import linear_sum_assignment
    overlap = np.asarray(table, dtype=float)
    rows, cols = linear_sum_assignment(-overlap)
    mapping_accuracy = float(overlap[rows, cols].sum() / len(case_ids))
    return {
        "label": label_name,
        "available_n": len(case_ids),
        "missing_n": missing_n,
        "set_count": len(set_levels),
        "known_label_count": len(known_levels),
        "contingency_table": contingency,
        "adjusted_mutual_information": round(
            float(adjusted_mutual_info_score(known_labels, candidate_labels)),
            6,
        ),
        "adjusted_rand_index": round(float(adjusted_rand_score(known_labels, candidate_labels)), 6),
        "optimal_mapping_accuracy": round(mapping_accuracy, 6),
        "homogeneity": round(
            float(homogeneity_score(known_labels, candidate_labels)), 6
        ),
        "completeness": round(
            float(completeness_score(known_labels, candidate_labels)), 6
        ),
    }


def known_label_echo_test(
    cluster_state,
    patient_states_by_id,
    output_root,
    config_dir="",
    all_cluster_states=None,
    scope="partition",
    target_ids=None,
    artifact_root=None,
):
    artifact_root = artifact_root or output_root
    cluster_id = str(cluster_state.get("cluster_id", "unknown_cluster"))
    clinical = clinical_table(patient_states_by_id)
    cluster_labels = cluster_labels_by_case(all_cluster_states, cluster_state)
    structure_comparison = {}
    for label_name, clinical_field in KNOWN_LABEL_FIELDS.items():
        structure_comparison[label_name] = compare_label_structures(
            label_name, clinical_field, cluster_labels, clinical
        )
    thresholds = {
        "ami": 0.80,
        "ari": 0.80,
        "optimal_mapping_accuracy": 0.90,
    }
    if config_dir:
        config = subtype_review_config(config_dir).get("known_label_echo", {})
        thresholds.update(
            {
                "ami": float(config.get("near_identity_ami", thresholds["ami"])),
                "ari": float(config.get("near_identity_ari", thresholds["ari"])),
                "optimal_mapping_accuracy": float(config.get("near_identity_mapping_accuracy", thresholds["optimal_mapping_accuracy"])),
            }
        )
    for comparison in structure_comparison.values():
        comparison["near_identity"] = bool(
            (comparison.get("adjusted_mutual_information") or 0) >= thresholds["ami"]
            or (comparison.get("adjusted_rand_index") or 0) >= thresholds["ari"]
        ) and (comparison.get("optimal_mapping_accuracy") or 0) >= thresholds["optimal_mapping_accuracy"]
        comparison["near_identity_thresholds"] = thresholds
    metrics = {
        "known_label_structure_comparison": structure_comparison,
        "analysis_scope": (
            f"{scope} comparison of the current partition versus stage and grade partitions"
        ),
        "decision_semantics": (
            "AMI measures chance-adjusted whole-structure overlap; homogeneity "
            "and completeness distinguish repetition, refinement, coarsening, "
            "and cross-cutting structure. No single-set enrichment is used."
        ),
    }
    return tool_result(
        tool_name="known_label_echo_test",
        status="success",
        cluster_id=cluster_id,
        output_root=artifact_root,
        summary="The complete candidate partition was compared with stage and grade structures.",
        metrics=metrics,
        decision_metrics=metrics,
        evidence_hints=[
            {
                "evidence_type": "known_label_echo",
                "summary": "Whole-partition stage and grade redundancy was evaluated.",
            }
        ],
        support_level="informational",
        concern_level="low",
        figures={},
    )
