#!/usr/bin/env python3
"""Complete the frozen four-view state results with compact, reproducible summaries."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from four_view_state_common import DEFAULT_INPUT, DEFAULT_MEMBERSHIP, ROOT, load_membership, load_states
from tools import post_discovery_characterization as stats
from tools.ct_radiomics import build_ct_discovery_feature_matrix

CHAR = ROOT / "output_kirc_v14/14_four_view_state_characterization"
MAP = ROOT / "output_kirc_v14/15_four_view_state_known_ccrcc_mapping"
AGENT = ROOT / "output_kirc_v14/11_four_view_no_cnv/agent_review"
AUDIT = ROOT / "output_kirc_v14/12_four_view_core_to_macro_state_audit"
STATE_ORDER = ["STATE_A", "STATE_B", "STATE_C", "STATE_D"]


def write_csv(frame, path):
    frame.to_csv(path, index=False)


def patient_state_features(input_root, membership, output_root):
    ids = membership.case_id.tolist()
    rows = []
    wsi_dir = ROOT / "output_kirc_raw/wsi_tumor_seg"
    for case_id in ids:
        path = wsi_dir / case_id / "patch_probabilities.csv"
        if not path.exists():
            continue
        patches = pd.read_csv(path)
        selected = patches[patches["selected_tumor"].astype(bool)]
        if selected.empty:
            selected = patches[patches["tumor_probability"] >= 0.9]
        if selected.empty:
            continue
        values = {column.removeprefix("prob_"): float(selected[column].mean())
                  for column in selected.columns if column.startswith("prob_")}
        values.update({"case_id": case_id, "state_id": membership.set_index("case_id").loc[case_id, "state_id"],
                       "tumor_patch_n": len(selected), "all_patch_n": len(patches)})
        rows.append(values)
    patient = pd.DataFrame(rows)
    if patient.empty:
        return patient, pd.DataFrame(), pd.DataFrame()
    feature_columns = [x for x in patient.columns if x not in {"case_id", "state_id", "tumor_patch_n", "all_patch_n"}]
    means = patient.groupby("state_id")[feature_columns].mean().reindex(STATE_ORDER)
    means.insert(0, "state_n_with_wsi_phenotype", patient.groupby("state_id").size().reindex(STATE_ORDER))
    omnibus = stats.continuous_omnibus(
        {row.case_id: row[feature_columns].to_dict() for _, row in patient.iterrows()},
        feature_columns,
        {state: patient.loc[patient.state_id == state, "case_id"].tolist() for state in STATE_ORDER},
    )
    write_csv(patient, output_root / "wsi_patient_phenotype_features.csv")
    write_csv(means.reset_index(), output_root / "wsi_state_phenotype_means.csv")
    write_csv(pd.DataFrame(omnibus), output_root / "wsi_state_phenotype_omnibus.csv")
    return patient, means, pd.DataFrame(omnibus)


def ct_summary(input_root, membership, states, output_root):
    payload = build_ct_discovery_feature_matrix(
        [states[x] for x in sorted(states)], config_dir=str(ROOT / "configs"), output_root=str(input_root)
    )
    frame = pd.DataFrame(payload["matrix"], index=payload["patient_ids"], columns=payload["feature_names"]).reindex(membership.case_id)
    omnibus = pd.read_csv(CHAR / "ct_radiomics_omnibus.csv")
    selected = omnibus[omnibus.q_value < 0.05].sort_values("q_value").head(12).feature.tolist()
    selected = [feature for feature in selected if feature in frame.columns]
    means = frame.assign(state_id=membership.set_index("case_id").loc[frame.index, "state_id"]).groupby("state_id")[selected].mean().reindex(STATE_ORDER)
    rows = omnibus[omnibus.feature.isin(selected)].copy()
    write_csv(rows, output_root / "ct_representative_feature_summary.csv")
    write_csv(means.reset_index(), output_root / "ct_state_feature_means.csv")
    return frame, means, rows


def micro_macro_summary(output_root):
    mapping = pd.read_csv(AUDIT / "core_to_macro_state.csv", dtype=str)
    counts = pd.crosstab(mapping.macro_state, mapping.core_id).reindex(index=STATE_ORDER, fill_value=0).fillna(0).astype(int)
    write_csv(counts.reset_index(), output_root / "micro_core_to_macro_state_counts.csv")
    mapping.to_csv(output_root / "micro_core_to_macro_state_membership.csv", index=False)
    return mapping, counts


def noncore_summary(input_root, membership, output_root):
    order = json.loads((Path(input_root) / "candidate_subtype/affinity_patient_order.json").read_text())
    matrix = np.load(Path(input_root) / "candidate_subtype/fused_similarity.npy")
    coassign = pd.read_csv(AGENT / "joint_accepted_coassignment_matrix.csv", index_col=0).reindex(index=order, columns=order).to_numpy(float)
    assigned = set(membership.case_id)
    state_map = membership.set_index("case_id")["state_id"].to_dict()
    state_members = {state: membership.loc[membership.state_id == state, "case_id"].tolist() for state in STATE_ORDER}
    acceptance = pd.read_csv(AGENT / "patient_acceptance_frequency.csv").set_index("patient_id")["acceptance_frequency"]
    rows = []
    for i, case_id in enumerate(order):
        state_scores = {}
        for state, members in state_members.items():
            indices = [order.index(member) for member in members if member != case_id]
            state_scores[state] = float(matrix[i, indices].mean()) if indices else np.nan
        finite = {state: score for state, score in state_scores.items() if np.isfinite(score)}
        scores = np.array(list(finite.values()), float)
        probabilities = np.clip(scores - scores.min() + 1e-6, 1e-6, None)
        probabilities /= probabilities.sum()
        entropy = float(-(probabilities * np.log(probabilities)).sum())
        ranked = sorted(finite.items(), key=lambda item: item[1], reverse=True)
        max_partner = float(np.max(np.delete(coassign[i], i)))
        rows.append({"case_id": case_id, "state_id": state_map.get(case_id, "NON_CORE"),
                     "is_non_core": case_id not in assigned, "acceptance_frequency": float(acceptance.get(case_id, np.nan)),
                     "max_coassignment": max_partner, "nearest_state": ranked[0][0], "nearest_state_affinity": ranked[0][1],
                     "second_state_affinity": ranked[1][1], "state_affinity_margin": ranked[0][1] - ranked[1][1],
                     "state_affinity_entropy": entropy})
    result = pd.DataFrame(rows)
    write_csv(result, output_root / "noncore_uncertainty_scores.csv")
    write_csv(result.groupby("is_non_core")[['acceptance_frequency', 'max_coassignment', 'state_affinity_margin', 'state_affinity_entropy']].mean().reset_index(), output_root / "core_noncore_stability_summary.csv")
    return result


def representative_cases(input_root, membership, states, wxs, output_root):
    order = json.loads((Path(input_root) / "candidate_subtype/affinity_patient_order.json").read_text())
    fused = np.load(Path(input_root) / "candidate_subtype/fused_similarity.npy")
    clinical = {}
    for case_id, state in states.items():
        clinical[case_id] = (state.get("clinical", {}) or {}).get("demographic", {}) or {}
    wxs_cols = [x for x in wxs.columns if x.startswith("mutation::")]
    rows = []
    for state in STATE_ORDER:
        members = membership.loc[membership.state_id == state, "case_id"].tolist()
        indices = [order.index(x) for x in members]
        local = fused[np.ix_(indices, indices)]
        case_id = members[int(np.argmax(local.mean(axis=1)))]
        mutation = [x.removeprefix("mutation::") for x in wxs.loc[case_id, wxs_cols][wxs.loc[case_id, wxs_cols] > 0].index]
        record = clinical.get(case_id, {})
        rows.append({"state_id": state, "case_id": case_id, "state_n": len(members), "age": record.get("age"),
                     "stage": record.get("stage_group"), "m_stage": record.get("m_stage"), "major_mutations": ";".join(mutation),
                     "selection_rule": "fused-affinity medoid within state"})
    result = pd.DataFrame(rows)
    write_csv(result, output_root / "representative_state_patients.csv")
    return result


def identity_card(membership, mapping, output_root):
    def state_specific(path, effect, p_value, state, clean=False):
        frame = pd.read_csv(path)
        if frame.empty:
            return []
        frame = frame[frame.feature.notna()].copy()
        frame["direction"] = np.where(
            ((frame.group_a == state) & (frame[effect] > 0)) | ((frame.group_b == state) & (frame[effect] < 0)), 1, 0
        )
        frame = frame[frame.direction == 1].sort_values(p_value)
        names = frame.feature.astype(str)
        if clean:
            names = names.str.replace("mutation::", "", regex=False)
        return names.drop_duplicates().head(4).tolist()

    cc = pd.read_csv(MAP / "clearcode34_state_by_label.csv").set_index("state_id")
    rows = []
    for state in STATE_ORDER:
        rna_names = state_specific(CHAR / "rna_hallmark_posthoc.csv", "cliffs_delta", "p_adjusted_holm", state)[:3]
        wxs_names = state_specific(CHAR / "wxs_mutation_posthoc.csv", "frequency_difference", "p_adjusted_holm", state, clean=True)[:3]
        ct_names = state_specific(CHAR / "ct_radiomics_posthoc.csv", "cliffs_delta", "p_adjusted_holm", state)[:2]
        cc_row = cc.loc[state] if state in cc.index else pd.Series(dtype=float)
        cc_total = float(cc_row.sum()) if len(cc_row) else 0
        rows.append({"state_id": state, "state_n": int((membership.state_id == state).sum()),
                     "micro_core_ids": ";".join(mapping.loc[mapping.macro_state == state, "core_id"]),
                     "top_state_specific_rna_hallmarks": ";".join(rna_names) or "none detected",
                     "top_state_specific_wxs_features": ";".join(wxs_names) or "none detected",
                     "top_state_specific_ct_features": ";".join(ct_names) or "none detected",
                     "clearcode_ccA_fraction": float(cc_row.get("ccA", np.nan) / cc_total) if cc_total else np.nan,
                     "clearcode_ccB_fraction": float(cc_row.get("ccB", np.nan) / cc_total) if cc_total else np.nan})
    result = pd.DataFrame(rows)
    write_csv(result, output_root / "state_identity_card.csv")
    return result


def run(input_root=DEFAULT_INPUT, membership_path=DEFAULT_MEMBERSHIP, output_root=ROOT / "output_kirc_v14/18_four_view_state_result_completion", force=False):
    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()) and not force:
        raise FileExistsError(f"Output exists; pass --force to overwrite: {output_root}")
    if force and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    input_root = Path(input_root)
    membership = load_membership(membership_path)
    states = load_states(input_root)
    if not set(membership.case_id).issubset(states):
        raise ValueError("State membership contains patients absent from patient_states")
    wsi, _, _ = patient_state_features(input_root, membership, output_root)
    ct, _, ct_rows = ct_summary(input_root, membership, states, output_root)
    mapping, _ = micro_macro_summary(output_root)
    noncore = noncore_summary(input_root, membership, output_root)
    wxs = pd.read_csv(input_root / "wxs/wxs_discovery_features.csv", index_col=0).reindex(membership.case_id)
    reps = representative_cases(input_root, membership, states, wxs, output_root)
    card = identity_card(membership, mapping, output_root)
    manifest = {"experiment": "four_view_state_result_completion", "input_root": str(input_root.resolve()),
                "membership_file": str(Path(membership_path).resolve()), "patient_count": int(len(membership)),
                "state_sizes": membership.groupby("state_id").size().to_dict(), "active_modalities": ["ct", "wsi", "rna", "wxs"],
                "wsi_summary": "patient-level means of tumor-selected patch class probabilities; no patch-level pseudoreplication",
                "ct_summary": "production-consistent 308-dimensional CT representation; top omnibus-FDR features exported",
                "noncore_summary": "post hoc stability explanation; non-core patients are not assigned to states",
                "outputs": {"wsi_patient_n": int(len(wsi)), "noncore_n": int(noncore.is_non_core.sum()), "representative_n": int(len(reps)), "identity_card_n": int(len(card))}}
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--membership-path", type=Path, default=DEFAULT_MEMBERSHIP)
    parser.add_argument("--output-root", type=Path, default=ROOT / "output_kirc_v14/18_four_view_state_result_completion")
    parser.add_argument("--force", action="store_true")
    print(json.dumps(run(**vars(parser.parse_args())), ensure_ascii=False, indent=2))
