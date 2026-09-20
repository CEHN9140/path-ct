from __future__ import annotations

import json
import re
from collections.abc import Mapping
from itertools import combinations
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd


def scoped_groups(scope: str, target_ids: list[str], sets: list[Mapping[str, Any]]) -> dict[str, list[str]]:
    members = {
        str(item["set_id"]): sorted(set(map(str, item["member_ids"])))
        for item in sets
    }
    if scope == "partition":
        return {"partition": sorted({case_id for values in members.values() for case_id in values})}
    if not set(target_ids).issubset(members):
        raise ValueError(f"Evidence targets are not current: {target_ids}")
    if scope == "set":
        return {target: members[target] for target in target_ids}
    pair_id = "|".join(sorted(target_ids))
    return {pair_id: sorted(set(members[target_ids[0]]) | set(members[target_ids[1]]))}


def tool_result(name: str, metrics: Mapping[str, Any], status: str = "success", reason: str = "") -> dict[str, Any]:
    return {
        "tool_name": name,
        "status": status,
        "metrics": dict(metrics),
        "warnings": [],
        "missing_reason": reason,
        "errors": [],
    }


def affinity_matrices(output_root: str) -> tuple[list[str], dict[str, np.ndarray]]:
    candidate_root = Path(output_root) / "candidate_subtype"
    patient_ids = [str(value) for value in json.loads((candidate_root / "affinity_patient_order.json").read_text())]
    matrices = {
        name: np.asarray(np.load(candidate_root / f"{name}_affinity.npy"), dtype=float)
        for name in ("ct", "wsi", "rna", "wxs")
    }
    matrices["fused"] = np.asarray(np.load(candidate_root / "fused_similarity.npy"), dtype=float)
    shape = (len(patient_ids), len(patient_ids))
    if any(matrix.shape != shape or not np.isfinite(matrix).all() for matrix in matrices.values()):
        raise ValueError("Production four-view affinity matrices do not match patient order")
    return patient_ids, matrices


def distance_kernel(distance: np.ndarray) -> np.ndarray:
    n = len(distance)
    centering = np.eye(n) - np.ones((n, n)) / n
    return -0.5 * centering @ np.square(distance) @ centering


def representation_concordance(
    patient_states_by_id: Mapping[str, Mapping[str, Any]],
    output_root: str,
    all_cluster_states: list[Mapping[str, Any]],
    scope: str,
    target_ids: list[str],
    config_dir: str = "",
) -> dict[str, Any]:
    patient_ids, matrices = affinity_matrices(output_root)
    index = {patient_id: position for position, patient_id in enumerate(patient_ids)}
    output = {}
    for key, members in scoped_groups(scope, target_ids, all_cluster_states).items():
        positions = [index[member] for member in members]
        grv = {}
        for left, right in combinations(("ct", "wsi", "rna", "wxs"), 2):
            kernels = []
            for name in (left, right):
                affinity = matrices[name][np.ix_(positions, positions)]
                distance = np.clip(1.0 - affinity / np.sqrt(np.outer(np.diag(affinity), np.diag(affinity))), 0, 1)
                kernel = distance_kernel(distance)
                kernels.append(kernel)
            denominator = np.sqrt(np.sum(kernels[0] ** 2) * np.sum(kernels[1] ** 2))
            grv[f"{left}__{right}"] = float(np.sum(kernels[0] * kernels[1]) / denominator) if denominator else None
        output[key] = {"patient_n": len(members), "pairwise_affinity_geometry_grv": grv}
    return tool_result("representation_concordance", {scope: output})


def structural_diagnostics(
    patient_states_by_id: Mapping[str, Mapping[str, Any]],
    output_root: str,
    config_dir: str,
    all_cluster_states: list[Mapping[str, Any]],
    scope: str,
    target_ids: list[str],
) -> dict[str, Any]:
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics import silhouette_score
    from utils.llm_utils import load_yaml_file

    patient_ids, matrices = affinity_matrices(output_root)
    fused = np.clip((matrices["fused"] + matrices["fused"].T) / 2, 0, 1)
    index = {patient_id: position for position, patient_id in enumerate(patient_ids)}
    settings = load_yaml_file(Path(config_dir) / "subtype_review.yaml")["cross_modal"]["structural"]
    max_children, min_size = int(settings["max_children"]), int(settings["min_child_size"])

    def eigengap(matrix: np.ndarray, max_k: int) -> dict[str, Any]:
        degree = np.maximum(matrix.sum(axis=1), np.finfo(float).eps)
        laplacian = np.eye(len(matrix)) - matrix / np.sqrt(np.outer(degree, degree))
        values = np.linalg.eigvalsh((laplacian + laplacian.T) / 2)
        gaps = {k: float(values[k] - values[k - 1]) for k in range(1, max_k + 1) if k < len(values)}
        best = max(gaps, key=gaps.get) if gaps else None
        return {
            "candidate_k": int(best) if best is not None else None,
            "candidate_eigengap": gaps.get(best) if best is not None else None,
            "eigengaps": {str(k): value for k, value in gaps.items()},
        }

    set_members = {
        str(item["set_id"]): list(map(str, item["member_ids"]))
        for item in all_cluster_states
    }
    if scope == "partition":
        internal = {}
        for set_id, members in set_members.items():
            local = fused[np.ix_([index[item] for item in members], [index[item] for item in members])]
            upper = local[np.triu_indices(len(local), 1)]
            max_k = min(max_children, len(members) // min_size)
            gaps = eigengap(local, max_k)
            internal[set_id] = {
                "member_n": len(members),
                "mean_within_affinity": float(upper.mean()) if len(upper) else None,
                "screen_candidate_k": gaps["candidate_k"],
                "candidate_eigengap": gaps["candidate_eigengap"],
                "eigengaps": gaps["eigengaps"],
            }
        boundaries = []
        for left, right in combinations(sorted(set_members), 2):
            cross = fused[np.ix_([index[item] for item in set_members[left]], [index[item] for item in set_members[right]])]
            boundaries.append({"target_ids": [left, right], "mean_between_affinity": float(cross.mean())})
        neighbors = int(settings["nearest_merge_neighbors"])
        selected = {
            tuple(row["target_ids"])
            for set_name in set_members
            for row in sorted(
                (row for row in boundaries if set_name in row["target_ids"]),
                key=lambda row: (-row["mean_between_affinity"], row["target_ids"]),
            )[:neighbors]
        }
        boundaries = [row for row in boundaries if tuple(row["target_ids"]) in selected]
        boundaries.sort(key=lambda row: (-row["mean_between_affinity"], row["target_ids"]))
        return tool_result("structural_diagnostics", {"partition": {
            "internal_structure": internal,
            "nearest_pair_targets": [row["target_ids"] for row in boundaries],
            "nearest_pair_affinities": boundaries,
        }})

    members = scoped_groups(scope, target_ids, all_cluster_states)
    output = {}
    for key, case_ids in members.items():
        local = fused[np.ix_([index[item] for item in case_ids], [index[item] for item in case_ids])]
        max_k = min(max_children, len(case_ids) // min_size)
        spectrum = eigengap(local, max_k)
        if scope == "set":
            solutions = {}
            for k in range(2, max_k + 1):
                labels = SpectralClustering(
                    n_clusters=k, affinity="precomputed", assign_labels="cluster_qr", random_state=0
                ).fit_predict(local)
                sizes = np.bincount(labels, minlength=k)
                if int(sizes.min()) < min_size or len(np.unique(labels)) != k:
                    continue
                distance = np.clip(1.0 - local, 0, 1)
                np.fill_diagonal(distance, 0)
                solutions[str(k)] = {
                    "silhouette": float(silhouette_score(distance, labels, metric="precomputed")),
                    "child_sizes": sorted(map(int, sizes)),
                }
            output[key] = {
                "member_n": len(case_ids),
                "screen_candidate_k": spectrum["candidate_k"],
                "candidate_eigengap": spectrum["candidate_eigengap"],
                "eigengaps": spectrum["eigengaps"],
                "solutions": solutions,
            }
        else:
            left, right = (set_members[target] for target in target_ids)
            labels = np.asarray([0 if item in left else 1 for item in case_ids])
            distance = np.clip(1.0 - local, 0, 1)
            np.fill_diagonal(distance, 0)
            within = [
                local[a, b] for a, b in combinations(range(len(case_ids)), 2)
                if (case_ids[a] in left) == (case_ids[b] in left)
            ]
            between = [
                local[a, b] for a, b in combinations(range(len(case_ids)), 2)
                if (case_ids[a] in left) != (case_ids[b] in left)
            ]
            within_mean = float(np.mean(within)) if within else None
            between_mean = float(np.mean(between)) if between else None
            boundary_silhouette = float(silhouette_score(distance, labels, metric="precomputed"))
            output[key] = {
                "member_n": len(case_ids),
                "mean_within_affinity": within_mean,
                "mean_between_affinity": between_mean,
                "boundary_silhouette": boundary_silhouette,
                "union_eigengap": spectrum,
            }
    return tool_result("structural_diagnostics", {scope: output})


def rna_pathway_enrichment(
    patient_states_by_id: Mapping[str, Mapping[str, Any]],
    output_root: str,
    config_dir: str,
    all_cluster_states: list[Mapping[str, Any]],
    scope: str,
    target_ids: list[str],
) -> dict[str, Any]:
    from scipy.stats import ttest_ind
    from gseapy import prerank
    from statsmodels.stats.multitest import multipletests
    from utils.llm_utils import load_yaml_file

    all_members = {
        str(member) for item in all_cluster_states for member in item["member_ids"]
    }
    missing_states = sorted(all_members - set(patient_states_by_id))
    if missing_states:
        raise ValueError(f"RNA biological-support tool is missing patient states: {missing_states}")

    rna_paths = set()
    missing_paths = []
    for case_id in sorted(all_members):
        omics = dict(patient_states_by_id[case_id].get("omics_evidence", {}) or {})
        path = str(omics.get("rna_pathway_feature_path", "") or "")
        if path:
            rna_paths.add(path)
        else:
            missing_paths.append(case_id)
    if missing_paths:
        raise ValueError(
            "Candidate-cohort patient states are missing rna_pathway_feature_path: "
            f"{missing_paths}"
        )
    if len(rna_paths) != 1:
        raise ValueError(
            "Candidate-cohort patient states must reference exactly one RNA pathway "
            f"feature artifact, found: {sorted(rna_paths)}"
        )

    frame = pd.read_csv(next(iter(rna_paths))).set_index("case_id")
    frame.index = frame.index.map(str)
    missing_rna = sorted(all_members - set(frame.index))
    if missing_rna:
        raise ValueError(
            "RNA pathway feature matrix does not cover the full candidate cohort: "
            f"{missing_rna}"
        )
    gene_settings = load_yaml_file(Path(config_dir) / "subtype_review.yaml")["rna"]
    gene_sets_path = gene_settings["hallmark_gene_sets_path"]
    gene_sets = {}
    for line in Path(gene_sets_path).read_text(encoding="utf-8").splitlines():
        fields = line.rstrip().split("\t")
        if len(fields) > 2:
            gene_sets[fields[0]] = set(fields[2:]) & set(frame.columns)
    min_overlap = int(gene_settings["min_pathway_overlap"])
    gene_sets = {name: genes for name, genes in gene_sets.items() if len(genes) >= min_overlap}
    results = {}
    for set_id in target_ids:
        members = set(next(item["member_ids"] for item in all_cluster_states if item["set_id"] == set_id))
        case_ids = sorted(all_members)
        group = frame.loc[[case_id for case_id in case_ids if case_id in members]]
        rest = frame.loc[[case_id for case_id in case_ids if case_id not in members]]
        set_n, rest_n = len(group), len(rest)
        if set_n < 2 or rest_n < 2:
            results[set_id] = {
                "analysis_status": "not_estimable", "set_n": set_n, "rest_n": rest_n,
                "finite_ranked_gene_n": 0,
                "reason": "Welch t-statistics require at least two samples in both groups.",
                "pathways": [],
            }
            continue
        ranking = pd.DataFrame({
            "gene": frame.columns,
            "t": [float(ttest_ind(group[gene], rest[gene], equal_var=False, nan_policy="omit").statistic) for gene in frame.columns],
        }).replace([np.inf, -np.inf], np.nan).dropna().sort_values("t", ascending=False)
        finite_ranked_gene_n = len(ranking)
        if finite_ranked_gene_n < 2:
            results[set_id] = {
                "analysis_status": "not_estimable", "set_n": set_n, "rest_n": rest_n,
                "finite_ranked_gene_n": finite_ranked_gene_n,
                "reason": "Fewer than two finite gene statistics were available for preranked enrichment.",
                "pathways": [],
            }
            continue
        ranked_genes = set(ranking["gene"].astype(str))
        eligible_gene_sets = {
            name: sorted(genes & ranked_genes)
            for name, genes in gene_sets.items()
            if len(genes & ranked_genes) >= min_overlap
        }
        if not eligible_gene_sets:
            results[set_id] = {
                "analysis_status": "not_estimable", "set_n": set_n, "rest_n": rest_n,
                "finite_ranked_gene_n": finite_ranked_gene_n,
                "reason": "No configured pathway met the minimum overlap with finite ranked genes.",
                "pathways": [],
            }
            continue
        result = prerank(
            rnk=ranking,
            gene_sets=eligible_gene_sets,
            min_size=min_overlap,
            max_size=500,
            permutation_num=1000,
            seed=20260920,
            outdir=None,
            verbose=False,
        ).res2d
        rows = []
        for row in result.to_dict("records"):
            rows.append({
                "pathway": str(row["Term"]),
                "nes": float(row["NES"]),
                "fdr_q": float(row["FDR q-val"]),
                "direction": "up" if float(row["NES"]) > 0 else "down",
                "leading_edge_genes": str(row.get("Lead_genes", "")),
            })
        rows.sort(key=lambda item: (item["fdr_q"], -abs(item["nes"])))
        results[set_id] = {
            "analysis_status": "success", "set_n": set_n, "rest_n": rest_n,
            "finite_ranked_gene_n": finite_ranked_gene_n, "pathways": rows[:30],
        }
    return tool_result("rna_pathway_enrichment", {"set": results})


def wxs_mutation_enrichment(
    patient_states_by_id: Mapping[str, Mapping[str, Any]],
    output_root: str,
    config_dir: str,
    all_cluster_states: list[Mapping[str, Any]],
    scope: str,
    target_ids: list[str],
) -> dict[str, Any]:
    from scipy.stats import fisher_exact
    from statsmodels.stats.contingency_tables import Table2x2
    from statsmodels.stats.multitest import multipletests

    path = Path(output_root) / "wxs" / "wxs_discovery_features.csv"
    frame = pd.read_csv(path).set_index("case_id")
    frame.index = frame.index.map(str)
    features = [name for name in frame.columns if name.startswith("mutation::")]
    from utils.llm_utils import load_yaml_file

    driver_genes = {
        str(gene).upper()
        for gene in load_yaml_file(Path(config_dir) / "wxs.yaml")["biological_support"]["driver_genes"]
    }
    universe = {str(member) for item in all_cluster_states for member in item["member_ids"]}
    rows_by_set = {}
    for set_id in target_ids:
        members = set(next(item["member_ids"] for item in all_cluster_states if item["set_id"] == set_id))
        set_ids = sorted(members & universe & set(frame.index))
        rest_ids = sorted((universe - members) & set(frame.index))
        if not set_ids or not rest_ids:
            rows_by_set[set_id] = {
                "analysis_status": "not_estimable", "set_n": len(set_ids), "rest_n": len(rest_ids),
                "reason": "Fisher enrichment requires at least one observed patient in both set and rest.",
                "gene_enrichment": [],
                "driver_panel": {"configured_genes": sorted(driver_genes), "available_genes": [],
                                 "not_in_selected_features": sorted(driver_genes), "results": []},
            }
            continue
        rows = []
        for gene in features:
            set_positive = int(frame.loc[set_ids, gene].astype(bool).sum())
            rest_positive = int(frame.loc[rest_ids, gene].astype(bool).sum())
            table = np.asarray([
                [set_positive, len(set_ids) - set_positive],
                [rest_positive, len(rest_ids) - rest_positive],
            ])
            odds_ratio, p_value = fisher_exact(table)
            ci = Table2x2(table).oddsratio_confint()
            gene_name = gene.removeprefix("mutation::")
            rows.append({
                "gene": gene_name,
                "driver_panel_member": gene_name.upper() in driver_genes,
                "set_mutated_n": set_positive,
                "set_n": len(set_ids),
                "rest_mutated_n": rest_positive,
                "rest_n": len(rest_ids),
                "odds_ratio": float(odds_ratio),
                "odds_ratio_ci95": [float(ci[0]), float(ci[1])],
                "p_value": float(p_value),
                "q_value": None,
            })
        q_values = multipletests([row["p_value"] for row in rows], method="fdr_bh")[1]
        for row, q_value in zip(rows, q_values):
            row["q_value"] = float(q_value)
        rows.sort(key=lambda item: (item["q_value"], -abs(item["odds_ratio"] - 1)))
        available_drivers = {row["gene"].upper() for row in rows} & driver_genes
        rows_by_set[set_id] = {
            "analysis_status": "success", "set_n": len(set_ids), "rest_n": len(rest_ids),
            "gene_enrichment": rows,
            "driver_panel": {
                "configured_genes": sorted(driver_genes),
                "available_genes": sorted(available_drivers),
                "not_in_selected_features": sorted(driver_genes - available_drivers),
                "results": [row for row in rows if row["driver_panel_member"]],
            },
        }
    return tool_result("wxs_mutation_enrichment", {"set": rows_by_set})


def clinical_labels(patient_states_by_id: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, str]]:
    result = {}
    for case_id, state in patient_states_by_id.items():
        clinical = dict(state.get("inventory", {}).get("Clinical", {}) or {})
        diagnoses = [dict(row) for row in clinical.get("diagnoses", [])]
        primary = [row for row in diagnoses if str(row.get("diagnosis_is_primary_disease", "")).lower() == "true"]
        diagnosis = (primary or diagnoses or [{}])[0]
        stage_text = str(diagnosis.get("ajcc_pathologic_stage", "")).upper()
        stage = next((value for value in ("IV", "III", "II", "I") if f"STAGE {value}" in stage_text or stage_text == value), "")
        t_match = re.search(r"T([1-4])", str(diagnosis.get("ajcc_pathologic_t", "")).upper())
        m_text = str(diagnosis.get("ajcc_pathologic_m", "")).upper()
        m_stage = m_text if m_text in {"M0", "M1"} else ""
        grade_match = re.search(r"G([1-4])", str(diagnosis.get("tumor_grade", "")).upper())
        result[str(case_id)] = {
            "stage": stage,
            "grade": f"G{grade_match.group(1)}" if grade_match else "",
            "t_stage": f"T{t_match.group(1)}" if t_match else "",
            "m_stage": m_stage,
        }
    return result


def cramers_v(table: np.ndarray) -> float:
    from scipy.stats import chi2_contingency

    observed = np.asarray(table, dtype=float)
    n = observed.sum()
    rows, cols = observed.shape
    if n <= 1 or min(rows, cols) <= 1:
        return 0.0
    chi2 = float(chi2_contingency(observed, correction=False)[0])
    phi2 = chi2 / n
    correction = (cols - 1) * (rows - 1) / (n - 1)
    phi2 = max(0.0, phi2 - correction)
    rows_adj = rows - (rows - 1) ** 2 / (n - 1)
    cols_adj = cols - (cols - 1) ** 2 / (n - 1)
    return float(np.sqrt(phi2 / min(rows_adj - 1, cols_adj - 1))) if min(rows_adj, cols_adj) > 1 else 0.0


def categorical_test(groups: list[str], values: list[str], permutations: int, seed: int, *, force_fisher: bool = False) -> dict[str, Any]:
    from scipy.stats import chi2_contingency

    table = pd.crosstab(pd.Series(groups, name="group"), pd.Series(values, name="label"))
    if table.shape[0] < 2 or table.shape[1] < 2:
        return {"test": "not_estimable", "p_value": None, "cramers_v": 0.0, "contingency": table.to_dict()}
    observed = float(chi2_contingency(table.to_numpy(), correction=False)[0])
    expected = chi2_contingency(table.to_numpy(), correction=False)[3]
    if force_fisher:
        if table.shape != (2, 2):
            raise ValueError("Fisher exact test requires a 2x2 table")
        from scipy.stats import fisher_exact

        p_value = float(fisher_exact(table.to_numpy()).pvalue)
        test = "fisher_exact"
    elif np.any(expected < 5):
        rng = np.random.default_rng(seed)
        group_values = np.asarray(groups)
        factor_values = np.asarray(values)
        exceed = 0
        for _ in range(permutations):
            shuffled = rng.permutation(group_values)
            permuted = pd.crosstab(shuffled, factor_values).reindex(index=table.index, columns=table.columns, fill_value=0)
            exceed += float(chi2_contingency(permuted, correction=False)[0]) >= observed
        p_value = (exceed + 1) / (permutations + 1)
        test = "permutation_chi_square"
    else:
        p_value = float(chi2_contingency(table.to_numpy(), correction=False)[1])
        test = "pearson_chi_square"
    return {"test": test, "p_value": p_value, "cramers_v": cramers_v(table.to_numpy()), "contingency": table.to_dict()}


def known_label_echo(
    patient_states_by_id: Mapping[str, Mapping[str, Any]],
    output_root: str,
    all_cluster_states: list[Mapping[str, Any]],
    config_dir: str,
    scope: str,
    target_ids: list[str],
) -> dict[str, Any]:
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
    from utils.llm_utils import load_yaml_file

    state_by_patient = {
        str(patient_id): str(item["set_id"])
        for item in all_cluster_states
        for patient_id in item["member_ids"]
    }
    labels = clinical_labels(patient_states_by_id)
    config = load_yaml_file(Path(config_dir) / "subtype_review.yaml")["known_label_echo"]
    clinical_rows = {}
    for name in ("stage", "grade", "t_stage", "m_stage"):
        pairs = [(case_id, state_by_patient[case_id], row[name]) for case_id, row in labels.items() if case_id in state_by_patient and row[name]]
        if name == "m_stage":
            exact = {}
            for state_id in sorted(set(state_by_patient.values())):
                target = [value for case_id, group, value in pairs if group == state_id]
                rest = [value for case_id, group, value in pairs if group != state_id]
                exact[state_id] = categorical_test(
                    [state_id] * len(target) + ["rest"] * len(rest),
                    target + rest,
                    int(config["permutations"]),
                    20260920,
                    force_fisher=True,
                )
            clinical_rows[name] = {
                "available_n": len(pairs),
                "missing_n": len(state_by_patient) - len(pairs),
                "m0_m1_fisher_by_set": exact,
                "contingency": categorical_test([row[1] for row in pairs], [row[2] for row in pairs], int(config["permutations"]), 20260920),
            }
        else:
            result = categorical_test(
                [row[1] for row in pairs], [row[2] for row in pairs],
                int(config["permutations"]), 20260920,
            )
            clinical_rows[name] = {"available_n": len(pairs), "missing_n": len(state_by_patient) - len(pairs), **result}
    taxonomy = {}
    for name, filename in (("tcga_m1_m4", config["mrna_m1_m4_path"]), ("clearcode34", config["clearcode34_path"])):
        frame = pd.read_csv(filename)
        frame = frame[(frame["reference_status"] == "matched") & frame["reference_subtype"].notna()]
        frame = frame[frame["case_id"].astype(str).isin(state_by_patient)].copy()
        candidate = [state_by_patient[str(case_id)] for case_id in frame["case_id"]]
        known = frame["reference_subtype"].astype(str).tolist()
        table = pd.crosstab(pd.Series(candidate, name="state"), pd.Series(known, name="known_label"))
        taxonomy[name] = {
            "reference_n": len(frame),
            "contingency": table.to_dict(),
            "ari": float(adjusted_rand_score(candidate, known)),
            "ami": float(adjusted_mutual_info_score(candidate, known)),
            "limitation": "Expression-derived known taxonomy overlaps the RNA discovery view and is correspondence analysis, not independent validation or an accept/drop gate.",
        }
    return tool_result("known_label_echo", {"partition": {"clinical": clinical_rows, "molecular_taxonomy": taxonomy}})


def technical_values(patient_states_by_id: Mapping[str, Mapping[str, Any]], output_root: str) -> dict[str, dict[str, Any]]:
    values = {}
    root = Path(output_root)
    for case_id, state in patient_states_by_id.items():
        inventory = dict(state.get("inventory", {}) or {})
        ct_records = [dict(row) for row in inventory.get("CT", [])]
        qc_root = root / "ct_qc" / case_id
        summary_path = qc_root / "selection_summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
        series = dict(summary.get("selected_series", {}) or {})
        selected = dict((summary.get("selected_files") or [{}])[0])
        sidecar_path = qc_root / "dcm2nii" / f"{selected.get('selected_ct_id', '')}.json"
        sidecar = json.loads(sidecar_path.read_text()) if sidecar_path.is_file() else {}
        source = str(selected.get("selected_source_file", ""))
        record = next((row for row in ct_records if str(row.get("File Path", "")) == source), {})
        case_parts = str(case_id).split("-")
        date = str(record.get("Study Date", ""))
        year_match = re.search(r"(?:19|20)\d{2}", date)
        spacing = series.get("pixel_spacing_row") or record.get("Pixel Spacing Row")
        if not spacing:
            spacing_text = str(record.get("PixelSpacing", record.get("Pixel Spacing", ""))).replace("\\", "/")
            spacing = spacing_text.split("/")[0] if spacing_text else ""
        def number(value: Any) -> float | None:
            try:
                parsed = float(value)
                return parsed if np.isfinite(parsed) else None
            except (TypeError, ValueError):
                return None
        values[str(case_id)] = {
            "tissue_source_site": case_parts[1] if len(case_parts) > 2 and case_parts[0] == "TCGA" else "",
            "ct_phase": str(series.get("inferred_phase", "") or ""),
            "ct_manufacturer": str(sidecar.get("Manufacturer", record.get("Manufacturer", "")) or ""),
            "ct_scanner_model": str(sidecar.get("ManufacturersModelName", sidecar.get("ManufacturerModelName", "")) or ""),
            "ct_reconstruction_kernel": str(sidecar.get("ConvolutionKernel", "") or ""),
            "ct_slice_thickness": number(series.get("slice_thickness_median", record.get("Slice Thickness"))),
            "ct_pixel_spacing": number(spacing),
            "ct_z_spacing": number(series.get("z_spacing_median", record.get("Spacing Between Slices"))),
            "study_year": int(year_match.group()) if year_match else None,
        }
    return values


def confounder_association(
    patient_states_by_id: Mapping[str, Mapping[str, Any]],
    output_root: str,
    config_dir: str,
    all_cluster_states: list[Mapping[str, Any]],
    scope: str,
    target_ids: list[str],
) -> dict[str, Any]:
    from scipy.stats import kruskal
    from statsmodels.stats.multitest import multipletests
    from utils.llm_utils import load_yaml_file

    scoped = scoped_groups(scope, target_ids, all_cluster_states)
    universe = {str(member) for item in all_cluster_states for member in item["member_ids"]}
    if scope == "set":
        analyses = {
            target: {target: members, "rest": sorted(universe - set(members))}
            for target, members in scoped.items()
        }
    elif scope == "partition":
        analyses = {"partition": {
            str(item["set_id"]): sorted(set(map(str, item["member_ids"])))
            for item in all_cluster_states
        }}
    else:
        analyses = {scope: scoped}
    factors = technical_values(patient_states_by_id, output_root)
    permutations = int(load_yaml_file(Path(config_dir) / "subtype_review.yaml")["confounder"]["permutations"])
    result_by_group = {}
    for group_name, groups in analyses.items():
        group_by_patient = {case_id: key for key, members in groups.items() for case_id in members}
        rows = []
        for factor in ("tissue_source_site", "ct_phase", "ct_manufacturer", "ct_scanner_model", "ct_reconstruction_kernel", "ct_slice_thickness", "ct_pixel_spacing", "ct_z_spacing", "study_year"):
            observations = [(case_id, group_by_patient[case_id], values.get(factor)) for case_id, values in factors.items() if case_id in group_by_patient and values.get(factor) not in (None, "")]
            if not observations:
                continue
            group_labels = [row[1] for row in observations]
            factor_values = [str(row[2]) if isinstance(row[2], str) else float(row[2]) for row in observations]
            if isinstance(factor_values[0], str):
                test = categorical_test(group_labels, factor_values, permutations, 20260920)
                rows.append({"factor": factor, "type": "categorical", "n": len(observations), **test})
            else:
                levels = sorted(set(group_labels))
                samples = [[float(row[2]) for row in observations if row[1] == level] for level in levels]
                if len(samples) < 2:
                    rows.append({"factor": factor, "type": "numeric", "n": len(observations), "test": "not_estimable", "p_value": None, "epsilon_squared": None})
                else:
                    statistic, p_value = kruskal(*samples)
                    n, k = len(observations), len(samples)
                    epsilon = max(0.0, (float(statistic) - k + 1) / (n - k)) if n > k else 0.0
                    rows.append({"factor": factor, "type": "numeric", "n": n, "test": "kruskal_wallis", "p_value": float(p_value), "epsilon_squared": float(epsilon)})
        tested_rows = [row for row in rows if row["p_value"] is not None]
        if tested_rows:
            q_values = multipletests([row["p_value"] for row in tested_rows], method="fdr_bh")[1]
            for row, q_value in zip(tested_rows, q_values):
                row["q_value"] = float(q_value)
        result_by_group[group_name] = rows
    return tool_result("confounder_association", {scope: result_by_group})


def confounder_representation_effect(
    patient_states_by_id: Mapping[str, Mapping[str, Any]],
    output_root: str,
    config_dir: str,
    all_cluster_states: list[Mapping[str, Any]],
    scope: str,
    target_ids: list[str],
) -> dict[str, Any]:
    from statsmodels.stats.multitest import multipletests
    from utils.llm_utils import load_yaml_file

    patient_ids, matrices = affinity_matrices(output_root)
    index = {case_id: position for position, case_id in enumerate(patient_ids)}
    factors = technical_values(patient_states_by_id, output_root)
    settings = load_yaml_file(Path(config_dir) / "subtype_review.yaml")["confounder"]
    permutations = int(settings["permutations"])
    categorical_factors = ("tissue_source_site", "ct_phase", "ct_manufacturer", "ct_scanner_model", "ct_reconstruction_kernel")
    continuous_factors = ("ct_slice_thickness", "ct_pixel_spacing", "ct_z_spacing", "study_year")
    results = {}
    for factor in categorical_factors:
        available = [case_id for case_id in patient_ids if factors.get(case_id, {}).get(factor) not in (None, "")]
        labels = np.asarray([str(factors[case_id][factor]) for case_id in available])
        if len(np.unique(labels)) < 2:
            results[factor] = {
                "modality": "fused" if factor == "tissue_source_site" else "ct",
                "variable_type": "categorical", "test": "not_estimable",
                "n": len(labels), "levels": int(len(np.unique(labels))),
                "permutation_p": None, "q_value": None,
            }
            continue
        modality = "fused" if factor in {"tissue_source_site"} else "ct"
        positions = [index[item] for item in available]
        affinity = matrices[modality][np.ix_(positions, positions)]
        distance = np.clip(1 - affinity, 0, 1)
        kernel = distance_kernel(distance)
        n = len(labels)
        levels = np.unique(labels)
        total = float(np.trace(kernel))
        def pseudo_f(groups: np.ndarray) -> tuple[float, float]:
            between = sum(float(kernel[np.ix_(groups == level, groups == level)].sum()) / int(np.sum(groups == level)) for level in np.unique(groups))
            residual = total - between
            df_group = len(np.unique(groups)) - 1
            df_resid = n - len(np.unique(groups))
            r2 = between / total if total > 0 else 0.0
            statistic = (between / df_group) / (residual / df_resid) if df_group > 0 and df_resid > 0 and residual > 0 else 0.0
            return r2, statistic
        r2, observed = pseudo_f(labels)
        rng = np.random.default_rng(20260920)
        exceed = sum(pseudo_f(rng.permutation(labels))[1] >= observed for _ in range(permutations))
        results[factor] = {
            "modality": modality,
            "variable_type": "categorical",
            "test": "permutation_permanova",
            "n": n,
            "levels": int(len(levels)),
            "r_squared": float(r2),
            "permutation_p": (exceed + 1) / (permutations + 1),
        }
    for factor in continuous_factors:
        available = [
            case_id for case_id in patient_ids
            if factors.get(case_id, {}).get(factor) is not None
            and np.isfinite(factors[case_id][factor])
        ]
        values = np.asarray([float(factors[case_id][factor]) for case_id in available])
        if len(values) < 3 or len(np.unique(values)) < 2:
            results[factor] = {
                "modality": "ct", "variable_type": "continuous", "test": "not_estimable",
                "n": len(values), "permutation_p": None, "q_value": None,
            }
            continue
        positions = [index[item] for item in available]
        affinity = matrices["ct"][np.ix_(positions, positions)]
        kernel = distance_kernel(np.clip(1 - affinity, 0, 1))
        centered = values - values.mean()
        total = float(np.trace(kernel))
        if total <= 0:
            results[factor] = {
                "modality": "ct", "variable_type": "continuous", "test": "not_estimable",
                "n": len(values), "permutation_p": None, "q_value": None,
            }
            continue
        model_ss = float(centered @ kernel @ centered / (centered @ centered))
        residual_ss = total - model_ss
        observed = np.inf if residual_ss <= 0 else model_ss / (residual_ss / (len(values) - 2))
        rng = np.random.default_rng(20260920)
        exceed = 0
        for _ in range(permutations):
            shuffled = rng.permutation(centered)
            permuted_ss = float(shuffled @ kernel @ shuffled / (shuffled @ shuffled))
            permuted_residual_ss = total - permuted_ss
            permuted_f = np.inf if permuted_residual_ss <= 0 else permuted_ss / (permuted_residual_ss / (len(values) - 2))
            exceed += permuted_f >= observed
        results[factor] = {
            "modality": "ct",
            "variable_type": "continuous",
            "test": "permutation_distance_based_regression",
            "n": len(values),
            "r_squared": model_ss / total,
            "permutation_p": (exceed + 1) / (permutations + 1),
        }
    tested = [row for row in results.values() if row["permutation_p"] is not None]
    if tested:
        for row, q_value in zip(tested, multipletests([row["permutation_p"] for row in tested], method="fdr_bh")[1]):
            row["q_value"] = float(q_value)
    return tool_result("confounder_representation_effect", {"partition": results})


TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    "representation_concordance": {
        "aspect": "affinity_geometry_concordance",
        "dimension": "cross_modal_consistency", "scopes": ("set", "pair", "partition"),
        "description": "Compare four-view affinity/network geometries using descriptive Generalized RV; this is not a test on native feature distances.",
        "function": representation_concordance,
    },
    "structural_diagnostics": {
        "aspect": "structural_diagnostics",
        "dimension": "cross_modal_consistency", "scopes": ("set", "pair", "partition"),
        "description": "Return partition triage measurements or requested set/pair structural measurements and feasible spectral solutions.",
        "function": structural_diagnostics,
    },
    "rna_pathway_enrichment": {
        "aspect": "rna_pathway_enrichment",
        "dimension": "biological_support", "scopes": ("set",),
        "description": "Run Hallmark preranked GSEA on the full filtered log2 RNA transcriptome for each requested set versus rest.",
        "function": rna_pathway_enrichment,
    },
    "wxs_mutation_enrichment": {
        "aspect": "wxs_mutation_enrichment",
        "dimension": "biological_support", "scopes": ("set",),
        "description": "Run Fisher exact mutation enrichment with odds ratios, confidence intervals, and BH-FDR for requested sets.",
        "function": wxs_mutation_enrichment,
    },
    "known_label_echo": {
        "aspect": "known_label_echo",
        "dimension": "known_label_echo", "scopes": ("partition",),
        "description": "Compare the full partition with stage, grade, major T/M stage, TCGA m1-m4, and ClearCode34 labels.",
        "function": known_label_echo,
    },
    "confounder_association": {
        "aspect": "confounder_association",
        "dimension": "confounder_exclusion", "scopes": ("set", "partition"),
        "description": "Test association of candidate membership with measured site and CT acquisition metadata.",
        "function": confounder_association,
    },
    "confounder_representation_effect": {
        "aspect": "confounder_representation_effect",
        "dimension": "confounder_exclusion", "scopes": ("partition",),
        "description": "Estimate permutation-based distance-model variance explained by measured technical factors in the related affinity representation.",
        "function": confounder_representation_effect,
    },
}


def compact_tool_result(raw: Mapping[str, Any], tool_name: str) -> dict[str, Any]:
    metrics = dict(raw.get("metrics", {}) or {})
    refs = []
    def collect(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                collect(item, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                collect(item, f"{path}[{index}]")
        else:
            refs.append(path)
    collect(metrics, f"tool_results.{tool_name}.metrics")
    return {
        "tool_name": tool_name,
        "status": str(raw["status"]),
        "metrics": metrics,
        "metric_refs": refs,
        "warnings": list(raw.get("warnings", [])),
        "missing_reason": str(raw.get("missing_reason", "")),
        "errors": list(raw.get("errors", [])),
    }


def build_validation_tools(registry: Mapping[str, Mapping[str, Any]] | None = None) -> list[Any]:
    from langchain_core.tools import tool

    available = registry or TOOL_REGISTRY
    result = []
    for name, metadata in available.items():
        def request_tool(scope: Literal["set", "pair", "partition"], target_ids: list[str]) -> str:
            return f"Python will calculate the requested analysis for {scope} targets {target_ids}."
        result.append(tool(name, description=metadata["description"])(request_tool))
    return result
