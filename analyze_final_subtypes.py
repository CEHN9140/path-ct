from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, fisher_exact
from statsmodels.stats.multitest import multipletests

BASE_SEED = 20260923
MODALITIES = ("ct", "wsi", "rna", "wxs")


def build_subtype_contrast(context: Mapping[str, Any], subtype_id: str, reference_population: str) -> tuple[list[str], list[str]]:
    if reference_population == "core_rest":
        universe = list(context["core_patient_ids"])
    elif reference_population == "candidate_rest":
        universe = list(context["candidate_patient_ids"])
    else:
        raise ValueError(f"Unknown reference population: {reference_population}")
    target = [patient for patient in universe if patient in context["subtypes"][subtype_id]]
    rest = [patient for patient in universe if patient not in target]
    if set(target) & set(rest) or len(target) + len(rest) != len(universe):
        raise ValueError(f"Invalid {reference_population} contrast for {subtype_id}")
    return target, rest


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def bh(values: list[float | None]) -> list[float | None]:
    result = [None] * len(values)
    valid = [(i, float(v)) for i, v in enumerate(values) if v is not None and np.isfinite(v)]
    if valid:
        result_values = multipletests([v for _, v in valid], method="fdr_bh")[1]
        for (i, _), value in zip(valid, result_values):
            result[i] = float(value)
    return result


def holm(values: list[float | None]) -> list[float | None]:
    result = [None] * len(values)
    valid = [(i, float(v)) for i, v in enumerate(values) if v is not None and np.isfinite(v)]
    if valid:
        result_values = multipletests([v for _, v in valid], method="holm")[1]
        for (i, _), value in zip(valid, result_values):
            result[i] = float(value)
    return result


def cliffs_delta(left: list[float], right: list[float]) -> float | None:
    if not left or not right:
        return None
    a, b = np.asarray(left), np.asarray(right)
    return float((np.greater.outer(a, b).sum() - np.less.outer(a, b).sum()) / (len(a) * len(b)))


def bootstrap_delta(left: list[float], right: list[float], iterations: int, seed: int) -> tuple[float | None, float | None, float | None]:
    observed = cliffs_delta(left, right)
    if observed is None or iterations <= 0:
        return observed, None, None
    rng = np.random.default_rng(seed)
    a, b = np.asarray(left), np.asarray(right)
    values = np.empty(iterations)
    for i in range(iterations):
        values[i] = cliffs_delta(rng.choice(a, len(a), replace=True).tolist(), rng.choice(b, len(b), replace=True).tolist())
    return observed, float(np.quantile(values, .025)), float(np.quantile(values, .975))


def load_states(output_root: Path) -> dict[str, dict[str, Any]]:
    path = output_root / "storage" / "patient_states" / "patient_states.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    states = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("qc") != "success":
            continue
        case_id = str(row["case_id"])
        if case_id in states:
            raise ValueError(f"Duplicate patient state: {case_id}")
        states[case_id] = row
    return states


def load_final_subtype_context(output_root: Path) -> dict[str, Any]:
    multi = output_root / "subtype_review" / "multi_k"
    subtype_path = multi / "patient_recurrence_subtypes.json"
    summary_path = multi / "aggregation_summary.json"
    manifest_path = multi / "aggregation_manifest.json"
    membership_path = multi / "patient_recurrence_membership.csv"
    for path in (subtype_path, summary_path, manifest_path, membership_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete" or summary.get("analysis_type") != "patient_recurrence_subtype_clustering":
        raise ValueError("multi-K output is not a complete patient-recurrence analysis")
    records = json.loads(subtype_path.read_text(encoding="utf-8"))
    subtypes = {}
    for row in records:
        subtype_id = str(row["subtype_id"])
        members = [str(x) for x in row.get("member_ids", [])]
        if not members or len(members) != int(row.get("member_count", len(members))):
            raise ValueError(f"Invalid membership for {subtype_id}")
        if len(members) != len(set(members)):
            raise ValueError(f"Duplicate patient in {subtype_id}")
        if set(members) & set().union(*(set(v) for v in subtypes.values())):
            raise ValueError(f"Subtype memberships overlap at {subtype_id}")
        subtypes[subtype_id] = members
    membership = pd.read_csv(membership_path, dtype=str)
    expected = {(sid, pid) for sid, members in subtypes.items() for pid in members}
    actual = {(str(row.subtype_id), str(row.patient_id)) for row in membership.itertuples()}
    if expected != actual:
        raise ValueError("patient_recurrence_membership.csv disagrees with canonical JSON")
    order_path = output_root / "candidate_subtype" / "affinity_patient_order.json"
    candidate_ids = [str(x) for x in json.loads(order_path.read_text(encoding="utf-8"))]
    core = [pid for members in subtypes.values() for pid in members]
    if not set(core).issubset(candidate_ids):
        raise ValueError("Final subtype patient is outside candidate cohort")
    states = load_states(output_root)
    missing = sorted(set(candidate_ids) - set(states))
    if missing:
        raise ValueError(f"Candidate cohort lacks patient states: {missing[:10]}")
    return {
        "subtypes": subtypes,
        "subtype_order": list(subtypes),
        "core_patient_ids": core,
        "candidate_patient_ids": candidate_ids,
        "noncore_patient_ids": sorted(set(candidate_ids) - set(core)),
        "subtype_metadata": {str(r["subtype_id"]): r for r in records},
        "aggregation_summary": summary,
        "aggregation_manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
        "patient_recurrence_pair_similarity": multi / "patient_recurrence_pair_similarity.csv",
        "patient_states": states,
        "source_paths": {"subtypes": subtype_path, "membership": membership_path, "summary": summary_path, "manifest": manifest_path, "pair_similarity": multi / "patient_recurrence_pair_similarity.csv", "order": order_path},
    }


def recurrence_analysis(context: Mapping[str, Any], out: Path) -> pd.DataFrame:
    pairs = pd.read_csv(context["patient_recurrence_pair_similarity"], dtype={"patient_id_left": str, "patient_id_right": str})
    rows = []
    for subtype_id, members in context["subtypes"].items():
        local = pairs[pairs.patient_id_left.isin(members) & pairs.patient_id_right.isin(members)]
        meta = context["subtype_metadata"][subtype_id]
        rows.append({"subtype_id": subtype_id, "member_count": len(members), "common_accept_set_count": meta["common_accept_set_count"], "occurrence_frequency": meta.get("occurrence_frequency"), "supporting_k_count": meta["supporting_k_count"], "supporting_ks": json.dumps(meta["supporting_ks"]), "supporting_run_count": len(meta.get("supporting_runs", [])), "min_pair_recurrence_similarity": float(local.similarity.min()) if not local.empty else 1.0, "mean_pair_recurrence_similarity": float(local.similarity.mean()) if not local.empty else 1.0, "median_pair_recurrence_similarity": float(local.similarity.median()) if not local.empty else 1.0})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "subtype_recurrence_summary.csv", index=False)
    return frame


def representation_analysis(context: Mapping[str, Any], output_root: Path, out: Path, config: Mapping[str, Any]) -> None:
    from agents.subtype_review.tools import modality_distance_matrices
    from skbio import DistanceMatrix
    from skbio.stats.distance import permanova, permdisp
    ids, matrices = modality_distance_matrices(str(output_root))
    if ids != context["candidate_patient_ids"]:
        raise ValueError("Native distance patient order differs from affinity_patient_order.json")
    core = context["core_patient_ids"]
    idx = [ids.index(x) for x in core]
    labels = np.asarray([next(sid for sid, m in context["subtypes"].items() if x in m) for x in core])
    rows, pair_rows = [], []
    permutations = int(config.get("cross_modal", {}).get("concordance", {}).get("permutations", 999))
    for modality, matrix in matrices.items():
        local = matrix[np.ix_(idx, idx)]
        local = np.asarray(local, dtype=float).copy()
        np.fill_diagonal(local, 0.0)
        dm = DistanceMatrix(local, ids=core)
        try:
            p = permanova(dm, labels, permutations=permutations)
            permanova_values = {"permanova_pseudo_f": float(p["test statistic"]), "permanova_p_value": float(p["p-value"])}
        except (ValueError, ZeroDivisionError):
            permanova_values = {"permanova_pseudo_f": None, "permanova_p_value": None}
        try:
            d = permdisp(dm, labels, permutations=permutations)
            permdisp_values = {"permdisp_f": float(d["test statistic"]), "permdisp_p_value": float(d["p-value"])}
        except (ValueError, ZeroDivisionError):
            permdisp_values = {"permdisp_f": None, "permdisp_p_value": None}
        pseudo_f = permanova_values["permanova_pseudo_f"]
        group_count = len(np.unique(labels))
        r_squared = (pseudo_f * (group_count - 1) / (pseudo_f * (group_count - 1) + len(labels) - group_count)) if pseudo_f is not None and np.isfinite(pseudo_f) else None
        rows.append({"modality": modality, "patient_n": len(core), "subtype_count": group_count, "permanova_r_squared": r_squared, **permanova_values, **permdisp_values, "analysis_role": "in_sample_discovery_space_diagnostic"})
        if permanova_values["permanova_p_value"] is None:
            continue
        for left, right in combinations(context["subtype_order"], 2):
            selected = [i for i, label in enumerate(labels) if label in (left, right)]
            pair_local = local[np.ix_(selected, selected)].copy()
            np.fill_diagonal(pair_local, 0.0)
            pair_dm = DistanceMatrix(pair_local, ids=[core[i] for i in selected])
            try:
                pp = permanova(pair_dm, labels[selected], permutations=permutations)
                pair_p = float(pp["p-value"])
            except (ValueError, ZeroDivisionError):
                pair_p = None
            pair_rows.append({"modality": modality, "subtype_a": left, "subtype_b": right, "permanova_p_value": pair_p})
    global_q = bh([r["permanova_p_value"] for r in rows])
    disp_q = bh([r["permdisp_p_value"] for r in rows])
    for r, q, dq in zip(rows, global_q, disp_q):
        r["permanova_q_value"], r["permdisp_q_value"] = q, dq
    significant_modalities = {
        r["modality"] for r in rows
        if r["permanova_q_value"] is not None and r["permanova_q_value"] < .05
    }
    pair_rows = [r for r in pair_rows if r["modality"] in significant_modalities]
    for modality in MODALITIES:
        current = [r for r in pair_rows if r["modality"] == modality]
        for r, q in zip(current, holm([x["permanova_p_value"] for x in current])): r["permanova_q_value"] = q
    pd.DataFrame(rows).to_csv(out / "representation_global.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(out / "representation_pairwise.csv", index=False)


def wsi_analysis(context: Mapping[str, Any], output_root: Path, out: Path, bootstrap_iterations: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, missing = [], []
    for case_id in context["candidate_patient_ids"]:
        path = output_root / "wsi_tumor_seg" / case_id / "patch_probabilities.csv"
        if not path.is_file():
            missing.append(case_id); continue
        patches = pd.read_csv(path)
        if "selected_tumor" in patches:
            selected_flag = patches["selected_tumor"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
            selected = patches[selected_flag]
        else:
            selected = pd.DataFrame()
        if selected.empty and "tumor_probability" in patches:
            selected = patches[patches["tumor_probability"] >= .9]
        if selected.empty:
            missing.append(case_id); continue
        values = {c: float(pd.to_numeric(selected[c], errors="coerce").mean()) for c in patches.columns if c.startswith("prob_")}
        rows.append({"case_id": case_id, **values})
    core = set(context["core_patient_ids"])
    if core - {r["case_id"] for r in rows}:
        raise ValueError(f"Final core patients missing WSI phenotype: {sorted(core - {r['case_id'] for r in rows})}")
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "wsi_patient_phenotypes.csv", index=False)
    features = [c for c in frame.columns if c.startswith("prob_")]
    groups = {sid: [x for x in members if x in set(frame.case_id)] for sid, members in context["subtypes"].items()}
    table = frame.set_index("case_id").to_dict("index")
    omnibus, pairwise, _ = continuous_tests(table, features, groups, context["core_patient_ids"], bootstrap_iterations, "wsi", "core_rest")
    omnibus_frame = pd.DataFrame(omnibus); omnibus_frame["analysis_role"] = "auxiliary_segmentation_classifier_characterization"
    pairwise_frame = pd.DataFrame(pairwise); pairwise_frame["analysis_role"] = "auxiliary_segmentation_classifier_characterization"
    omnibus_frame.to_csv(out / "wsi_segmentation_probability_omnibus.csv", index=False)
    pairwise_frame.to_csv(out / "wsi_segmentation_probability_pairwise.csv", index=False)
    for reference, ids in (("core_rest", context["core_patient_ids"]), ("candidate_rest", context["candidate_patient_ids"])):
        _, _, current = continuous_tests(table, features, groups, ids, bootstrap_iterations, "wsi", reference)
        current_frame = pd.DataFrame(current); current_frame["analysis_role"] = "auxiliary_segmentation_classifier_characterization"
        current_frame.to_csv(out / f"wsi_segmentation_probability_vs_{reference}.csv", index=False)
    frame.to_csv(out / "wsi_segmentation_probability_patient.csv", index=False)
    return pd.read_csv(out / "wsi_segmentation_probability_vs_core_rest.csv"), frame


def continuous_tests(table: Mapping[str, Mapping[str, Any]], features: list[str], groups: Mapping[str, list[str]], candidate_ids: list[str], bootstrap_iterations: int, prefix: str, reference_population: str = "core_rest") -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    omnibus = []
    for feature in features:
        samples = [[float(table[x][feature]) for x in members if x in table and pd.notna(table[x].get(feature))] for members in groups.values()]
        p = None
        if len(samples) > 1 and all(samples):
            try: p = float(kruskal(*samples).pvalue)
            except ValueError: pass
        omnibus.append({"feature": feature, "reference_population": "core", "group_count": len(groups), "available_n": sum(map(len, samples)), "kruskal_p": p, "epsilon_squared": None})
        if p is not None:
            n, k = sum(map(len, samples)), len(samples); h = float(kruskal(*samples).statistic); omnibus[-1]["epsilon_squared"] = max(0., min(1., (h-k+1)/(n-k))) if n > k else 0.
    q = bh([r["kruskal_p"] for r in omnibus])
    for r, v in zip(omnibus, q): r["q_value"] = v
    pairwise = []
    significant = {r["feature"] for r in omnibus if r.get("q_value") is not None and r["q_value"] < .05}
    for fidx, feature in enumerate(features):
        if feature not in significant:
            continue
        current = []
        for a, b in combinations(groups, 2):
            left = [float(table[x][feature]) for x in groups[a] if x in table and pd.notna(table[x].get(feature))]
            right = [float(table[x][feature]) for x in groups[b] if x in table and pd.notna(table[x].get(feature))]
            p = float(mannwhitneyu(left, right).pvalue) if left and right else None
            delta, low, high = bootstrap_delta(left, right, bootstrap_iterations, BASE_SEED + fidx)
            current.append({"feature": feature, "reference_population": "core", "group_a": a, "group_b": b, "n_a": len(left), "n_b": len(right), "median_a": float(np.median(left)) if left else None, "median_b": float(np.median(right)) if right else None, "cliffs_delta": delta, "ci_low": low, "ci_high": high, "p_value": p})
        for r, v in zip(current, holm([x["p_value"] for x in current])): r["p_adjusted_holm"] = v
        pairwise.extend(current)
    rest = []
    for sidx, (sid, members) in enumerate(groups.items()):
        rest_ids = [x for x in candidate_ids if x not in members]
        current = []
        for fidx, feature in enumerate(features):
            left = [float(table[x][feature]) for x in members if x in table and pd.notna(table[x].get(feature))]
            right = [float(table[x][feature]) for x in rest_ids if x in table and pd.notna(table[x].get(feature))]
            p = float(mannwhitneyu(left, right).pvalue) if left and right else None
            delta, low, high = bootstrap_delta(left, right, bootstrap_iterations, BASE_SEED + sidx * 10000 + fidx)
            current.append({"subtype_id": sid, "feature": feature, "reference_population": reference_population, "target_n": len(left), "rest_n": len(right), "n_subtype": len(left), "n_rest": len(right), "median_subtype": float(np.median(left)) if left else None, "median_rest": float(np.median(right)) if right else None, "cliffs_delta": delta, "ci_low": low, "ci_high": high, "p_value": p})
        for r, v in zip(current, bh([x["p_value"] for x in current])): r["q_value"] = v
        rest.extend(current)
    return omnibus, pairwise, rest


def ct_analysis(context: Mapping[str, Any], output_root: Path, config_dir: Path, out: Path, bootstrap_iterations: int) -> pd.DataFrame:
    from tools.ct_radiomics import build_ct_discovery_feature_matrix
    ordered_states = [context["patient_states"][x] for x in context["candidate_patient_ids"]]
    payload = build_ct_discovery_feature_matrix(ordered_states, config_dir=str(config_dir), output_root=str(output_root))
    if payload["patient_ids"] != context["candidate_patient_ids"]:
        raise ValueError("CT production matrix order mismatch")
    matrix = np.asarray(payload["matrix"], dtype=float)
    features = [str(x) for x in payload["feature_names"]]
    table = {pid: dict(zip(features, row)) for pid, row in zip(payload["patient_ids"], matrix)}
    pd.DataFrame(matrix, index=payload["patient_ids"], columns=features).rename_axis("case_id").reset_index().to_csv(out / "ct_patient_features.csv", index=False)
    json_dump(out / "ct_preprocessing_audit.json", payload.get("audit", {}))
    groups = context["subtypes"]
    omnibus, pairwise, _ = continuous_tests(table, features, groups, context["core_patient_ids"], bootstrap_iterations, "ct", "core_rest")
    pd.DataFrame(omnibus).to_csv(out / "ct_omnibus.csv", index=False)
    pd.DataFrame(pairwise).to_csv(out / "ct_pairwise.csv", index=False)
    for reference, ids in (("core_rest", context["core_patient_ids"]), ("candidate_rest", context["candidate_patient_ids"])):
        _, _, current = continuous_tests(table, features, groups, ids, bootstrap_iterations, "ct", reference)
        frame = pd.DataFrame(current)
        frame.to_csv(out / f"ct_subtype_vs_{reference}.csv", index=False)
        profile = frame[frame.get("q_value", pd.Series(index=frame.index, dtype=float)).fillna(1.0) < .05].copy()
        if not profile.empty:
            profile = profile.sort_values(["q_value", "cliffs_delta"], key=lambda s: s.abs() if s.name == "cliffs_delta" else s, ascending=[True, False])
        profile.to_csv(out / f"ct_subtype_profile_{reference}.csv", index=False)
    return pd.read_csv(out / "ct_subtype_vs_core_rest.csv")


def rna_analysis(context: Mapping[str, Any], output_root: Path, config_dir: Path, out: Path, top_marker_genes: int = 10, bootstrap_iterations: int = 2000) -> pd.DataFrame:
    from gseapy import prerank, ssgsea
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    counts = pd.read_csv(output_root / "rna" / "case_raw_counts.csv").set_index("case_id")
    expr = pd.read_csv(output_root / "rna" / "case_pathway_features.csv").set_index("case_id")
    counts.index = counts.index.astype(str); expr.index = expr.index.astype(str)
    if set(context["candidate_patient_ids"]) - set(counts.index) or set(context["candidate_patient_ids"]) - set(expr.index):
        raise ValueError("RNA matrices lack candidate patients")
    cfg = __import__("yaml").safe_load((config_dir / "subtype_review.yaml").read_text())["rna"]
    collections = {"HALLMARK": cfg["hallmark_gene_sets_path"], "REACTOME": cfg["reactome_gene_sets_path"], "KEGG_MEDICUS": cfg["kegg_medicus_gene_sets_path"]}
    gene_sets = {}
    for collection, path in collections.items():
        sets = {}
        for line in Path(path).read_text().splitlines():
            fields = line.rstrip().split("\t")
            if len(fields) > 2: sets[fields[0]] = sorted(set(fields[2:]) & set(counts.columns))
        gene_sets[collection] = {k: v for k, v in sets.items() if len(v) >= int(cfg["min_pathway_overlap"])}
    de_by_ref, gsea_by_ref = {}, {}
    for reference, universe in (("core_rest", context["core_patient_ids"]), ("candidate_rest", context["candidate_patient_ids"])):
        all_de, all_gsea = [], []
        for sidx, sid in enumerate(context["subtype_order"]):
            target, rest = build_subtype_contrast(context, sid, reference)
            ids = target + rest
            metadata = pd.DataFrame({"condition": ["target"] * len(target) + ["rest"] * len(rest)}, index=ids)
            dds = DeseqDataSet(counts=counts.loc[ids].astype(np.int64), metadata=metadata, design="~condition", refit_cooks=True, n_cpus=int(cfg.get("n_cpus", 1)), quiet=True)
            dds.deseq2()
            ds = DeseqStats(dds, contrast=["condition", "target", "rest"], cooks_filter=False, independent_filter=False, n_cpus=int(cfg.get("n_cpus", 1)), quiet=True)
            ds.run_wald_test()
            stat = pd.Series(np.asarray(ds.statistics, dtype=float).reshape(-1), index=counts.columns.astype(str))
            result = getattr(ds, "results_df", pd.DataFrame(index=counts.columns))
            for gene, value in stat.items():
                row = result.loc[gene] if gene in result.index else pd.Series(dtype=float)
                all_de.append({"subtype_id": sid, "gene": gene, "reference_population": reference, "wald_stat": float(value) if np.isfinite(value) else None, "log2_fold_change": float(row.get("log2FoldChange")) if pd.notna(row.get("log2FoldChange")) else None, "p_value": float(row.get("pvalue")) if pd.notna(row.get("pvalue")) else None, "padj": float(row.get("padj")) if pd.notna(row.get("padj")) else None, "target_n": len(target), "rest_n": len(rest)})
            ranking = pd.DataFrame({"gene": stat.index, "stat": stat.values}).replace([np.inf, -np.inf], np.nan).dropna().sort_values(["stat", "gene"], ascending=[False, True])
            ranked_genes = set(ranking.gene.astype(str))
            for collection, sets in gene_sets.items():
                eligible = {name: sorted(set(genes) & ranked_genes) for name, genes in sets.items()}
                eligible = {name: genes for name, genes in eligible.items() if len(genes) >= int(cfg["min_pathway_overlap"])}
                if len(ranking) < 2 or not eligible: continue
                res = prerank(rnk=ranking, gene_sets=eligible, min_size=int(cfg["min_pathway_overlap"]), max_size=500, permutation_num=int(cfg.get("gsea_permutations", 1000)), seed=BASE_SEED+sidx, threads=int(cfg.get("gsea_threads", 1)), outdir=None, verbose=False).res2d
                for row in res.to_dict("records"):
                    all_gsea.append({"subtype_id": sid, "collection": collection, "pathway": str(row["Term"]), "reference_population": reference, "ES": float(row.get("ES")), "NES": float(row["NES"]), "nominal_p": float(row.get("NOM p-val")), "fdr_q": float(row.get("FDR q-val")), "fwer_p": float(row.get("FWER p-val")), "lead_genes": str(row.get("Lead_genes", "")), "target_n": len(target), "rest_n": len(rest)})
        de = pd.DataFrame(all_de); gsea = pd.DataFrame(all_gsea)
        de_by_ref[reference], gsea_by_ref[reference] = de, gsea
        de.to_csv(out / f"rna_deseq2_{reference}.csv", index=False)
        gsea.to_csv(out / f"rna_gsea_{reference}.csv", index=False)
        pathways = sorted(gsea.loc[gsea.fdr_q < .05, "pathway"].unique()) if not gsea.empty else []
        matrix = gsea[gsea.pathway.isin(pathways)].pivot_table(index="pathway", columns="subtype_id", values="NES", aggfunc="first").reindex(columns=context["subtype_order"]) if pathways else pd.DataFrame(columns=context["subtype_order"])
        matrix.to_csv(out / f"rna_pathway_nes_matrix_{reference}.csv")
    primary_de, primary_gsea = de_by_ref["core_rest"], gsea_by_ref["core_rest"]
    markers = []
    for sid in context["subtype_order"]:
        local = primary_de[(primary_de.subtype_id == sid) & primary_de.padj.notna() & (primary_de.padj < .05)]
        for direction, mask in (("up", local.log2_fold_change > 0), ("down", local.log2_fold_change < 0)):
            selected = local[mask].sort_values(["padj", "log2_fold_change"], ascending=[True, direction == "down"]).head(top_marker_genes)
            markers.extend({"subtype_id": sid, "gene": row.gene, "direction": direction, "padj": row.padj, "log2_fold_change": row.log2_fold_change} for row in selected.itertuples())
    marker_frame = pd.DataFrame(markers, columns=["subtype_id", "gene", "direction", "padj", "log2_fold_change"]); marker_frame.to_csv(out / "rna_marker_genes.csv", index=False)
    marker_genes = [gene for gene in dict.fromkeys(marker_frame.gene.tolist()) if gene in expr.columns]
    z = expr.loc[context["core_patient_ids"], marker_genes].astype(float)
    z = (z - z.mean()) / z.std(ddof=0).replace(0, np.nan)
    z.T.to_csv(out / "rna_marker_gene_matrix.csv")
    pd.DataFrame({"case_id": context["core_patient_ids"], "subtype_id": [next(s for s,m in context["subtypes"].items() if p in m) for p in context["core_patient_ids"]]}).to_csv(out / "rna_marker_patient_annotation.csv", index=False)
    hallmark_sets = gene_sets["HALLMARK"]
    hallmark_sets = {name: sorted(set(genes) & set(expr.columns)) for name, genes in hallmark_sets.items()}
    hallmark_sets = {name: genes for name, genes in hallmark_sets.items() if len(genes) >= int(cfg["min_pathway_overlap"])}
    try:
        ss = ssgsea(data=expr.loc[context["candidate_patient_ids"], :].T, gene_sets=hallmark_sets, sample_norm_method="rank", min_size=int(cfg["min_pathway_overlap"]), outdir=None, no_plot=True, threads=int(cfg.get("gsea_threads", 1)), verbose=False).res2d
        scores = ss.pivot(index="Name", columns="Term", values="ES").rename_axis("case_id").reindex(context["candidate_patient_ids"])
    except Exception as exc:
        raise RuntimeError("Hallmark ssGSEA failed; no alternate scoring method was substituted.") from exc
    scores.insert(0, "subtype_id", [next((s for s,m in context["subtypes"].items() if p in m), "") for p in scores.index])
    scores.insert(1, "is_core", scores.index.isin(context["core_patient_ids"]))
    scores.reset_index().to_csv(out / "rna_hallmark_ssgsea_scores.csv", index=False)
    score_table = scores.drop(columns=["subtype_id", "is_core"])
    hallmark_omnibus, _, _ = continuous_tests(score_table.to_dict("index"), list(score_table.columns), context["subtypes"], context["core_patient_ids"], 0, "rna_hallmark", "core_rest")
    pd.DataFrame(hallmark_omnibus).to_csv(out / "rna_hallmark_omnibus.csv", index=False)
    rest_frames = {}
    for reference, universe in (("core_rest", context["core_patient_ids"]), ("candidate_rest", context["candidate_patient_ids"])):
        _, _, rows = continuous_tests(score_table.to_dict("index"), list(score_table.columns), context["subtypes"], universe, bootstrap_iterations, "rna_hallmark", reference)
        frame = pd.DataFrame(rows); rest_frames[reference] = frame; frame.to_csv(out / f"rna_hallmark_vs_{reference}.csv", index=False)
    consistency = []
    for row in rest_frames["core_rest"].itertuples():
        values = score_table.loc[context["subtypes"][row.subtype_id], row.feature].astype(float)
        rest_values = score_table.loc[build_subtype_contrast(context, row.subtype_id, "core_rest")[1], row.feature].astype(float)
        direction = "up" if row.cliffs_delta > 0 else "down" if row.cliffs_delta < 0 else "neutral"
        fraction = float((values > rest_values.median()).mean()) if direction == "up" else float((values < rest_values.median()).mean()) if direction == "down" else None
        consistency.append({"subtype_id": row.subtype_id, "pathway": row.feature, "direction": direction, "cliffs_delta": row.cliffs_delta, "q_value": row.q_value, "subtype_median": float(values.median()), "rest_median": float(rest_values.median()), "consistent_patient_n": int(round(fraction * len(values))) if fraction is not None else 0, "subtype_n": len(values), "consistency_fraction": fraction})
    consistency_frame = pd.DataFrame(consistency); consistency_frame.to_csv(out / "rna_hallmark_consistency.csv", index=False)
    concordance = primary_gsea[primary_gsea.collection == "HALLMARK"].merge(consistency_frame, left_on=["subtype_id", "pathway"], right_on=["subtype_id", "pathway"], how="outer")
    if not concordance.empty:
        concordance["direction_concordant"] = ((concordance.NES > 0) & (concordance.cliffs_delta > 0)) | ((concordance.NES < 0) & (concordance.cliffs_delta < 0))
        concordance["strong_concordant_signal"] = (concordance.fdr_q < .05) & (concordance.q_value < .05) & concordance.direction_concordant
    concordance.to_csv(out / "rna_hallmark_concordance.csv", index=False)
    immune = {"HALLMARK_ALLOGRAFT_REJECTION", "HALLMARK_COMPLEMENT", "HALLMARK_IL2_STAT5_SIGNALING", "HALLMARK_IL6_JAK_STAT3_SIGNALING", "HALLMARK_INFLAMMATORY_RESPONSE", "HALLMARK_INTERFERON_ALPHA_RESPONSE", "HALLMARK_INTERFERON_GAMMA_RESPONSE", "HALLMARK_TNFA_SIGNALING_VIA_NFKB"}
    consistency_frame[consistency_frame.pathway.isin(immune)].to_csv(out / "rna_immune_transcriptional_programs.csv", index=False)
    return primary_gsea

def wxs_analysis(context: Mapping[str, Any], output_root: Path, config_dir: Path, out: Path, permutations: int) -> pd.DataFrame:
    from tools import post_discovery_characterization as stats
    from agents.subtype_review.tools import odds_ratio_with_ci
    discovery = pd.read_csv(output_root / "wxs" / "wxs_discovery_features.csv").set_index("case_id").astype(float)
    interpretation = pd.read_csv(output_root / "wxs" / "wxs_interpretation_features.csv").set_index("case_id").astype(float)
    candidate = context["candidate_patient_ids"]
    if set(context["core_patient_ids"]) - set(discovery.index) or set(candidate) - set(interpretation.index): raise ValueError("WXS matrices lack required patients")
    groups = context["subtypes"]
    core = discovery.loc[context["core_patient_ids"]].astype(int).to_dict("index")
    features = list(discovery.columns)
    omnibus = stats.binary_permutation_omnibus(core, features, groups, permutations=permutations, seed=BASE_SEED)
    pairwise = stats.binary_posthoc(core, features, groups)
    pd.DataFrame(omnibus).to_csv(out / "wxs_core_omnibus.csv", index=False)
    pd.DataFrame(pairwise).to_csv(out / "wxs_core_pairwise.csv", index=False)
    driver = set()
    try:
        import yaml
        cfg = yaml.safe_load((config_dir / "wxs.yaml").read_text())
        driver = {str(x).upper() for x in cfg.get("biological_support", {}).get("driver_genes", [])}
    except FileNotFoundError: pass
    all_results = {}
    for reference, universe in (("core_rest", context["core_patient_ids"]), ("candidate_rest", candidate)):
        rows = []
        for sid in groups:
            target, rest = build_subtype_contrast(context, sid, reference)
            for feature in interpretation.columns:
                a = int(interpretation.loc[target, feature].sum()); b = len(target)-a; c = int(interpretation.loc[rest, feature].sum()); d = len(rest)-c
                odds, p = fisher_exact([[a,b],[c,d]])
                ci = odds_ratio_with_ci(np.asarray([[a, b], [c, d]], dtype=float)).get("odds_ratio_ci95")
                target_frequency, rest_frequency = a / len(target), c / len(rest)
                direction = "enriched" if target_frequency > rest_frequency else "depleted" if target_frequency < rest_frequency else "neutral"
                rows.append({"subtype_id": sid, "gene": feature, "reference_population": reference, "direction": direction, "subtype_mutated_n": a, "subtype_n": len(target), "subtype_frequency": target_frequency, "rest_mutated_n": c, "rest_n": len(rest), "rest_frequency": rest_frequency, "odds_ratio": float(odds), "ci_low": ci[0] if ci else None, "ci_high": ci[1] if ci else None, "p_value": float(p), "driver_panel_member": str(feature).upper() in driver})
        result = pd.DataFrame(rows); result["q_global"] = np.nan; result["q_driver"] = np.nan
        for sid, idx in result.groupby("subtype_id").groups.items():
            result.loc[idx, "q_global"] = bh(result.loc[idx, "p_value"].tolist())
            d_idx = [i for i in idx if bool(result.loc[i, "driver_panel_member"])]
            if d_idx: result.loc[d_idx, "q_driver"] = bh(result.loc[d_idx, "p_value"].tolist())
        result.to_csv(out / f"wxs_subtype_vs_{reference}.csv", index=False)
        result[(result.driver_panel_member) & (result.q_driver < .05)].to_csv(out / f"wxs_driver_profile_{reference}.csv", index=False)
        all_results[reference] = result
    oncoplot = interpretation.loc[context["core_patient_ids"]].copy()
    oncoplot.insert(0, "subtype_id", [next(s for s, m in groups.items() if p in m) for p in oncoplot.index])
    configured = sorted(driver)
    for gene in configured:
        if gene not in oncoplot.columns: oncoplot[gene] = 0
    oncoplot.reset_index(names="case_id")[(["case_id", "subtype_id"] + configured)].to_csv(out / "wxs_oncoplot_matrix.csv", index=False)
    summary_rows = []
    for sid, members in groups.items():
        for gene in configured:
            values = oncoplot.loc[members, gene] if gene in oncoplot else pd.Series(0, index=members)
            summary_rows.append({"subtype_id": sid, "gene": gene, "mutated_n": int(values.sum()), "total_n": len(members), "frequency": float(values.mean())})
    pd.DataFrame(summary_rows).to_csv(out / "wxs_driver_subtype_summary.csv", index=False)
    return all_results["core_rest"]


def known_and_confounders(context: Mapping[str, Any], output_root: Path, config_dir: Path, out: Path) -> None:
    from agents.subtype_review.tools import known_label_echo, confounder_association, confounder_representation_effect, categorical_test
    states = context["patient_states"]
    clusters = [{"set_id": sid, "member_ids": members} for sid, members in context["subtypes"].items()]
    known = known_label_echo(states, str(output_root), clusters, str(config_dir), "partition", [])
    json_dump(out / "known_label_echo.json", known)
    tax_rows = []
    for reference, payload in known.get("metrics", {}).get("partition", {}).get("molecular_taxonomy", {}).items():
        table = pd.DataFrame(payload.get("contingency", {})).fillna(0)
        for subtype_id, row in table.iterrows():
            total = float(row.sum())
            for label, n in row.items():
                tax_rows.append({"subtype_id": str(subtype_id), "reference": reference, "reference_label": str(label), "n": int(n), "fraction": float(n / total) if total else None})
    taxonomy = pd.DataFrame(tax_rows)
    taxonomy.to_csv(out / "known_taxonomy_profile.csv", index=False)
    association_rows, enrichment_rows = [], []
    import yaml
    known_cfg = yaml.safe_load((config_dir / "subtype_review.yaml").read_text()).get("known_label_echo", {})
    from agents.subtype_review.tools import odds_ratio_with_ci
    for name, key in (("clearcode34", "clearcode34_path"), ("tcga_m1_m4", "mrna_m1_m4_path")):
        path = known_cfg.get(key)
        if not path or not Path(path).is_file(): continue
        labels = pd.read_csv(path)
        labels = labels[(labels.reference_status == "matched") & labels.reference_subtype.notna()]
        labels = labels[labels.case_id.astype(str).isin(context["core_patient_ids"])].copy()
        labels["subtype_id"] = labels.case_id.astype(str).map({p: sid for sid, m in context["subtypes"].items() for p in m})
        table = pd.crosstab(labels.subtype_id, labels.reference_subtype)
        if table.shape[0] > 1 and table.shape[1] > 1:
            test_result = categorical_test(labels.subtype_id.tolist(), labels.reference_subtype.astype(str).tolist(), 9999, BASE_SEED)
            association_rows.append({"taxonomy": name, "available_n": len(labels), "test": test_result.get("test"), "p_value": test_result.get("p_value"), "cramers_v": test_result.get("cramers_v")})
        for sid in context["subtype_order"]:
            target, rest = build_subtype_contrast(context, sid, "core_rest")
            local = labels.set_index("case_id")
            for label in sorted(labels.reference_subtype.astype(str).unique()):
                a = int((local.loc[local.index.intersection(target), "reference_subtype"] == label).sum()); b = len(local.index.intersection(target)) - a
                c = int((local.loc[local.index.intersection(rest), "reference_subtype"] == label).sum()); d = len(local.index.intersection(rest)) - c
                odds, p = fisher_exact([[a, b], [c, d]])
                ci = odds_ratio_with_ci(np.asarray([[a, b], [c, d]], dtype=float)).get("odds_ratio_ci95")
                enrichment_rows.append({"subtype_id": sid, "taxonomy": name, "reference_label": label, "subtype_positive_n": a, "subtype_available_n": a + b, "rest_positive_n": c, "rest_available_n": c + d, "odds_ratio": float(odds), "ci_low": ci[0] if ci else None, "ci_high": ci[1] if ci else None, "p_value": float(p)})
    association = pd.DataFrame(association_rows); association["q_value"] = bh(association.p_value.tolist()) if not association.empty else []; association.to_csv(out / "known_taxonomy_association.csv", index=False)
    enrichment = pd.DataFrame(enrichment_rows)
    if not enrichment.empty:
        enrichment["q_value"] = np.nan
        for taxonomy_name, idx in enrichment.groupby("taxonomy").groups.items(): enrichment.loc[idx, "q_value"] = bh(enrichment.loc[idx, "p_value"].tolist())
    enrichment.to_csv(out / "known_taxonomy_enrichment.csv", index=False)
    conf = confounder_association(states, str(output_root), str(config_dir), clusters, "partition", [])
    json_dump(out / "technical_confounder_association.json", conf)
    effect = confounder_representation_effect(states, str(output_root), str(config_dir), clusters, "partition", [])
    json_dump(out / "technical_confounder_representation_effect.json", effect)
    return taxonomy


def clinical_analysis(context: Mapping[str, Any], clinical_path: Path, out: Path, permutations: int) -> pd.DataFrame:
    from agents.subtype_review.tools import categorical_test, clinical_labels, cramers_v
    raw = json.loads(clinical_path.read_text(encoding="utf-8")) if clinical_path.is_file() else []
    by_case = {str(row.get("Case_ID")): row for row in raw}
    normalized = clinical_labels(context["patient_states"])
    rows = []
    for sid, members in context["subtypes"].items():
        for case_id in members:
            clinical = (by_case.get(case_id) or {}).get("Clinical", {})
            demographic = clinical.get("demographic", {}) or {}
            diagnoses = clinical.get("diagnoses") or [{}]
            primary = [r for r in diagnoses if str(r.get("diagnosis_is_primary_disease", "")).lower() == "true"]
            diagnosis = (primary or diagnoses)[0]
            age = demographic.get("age_at_index")
            if age is None:
                try: age = float(diagnosis.get("age_at_diagnosis")) / 365.25
                except (TypeError, ValueError): age = np.nan
            try: age = float(age)
            except (TypeError, ValueError): age = np.nan
            sex = str(demographic.get("gender", "unknown")).strip().lower()
            sex = sex if sex in {"male", "female"} else "unknown"
            labels = normalized.get(case_id, {})
            rows.append({"case_id": case_id, "subtype_id": sid, "age_at_diagnosis_years": age, "sex": sex, "stage": labels.get("stage") or "unknown", "grade": labels.get("grade") or "unknown", "t_stage": labels.get("t_stage") or "unknown", "m_stage": labels.get("m_stage") or "unknown"})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "clinical_patient_table.csv", index=False)
    global_rows = []
    for variable in ("age_at_diagnosis_years", "sex", "stage", "grade", "t_stage", "m_stage"):
        if variable == "age_at_diagnosis_years":
            valid = frame[variable].notna()
            samples = [frame.loc[(frame.subtype_id == sid) & valid, variable].tolist() for sid in context["subtype_order"]]
            p = float(kruskal(*samples).pvalue) if len(samples) > 1 and all(samples) else None
            n, k = sum(map(len, samples)), len(samples)
            h = float(kruskal(*samples).statistic) if p is not None else None
            effect = max(0.0, min(1.0, (h - k + 1) / (n - k))) if h is not None and n > k else None
            test = "kruskal_wallis"; available_n = int(valid.sum())
        else:
            valid = frame[variable].notna() & (frame[variable] != "unknown")
            groups = frame.loc[valid, "subtype_id"].tolist(); values = frame.loc[valid, variable].tolist()
            result = categorical_test(groups, values, permutations, BASE_SEED, force_fisher=False) if len(set(groups)) > 1 and len(set(values)) > 1 else {"test": "not_estimable", "p_value": None, "cramers_v": None}
            p, effect, test = result.get("p_value"), result.get("cramers_v"), result.get("test")
            available_n = int(valid.sum())
        global_rows.append({"variable": variable, "test": test, "available_n": available_n, "missing_n": len(frame) - available_n, "p_value": p, "effect_size": effect})
    global_frame = pd.DataFrame(global_rows); global_frame["q_value"] = bh(global_frame.p_value.tolist()); global_frame.to_csv(out / "clinical_global_association.csv", index=False)
    posthoc = []
    for variable in global_frame.loc[global_frame.q_value < .05, "variable"]:
        for sid in context["subtype_order"]:
            target, rest = build_subtype_contrast(context, sid, "core_rest")
            lookup = frame.set_index("case_id")
            if variable == "age_at_diagnosis_years":
                left = pd.to_numeric(lookup.loc[target, variable], errors="coerce").dropna(); right = pd.to_numeric(lookup.loc[rest, variable], errors="coerce").dropna()
                p = float(mannwhitneyu(left, right).pvalue) if len(left) and len(right) else None
                posthoc.append({"variable": variable, "subtype_id": sid, "test": "mann_whitney", "target_n": len(left), "rest_n": len(right), "cliffs_delta": cliffs_delta(left.tolist(), right.tolist()), "p_value": p})
            else:
                values = pd.concat([lookup.loc[target, variable], lookup.loc[rest, variable]]).astype(str)
                groups = [sid] * len(target) + ["rest"] * len(rest)
                valid = values != "unknown"
                result = categorical_test(list(np.asarray(groups)[valid]), values[valid].tolist(), permutations, BASE_SEED, force_fisher=False) if len(set(values[valid])) > 1 else {"test": "not_estimable", "p_value": None, "cramers_v": None}
                posthoc.append({"variable": variable, "subtype_id": sid, "test": result.get("test"), "target_n": int(valid.iloc[:len(target)].sum()), "rest_n": int(valid.iloc[len(target):].sum()), "cramers_v": result.get("cramers_v"), "p_value": result.get("p_value")})
        idx = [i for i, row in enumerate(posthoc) if row["variable"] == variable]
        for i, q in zip(idx, holm([posthoc[j]["p_value"] for j in idx])): posthoc[i]["q_value"] = q
    pd.DataFrame(posthoc).to_csv(out / "clinical_subtype_posthoc.csv", index=False)
    return frame

def representatives(context: Mapping[str, Any], output_root: Path, out: Path) -> pd.DataFrame:
    pairs = pd.read_csv(context["patient_recurrence_pair_similarity"], dtype={"patient_id_left": str, "patient_id_right": str})
    ids, distances = __import__("agents.subtype_review.tools", fromlist=["modality_distance_matrices"]).modality_distance_matrices(str(output_root))
    position = {x: i for i, x in enumerate(ids)}; rows = []
    for sid, members in context["subtypes"].items():
        centrality = {}
        for patient in members:
            vals = pairs[((pairs.patient_id_left == patient) & pairs.patient_id_right.isin(members)) | ((pairs.patient_id_right == patient) & pairs.patient_id_left.isin(members))].similarity.tolist()
            centrality[patient] = float(np.mean(vals)) if vals else 1.0
        best = max(centrality.values()); candidates = sorted([x for x, v in centrality.items() if v == best])
        if len(candidates) > 1:
            scores = {x: float(np.mean([distances[m][position[x], position[y]] for m in MODALITIES for y in members if y != x])) for x in candidates}
            best_score = min(scores.values()); candidates = sorted([x for x, v in scores.items() if v == best_score])
        patient = candidates[0]
        scores = {m: float(np.mean([distances[m][position[patient], position[y]] for y in members if y != patient])) if len(members) > 1 else 0.0 for m in MODALITIES}
        rows.append({"subtype_id": sid, "case_id": patient, "member_count": len(members), "recurrence_centrality": centrality[patient], **{f"{m}_mean_distance": scores[m] for m in MODALITIES}, "multimodal_mean_distance": float(np.mean(list(scores.values()))), "selection_rule": "max recurrence centrality; native-distance mean; lexical case_id"})
    frame = pd.DataFrame(rows); frame.to_csv(out / "representative_patients.csv", index=False); return frame


def survival_analysis(context: Mapping[str, Any], clinical_path: Path, out: Path, survival_file: Path | None = None) -> tuple[str, pd.DataFrame]:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test
    allowed_endpoints = {"OS", "PFI", "DFI", "DFS", "DSS"}
    subtype_by_patient = {patient: sid for sid, members in context["subtypes"].items() for patient in members}
    candidate = context["candidate_patient_ids"]
    if survival_file is not None:
        frame = pd.read_csv(survival_file, dtype={"case_id": str})
        required = {"case_id", "endpoint", "time_days", "event"}
        if not required.issubset(frame.columns): raise ValueError(f"Survival file must contain {sorted(required)}")
        frame["case_id"] = frame["case_id"].astype(str); frame["endpoint"] = frame["endpoint"].astype(str).str.upper()
        unknown = sorted(set(frame.endpoint) - allowed_endpoints)
        if unknown: raise ValueError(f"Unsupported survival endpoints: {unknown}")
        frame = frame[frame.case_id.isin(candidate)].copy()
        frame["time_days"] = pd.to_numeric(frame.time_days, errors="coerce"); frame["event"] = pd.to_numeric(frame.event, errors="coerce")
        frame = frame[frame.time_days.notna() & frame.event.isin([0, 1])]
        frame["subtype_id"] = frame.case_id.map(subtype_by_patient); frame["is_core"] = frame.case_id.isin(context["core_patient_ids"])
    else:
        if not clinical_path.is_file(): return "skipped_missing_source", pd.DataFrame()
        raw = json.loads(clinical_path.read_text(encoding="utf-8")); by_case = {str(row.get("Case_ID")): row for row in raw}; rows = []
        for case_id in candidate:
            clinical = (by_case.get(case_id) or {}).get("Clinical", {}); demographic = clinical.get("demographic", {}) or {}; diagnoses = clinical.get("diagnoses") or [{}]
            primary = [row for row in diagnoses if str(row.get("diagnosis_is_primary_disease", "")).lower() == "true"]; diagnosis = (primary or diagnoses)[0]
            event = int(str(demographic.get("vital_status", "")).strip().lower() == "dead"); followups = []
            for row in diagnoses:
                try:
                    if row.get("days_to_last_follow_up") not in (None, ""): followups.append(float(row["days_to_last_follow_up"]))
                except (TypeError, ValueError): pass
            time_value = demographic.get("days_to_death") if event else (max(followups) if followups else diagnosis.get("days_to_last_follow_up"))
            try: time_days = float(time_value)
            except (TypeError, ValueError): continue
            if np.isfinite(time_days) and time_days >= 0: rows.append({"case_id": case_id, "subtype_id": subtype_by_patient.get(case_id), "is_core": case_id in context["core_patient_ids"], "endpoint": "OS", "time_days": time_days, "event": event})
        frame = pd.DataFrame(rows)
    if frame.empty: return "skipped_no_valid_endpoint", frame
    if "OS" in set(frame.endpoint):
        frame["os_time_days"] = np.where(frame.endpoint == "OS", frame.time_days, np.nan)
        frame["os_event"] = np.where(frame.endpoint == "OS", frame.event, np.nan)
    frame.to_csv(out / "survival_patient_table.csv", index=False)
    summary_rows, km_rows, global_rows = [], [], []; completed_endpoints = []
    for endpoint, endpoint_frame in frame.groupby("endpoint", sort=True):
        core_endpoint = endpoint_frame[endpoint_frame.is_core & endpoint_frame.subtype_id.notna()].copy()
        if core_endpoint.empty or core_endpoint.subtype_id.nunique() < 2 or int(core_endpoint.event.sum()) < 1: continue
        completed_endpoints.append(endpoint)
        for sid in context["subtype_order"]:
            group = core_endpoint[core_endpoint.subtype_id == sid]
            available_n = len(group); missing_n = len(context["subtypes"][sid]) - available_n
            if not available_n: continue
            kmf = KaplanMeierFitter().fit(group.time_days, group.event, label=sid); median = float(kmf.median_survival_time_) if np.isfinite(kmf.median_survival_time_) else None
            summary_rows.append({"endpoint": endpoint, "reference_population": "core", "subtype_id": sid, "n": available_n, "available_n": available_n, "missing_n": missing_n, "event_n": int(group.event.sum()), "censored_n": int(available_n - group.event.sum()), "median_survival_days": median, "low_event_count": int(group.event.sum()) < 5})
            for time, value in kmf.survival_function_[sid].items(): km_rows.append({"endpoint": endpoint, "reference_population": "core", "subtype_id": sid, "timeline_days": float(time), "survival_probability": float(value)})
        result = multivariate_logrank_test(core_endpoint.time_days, core_endpoint.subtype_id, core_endpoint.event)
        global_rows.append({"endpoint": endpoint, "reference_population": "core", "n": len(core_endpoint), "event_n": int(core_endpoint.event.sum()), "low_event_count": int(core_endpoint.event.sum()) < 5, "test": "multivariate_logrank", "test_statistic": float(result.test_statistic), "p_value": float(result.p_value)})
    pd.DataFrame(summary_rows).to_csv(out / "survival_summary.csv", index=False); pd.DataFrame(global_rows).to_csv(out / "survival_global_tests.csv", index=False); pd.DataFrame(km_rows).to_csv(out / "survival_km_points.csv", index=False)
    if not completed_endpoints: return "skipped_insufficient_events", pd.DataFrame(summary_rows)
    return ("complete" if len(completed_endpoints) == frame.endpoint.nunique() else "partial"), pd.DataFrame(summary_rows)

def identity_cards(context: Mapping[str, Any], recurrence: pd.DataFrame, wsi: pd.DataFrame, ct: pd.DataFrame, rna: pd.DataFrame, wxs: pd.DataFrame, taxonomy: pd.DataFrame, reps: pd.DataFrame, survival: pd.DataFrame, out: Path) -> None:
    def top_cont(frame, sid):
        if frame.empty: return [], []
        sub = frame[(frame.subtype_id == sid) & (frame.q_value < .05)].copy() if "q_value" in frame else pd.DataFrame()
        def rows(part, direction):
            part = part.sort_values(["q_value", "cliffs_delta"], key=lambda s: s.abs() if s.name == "cliffs_delta" else s, ascending=[True, False]).head(3)
            return [{"feature": r.feature, "direction": direction, "median_subtype": r.median_subtype, "median_rest": r.median_rest, "cliffs_delta": r.cliffs_delta, "ci95": [r.ci_low, r.ci_high], "q_value": r.q_value} for r in part.itertuples()]
        up = rows(sub[sub.cliffs_delta > 0], "enriched")
        down = rows(sub[sub.cliffs_delta < 0], "depleted")
        return up, down
    cards = []
    clinical = pd.read_csv(out / "clinical_patient_table.csv") if (out / "clinical_patient_table.csv").is_file() else pd.DataFrame()
    for sid in context["subtype_order"]:
        rec = recurrence[recurrence.subtype_id == sid].iloc[0].to_dict()
        cu, cd = top_cont(ct, sid)
        concordance = pd.read_csv(out / "rna_hallmark_concordance.csv") if (out / "rna_hallmark_concordance.csv").is_file() else pd.DataFrame()
        hall = concordance[(concordance.subtype_id == sid) & concordance.strong_concordant_signal] if not concordance.empty and {"subtype_id", "strong_concordant_signal", "NES", "fdr_q", "cliffs_delta", "q_value", "consistency_fraction"}.issubset(concordance.columns) else pd.DataFrame()
        hallmark_up = [{"pathway": r.pathway, "gsea_nes": r.NES, "gsea_fdr": r.fdr_q, "ssgsea_cliffs_delta": r.cliffs_delta, "ssgsea_q": r.q_value, "consistency_fraction": r.consistency_fraction} for r in hall[hall.NES > 0].itertuples()] if not hall.empty else []
        hallmark_down = [{"pathway": r.pathway, "gsea_nes": r.NES, "gsea_fdr": r.fdr_q, "ssgsea_cliffs_delta": r.cliffs_delta, "ssgsea_q": r.q_value, "consistency_fraction": r.consistency_fraction} for r in hall[hall.NES < 0].itertuples()] if not hall.empty else []
        markers = pd.read_csv(out / "rna_marker_genes.csv") if (out / "rna_marker_genes.csv").is_file() else pd.DataFrame()
        marker_up = markers[(markers.subtype_id == sid) & (markers.direction == "up")].to_dict("records") if not markers.empty else []
        marker_down = markers[(markers.subtype_id == sid) & (markers.direction == "down")].to_dict("records") if not markers.empty else []
        x = wxs[(wxs.subtype_id == sid) & (wxs.driver_panel_member) & (wxs.q_driver < .05)] if not wxs.empty else pd.DataFrame()
        enriched = x[x.direction == "enriched"].to_dict("records") if not x.empty else []
        depleted = x[x.direction == "depleted"].to_dict("records") if not x.empty else []
        profiles = {}
        for reference in ("clearcode34", "tcga_m1_m4"):
            selected = taxonomy[(taxonomy.subtype_id == sid) & (taxonomy.reference == reference)] if not taxonomy.empty else pd.DataFrame()
            profiles[reference] = {str(row.reference_label): float(row.fraction) for row in selected.itertuples()}
        survival_rows = survival[survival.subtype_id == sid] if not survival.empty else pd.DataFrame()
        survival_profile = {str(row.endpoint): {"n": int(row.n), "event_n": int(row.event_n), "median_survival_days": row.median_survival_days} for row in survival_rows.itertuples()}
        local_clinical = clinical[clinical.subtype_id == sid] if not clinical.empty else pd.DataFrame()
        clinical_profile = {"age_median_years": float(local_clinical.age_at_diagnosis_years.median()) if not local_clinical.empty else None, "sex_counts": local_clinical.sex.value_counts().to_dict() if not local_clinical.empty else {}, "stage_counts": local_clinical.stage.value_counts().to_dict() if not local_clinical.empty else {}, "grade_counts": local_clinical.grade.value_counts().to_dict() if not local_clinical.empty else {}, "t_stage_counts": local_clinical.t_stage.value_counts().to_dict() if not local_clinical.empty else {}, "m_stage_counts": local_clinical.m_stage.value_counts().to_dict() if not local_clinical.empty else {}}
        cards.append({"subtype_id": sid, "member_count": int(rec["member_count"]), "stability": {"common_accept_set_count": int(rec["common_accept_set_count"]), "supporting_run_count": int(rec["supporting_run_count"]), "supporting_k_count": int(rec["supporting_k_count"]), "supporting_ks": rec["supporting_ks"], "mean_pair_recurrence_similarity": rec["mean_pair_recurrence_similarity"], "min_pair_recurrence_similarity": rec["min_pair_recurrence_similarity"]}, "clinical": clinical_profile, "survival": survival_profile, "rna": {"hallmark_concordant_up": hallmark_up, "hallmark_concordant_down": hallmark_down, "marker_genes_up": marker_up, "marker_genes_down": marker_down}, "wxs": {"enriched_drivers": enriched, "depleted_drivers": depleted}, "known_taxonomy": {"clearcode34": profiles["clearcode34"], "tcga_mrna_profile": profiles["tcga_m1_m4"]}, "ct_quantitative": {"enriched": cu, "depleted": cd}, "wsi_semantic_phenotype": None, "wsi_semantic_status": "not_available", "warnings": [], "representative_patient": reps.loc[reps.subtype_id == sid, "case_id"].iloc[0]})
    json_dump(out / "subtype_identity_card.json", cards)
    pd.DataFrame(cards).to_csv(out / "subtype_identity_card.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Characterize frozen patient-recurrence subtypes from production artifacts.")
    parser.add_argument("--output-root", default="output_kirc")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--analysis-root", default=None)
    parser.add_argument("--clinical-file", default="data/tcga_kirc_data.json")
    parser.add_argument("--survival-file", default=None, help="Optional CSV with case_id, endpoint, time_days, event.")
    parser.add_argument("--top-marker-genes", type=int, default=10)
    parser.add_argument("--permutations", type=int, default=9999)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output_root, config_dir = Path(args.output_root), Path(args.config_dir)
    out = Path(args.analysis_root) if args.analysis_root else output_root / "final_subtype_analysis"
    if out.exists():
        if not args.force: raise FileExistsError(f"Analysis output exists; use --force: {out}")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    context = load_final_subtype_context(output_root)
    import yaml
    config = yaml.safe_load((config_dir / "subtype_review.yaml").read_text(encoding="utf-8"))
    print("[final_subtype_analysis] recurrence", flush=True)
    recurrence = recurrence_analysis(context, out)
    print("[final_subtype_analysis] representation", flush=True)
    representation_analysis(context, output_root, out, config)
    print("[final_subtype_analysis] WSI", flush=True)
    wsi_rest, wsi_patient = wsi_analysis(context, output_root, out, args.bootstrap_iterations)
    print("[final_subtype_analysis] CT", flush=True)
    ct_rest = ct_analysis(context, output_root, config_dir, out, args.bootstrap_iterations)
    print("[final_subtype_analysis] RNA DESeq2/GSEA", flush=True)
    rna = rna_analysis(context, output_root, config_dir, out, args.top_marker_genes, args.bootstrap_iterations)
    print("[final_subtype_analysis] WXS", flush=True)
    wxs = wxs_analysis(context, output_root, config_dir, out, args.permutations)
    print("[final_subtype_analysis] taxonomy/confounders", flush=True)
    taxonomy = known_and_confounders(context, output_root, config_dir, out)
    clinical_path = Path(args.clinical_file)
    print("[final_subtype_analysis] clinical", flush=True)
    clinical = clinical_analysis(context, clinical_path, out, args.permutations)
    print("[final_subtype_analysis] survival", flush=True)
    survival_status, survival_summary = survival_analysis(
        context,
        clinical_path,
        out,
        Path(args.survival_file) if args.survival_file else None,
    )
    print("[final_subtype_analysis] representatives and identity cards", flush=True)
    reps = representatives(context, output_root, out)
    identity_cards(context, recurrence, wsi_rest, ct_rest, rna, wxs, taxonomy, reps, survival_summary, out)
    subtype_cfg = config_dir / "subtype_review.yaml"
    wxs_cfg = config_dir / "wxs.yaml"
    candidate_cfg = config_dir / "candidate_proposer.yaml"
    source_files = list(context["source_paths"].values()) + [output_root / "rna/case_raw_counts.csv", output_root / "wxs/wxs_discovery_features.csv", output_root / "wxs/wxs_interpretation_features.csv", subtype_cfg, wxs_cfg, candidate_cfg, clinical_path, Path(__file__)]
    if args.survival_file:
        source_files.append(Path(args.survival_file))
    try:
        subtype_yaml = yaml.safe_load(subtype_cfg.read_text(encoding="utf-8"))
        source_files.extend(Path(subtype_yaml["rna"][key]) for key in ("hallmark_gene_sets_path", "reactome_gene_sets_path", "kegg_medicus_gene_sets_path"))
    except (KeyError, TypeError):
        pass
    wxs_yaml = yaml.safe_load(wxs_cfg.read_text(encoding="utf-8"))
    configured_drivers = sorted(str(x).upper() for x in wxs_yaml.get("biological_support", {}).get("driver_genes", []))
    wxs_columns = set(pd.read_csv(output_root / "wxs/wxs_interpretation_features.csv", nrows=0).columns.astype(str).str.upper())
    survival_source = Path(args.survival_file) if args.survival_file else clinical_path
    manifest = {"analysis_type": "final_stable_subtype_characterization", "analysis_role": "post_discovery_in_sample_characterization", "final_subtype_source": str(context["source_paths"]["subtypes"].resolve()), "candidate_patient_order": str(context["source_paths"]["order"].resolve()), "core_patient_count": len(context["core_patient_ids"]), "noncore_patient_count": len(context["noncore_patient_ids"]), "subtype_count": len(context["subtypes"]), "subtype_sizes": {k: len(v) for k, v in context["subtypes"].items()}, "active_modalities": list(MODALITIES), "primary_comparison": "subtype_vs_other_stable_core_patients", "secondary_comparison": "subtype_vs_all_other_candidate_patients", "survival_source": str(survival_source.resolve()), "survival_endpoint_policy": "User-supplied endpoints preserved by name" if args.survival_file else "OS from vital_status, days_to_death, and follow-up; no DFS/PFS synthesis", "available_driver_genes": sorted(set(configured_drivers) & wxs_columns), "missing_driver_genes": sorted(set(configured_drivers) - wxs_columns), "scripts_dependency": False, "legacy_output_dependency": False, "input_sha256": {str(p): sha256(p) for p in source_files if p.is_file()}}
    json_dump(out / "source_manifest.json", manifest)
    ct_confounding = False; significant_technical_factors = []
    confounder_path = out / "technical_confounder_association.json"
    if confounder_path.is_file():
        confounder = json.loads(confounder_path.read_text(encoding="utf-8"))
        rows = confounder.get("metrics", {}).get("partition", {}).get("partition", [])
        significant_technical_factors = [str(row.get("factor")) for row in rows if row.get("q_value") is not None and float(row["q_value"]) < .05]
        ct_confounding = bool(significant_technical_factors)
    summary = {"status": "complete", "subtype_count": len(context["subtypes"]), "core_patient_count": len(context["core_patient_ids"]), "candidate_patient_count": len(context["candidate_patient_ids"]), "noncore_patient_count": len(context["noncore_patient_ids"]), "primary_comparison": "subtype_vs_other_stable_core_patients", "secondary_comparison": "subtype_vs_all_other_candidate_patients", "subtype_sizes": {k: len(v) for k, v in context["subtypes"].items()}, "analyses": {"stability": "complete", "representation": "complete", "clinical": "complete", "survival": survival_status, "rna_deseq2_gsea": "complete", "rna_patient_level_ssgsea": "complete", "rna_marker_genes": "complete", "wxs": "complete", "known_taxonomy": "complete", "technical_confounders": "complete"}, "warnings": {"ct_technical_confounding_detected": ct_confounding, "significant_technical_factors": significant_technical_factors, "wsi_semantic_characterization_available": False, "external_validation_available": False}}
    json_dump(out / "analysis_summary.json", summary)
    print("[final_subtype_analysis]")
    for key, value in (("candidate cohort", summary["candidate_patient_count"]), ("core patients", summary["core_patient_count"]), ("noncore patients", summary["noncore_patient_count"]), ("final subtypes", summary["subtype_count"])): print(f"{key}: {value}")
    print("sizes: " + ", ".join(f"{k}={v}" for k, v in summary["subtype_sizes"].items()))
    print(f"output: {out}")


if __name__ == "__main__":
    main()
