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
    omnibus, pairwise, rest = continuous_tests(frame.set_index("case_id").to_dict("index"), features, groups, context["candidate_patient_ids"], bootstrap_iterations, "wsi")
    pd.DataFrame(omnibus).to_csv(out / "wsi_omnibus.csv", index=False)
    pd.DataFrame(pairwise).to_csv(out / "wsi_pairwise.csv", index=False)
    rest_frame = pd.DataFrame(rest)
    rest_frame.to_csv(out / "wsi_subtype_vs_rest.csv", index=False)
    rest_frame[rest_frame.get("q_value", pd.Series(index=rest_frame.index, dtype=float)).fillna(1.0) < .05].to_csv(out / "wsi_subtype_profile.csv", index=False)
    return pd.DataFrame(rest), frame


def continuous_tests(table: Mapping[str, Mapping[str, Any]], features: list[str], groups: Mapping[str, list[str]], candidate_ids: list[str], bootstrap_iterations: int, prefix: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    omnibus = []
    for feature in features:
        samples = [[float(table[x][feature]) for x in members if x in table and pd.notna(table[x].get(feature))] for members in groups.values()]
        p = None
        if len(samples) > 1 and all(samples):
            try: p = float(kruskal(*samples).pvalue)
            except ValueError: pass
        omnibus.append({"feature": feature, "group_count": len(groups), "available_n": sum(map(len, samples)), "kruskal_p": p, "epsilon_squared": None})
        if p is not None:
            n, k = sum(map(len, samples)), len(samples); h = float(kruskal(*samples).statistic); omnibus[-1]["epsilon_squared"] = max(0., min(1., (h-k+1)/(n-k))) if n > k else 0.
    q = bh([r["kruskal_p"] for r in omnibus])
    for r, v in zip(omnibus, q): r["q_value"] = v
    pairwise = []
    for fidx, feature in enumerate(features):
        current = []
        for a, b in combinations(groups, 2):
            left = [float(table[x][feature]) for x in groups[a] if x in table and pd.notna(table[x].get(feature))]
            right = [float(table[x][feature]) for x in groups[b] if x in table and pd.notna(table[x].get(feature))]
            p = float(mannwhitneyu(left, right).pvalue) if left and right else None
            delta, low, high = bootstrap_delta(left, right, bootstrap_iterations, BASE_SEED + fidx)
            current.append({"feature": feature, "group_a": a, "group_b": b, "n_a": len(left), "n_b": len(right), "median_a": float(np.median(left)) if left else None, "median_b": float(np.median(right)) if right else None, "cliffs_delta": delta, "ci_low": low, "ci_high": high, "p_value": p})
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
            current.append({"subtype_id": sid, "feature": feature, "n_subtype": len(left), "n_rest": len(right), "median_subtype": float(np.median(left)) if left else None, "median_rest": float(np.median(right)) if right else None, "cliffs_delta": delta, "ci_low": low, "ci_high": high, "p_value": p})
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
    omnibus, pairwise, rest = continuous_tests(table, features, groups, context["candidate_patient_ids"], bootstrap_iterations, "ct")
    pd.DataFrame(omnibus).to_csv(out / "ct_omnibus.csv", index=False)
    pd.DataFrame(pairwise).to_csv(out / "ct_pairwise.csv", index=False)
    rest_frame = pd.DataFrame(rest)
    rest_frame.to_csv(out / "ct_subtype_vs_rest.csv", index=False)
    rest_frame[rest_frame.get("q_value", pd.Series(index=rest_frame.index, dtype=float)).fillna(1.0) < .05].to_csv(out / "ct_subtype_profile.csv", index=False)
    return pd.DataFrame(rest)


def rna_analysis(context: Mapping[str, Any], output_root: Path, config_dir: Path, out: Path) -> pd.DataFrame:
    from gseapy import prerank
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    counts_path = output_root / "rna" / "case_raw_counts.csv"
    counts = pd.read_csv(counts_path).set_index("case_id")
    counts.index = counts.index.astype(str)
    candidate = context["candidate_patient_ids"]
    if set(candidate) - set(counts.index): raise ValueError("RNA raw counts lack candidate patients")
    cfg = __import__("yaml").safe_load((config_dir / "subtype_review.yaml").read_text())['rna']
    all_de, all_gsea = [], []
    collections = {"HALLMARK": cfg["hallmark_gene_sets_path"], "REACTOME": cfg["reactome_gene_sets_path"], "KEGG_MEDICUS": cfg["kegg_medicus_gene_sets_path"]}
    gene_sets = {}
    for collection, path in collections.items():
        gs = {}
        for line in Path(path).read_text().splitlines():
            fields = line.rstrip().split("\t")
            if len(fields) > 2: gs[fields[0]] = sorted(set(fields[2:]) & set(counts.columns))
        gene_sets[collection] = {k: v for k, v in gs.items() if len(v) >= int(cfg["min_pathway_overlap"])}
    for sidx, (sid, members) in enumerate(context["subtypes"].items()):
        print(f"[final_subtype_analysis] RNA {sid} ({sidx + 1}/{len(context['subtypes'])})", flush=True)
        target = [x for x in candidate if x in members]; rest = [x for x in candidate if x not in members]
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
            all_de.append({"subtype_id": sid, "gene": gene, "wald_stat": float(value) if np.isfinite(value) else None, "log2_fold_change": float(row.get("log2FoldChange")) if pd.notna(row.get("log2FoldChange")) else None, "p_value": float(row.get("pvalue")) if pd.notna(row.get("pvalue")) else None, "padj": float(row.get("padj")) if pd.notna(row.get("padj")) else None, "target_n": len(target), "rest_n": len(rest)})
        ranking = pd.DataFrame({"gene": stat.index, "stat": stat.values}).replace([np.inf, -np.inf], np.nan).dropna().sort_values(["stat", "gene"], ascending=[False, True])
        ranked_genes = set(ranking["gene"].astype(str))
        for collection, sets in gene_sets.items():
            print(f"[final_subtype_analysis] RNA {sid} / {collection}", flush=True)
            if len(ranking) < 2: continue
            min_overlap = int(cfg["min_pathway_overlap"])
            eligible_sets = {name: sorted(set(genes) & ranked_genes) for name, genes in sets.items()}
            eligible_sets = {name: genes for name, genes in eligible_sets.items() if len(genes) >= min_overlap}
            if not eligible_sets: continue
            res = prerank(rnk=ranking, gene_sets=eligible_sets, min_size=min_overlap, max_size=500, permutation_num=int(cfg.get("gsea_permutations", 1000)), seed=BASE_SEED+sidx, threads=int(cfg.get("gsea_threads", 1)), outdir=None, verbose=False).res2d
            for row in res.to_dict("records"):
                all_gsea.append({"subtype_id": sid, "collection": collection, "pathway": str(row["Term"]), "ES": float(row.get("ES")), "NES": float(row["NES"]), "nominal_p": float(row.get("NOM p-val")), "fdr_q": float(row.get("FDR q-val")), "fwer_p": float(row.get("FWER p-val")), "lead_genes": str(row.get("Lead_genes", "")), "target_n": len(target), "rest_n": len(rest)})
    de = pd.DataFrame(all_de); de.to_csv(out / "rna_deseq2.csv", index=False)
    gsea = pd.DataFrame(all_gsea); gsea.to_csv(out / "rna_gsea.csv", index=False)
    if not gsea.empty:
        pathways = sorted(gsea.loc[gsea.fdr_q < .05, "pathway"].unique())
        matrix = gsea[gsea.pathway.isin(pathways)].pivot_table(index="pathway", columns="subtype_id", values="NES", aggfunc="first").reindex(columns=context["subtype_order"])
    else: matrix = pd.DataFrame(columns=context["subtype_order"])
    matrix.to_csv(out / "rna_pathway_nes_matrix.csv")
    return gsea


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
    rows = []
    for sid, members in groups.items():
        rest = [x for x in candidate if x not in members]
        for feature in interpretation.columns:
            a = int(interpretation.loc[members, feature].sum()); b = len(members)-a; c = int(interpretation.loc[rest, feature].sum()); d = len(rest)-c
            odds, p = fisher_exact([[a,b],[c,d]])
            ci = odds_ratio_with_ci(np.asarray([[a, b], [c, d]], dtype=float))
            rows.append({"subtype_id": sid, "gene": feature, "subtype_mutated_n": a, "subtype_n": len(members), "subtype_frequency": a/len(members), "rest_mutated_n": c, "rest_n": len(rest), "rest_frequency": c/len(rest), "odds_ratio": float(odds), "ci_low": ci.get("odds_ratio_ci95", [None, None])[0] if ci.get("odds_ratio_ci95") else None, "ci_high": ci.get("odds_ratio_ci95", [None, None])[1] if ci.get("odds_ratio_ci95") else None, "p_value": float(p), "driver_panel_member": str(feature).upper() in driver})
    result = pd.DataFrame(rows)
    result["q_global"] = np.nan; result["q_driver"] = np.nan
    for sid, idx in result.groupby("subtype_id").groups.items():
        result.loc[idx, "q_global"] = bh(result.loc[idx, "p_value"].tolist())
        d_idx = [i for i in idx if bool(result.loc[i, "driver_panel_member"])]
        if d_idx: result.loc[d_idx, "q_driver"] = bh(result.loc[d_idx, "p_value"].tolist())
    result.to_csv(out / "wxs_subtype_vs_rest.csv", index=False)
    pd.DataFrame(result[(result.driver_panel_member) & (result.q_driver < .05)]).to_csv(out / "wxs_driver_profile.csv", index=False)
    return result


def known_and_confounders(context: Mapping[str, Any], output_root: Path, config_dir: Path, out: Path) -> None:
    from agents.subtype_review.tools import known_label_echo, confounder_association
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
    conf = confounder_association(states, str(output_root), str(config_dir), clusters, "partition", [])
    json_dump(out / "technical_confounder_association.json", conf)
    return taxonomy


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


def identity_cards(context: Mapping[str, Any], recurrence: pd.DataFrame, wsi: pd.DataFrame, ct: pd.DataFrame, rna: pd.DataFrame, wxs: pd.DataFrame, taxonomy: pd.DataFrame, reps: pd.DataFrame, out: Path) -> None:
    def top_cont(frame, sid):
        if frame.empty: return [], []
        sub = frame[(frame.subtype_id == sid) & (frame.q_value < .05)].copy() if "q_value" in frame else pd.DataFrame()
        up = sub[sub.cliffs_delta > 0].sort_values(["q_value", "cliffs_delta"], ascending=[True, False]).feature.head(3).tolist()
        down = sub[sub.cliffs_delta < 0].sort_values(["q_value", "cliffs_delta"], ascending=[True, False]).feature.head(3).tolist()
        return up, down
    cards = []
    for sid in context["subtype_order"]:
        rec = recurrence[recurrence.subtype_id == sid].iloc[0].to_dict()
        wu, wd = top_cont(wsi, sid); cu, cd = top_cont(ct, sid)
        g = rna[(rna.subtype_id == sid) & (rna.fdr_q < .05)] if not rna.empty else pd.DataFrame()
        pathways = {c: {"up": g[(g.collection == c) & (g.NES > 0)].sort_values(["fdr_q", "NES"], ascending=[True, False]).pathway.head(3).tolist(), "down": g[(g.collection == c) & (g.NES < 0)].sort_values(["fdr_q", "NES"], ascending=[True, True]).pathway.head(3).tolist()} for c in ("HALLMARK", "REACTOME", "KEGG_MEDICUS")}
        x = wxs[(wxs.subtype_id == sid) & (((wxs.driver_panel_member) & (wxs.q_driver < .05)) | ((~wxs.driver_panel_member) & (wxs.q_global < .05)))] if not wxs.empty else pd.DataFrame()
        profiles = {}
        for reference in ("clearcode34", "tcga_m1_m4"):
            selected = taxonomy[(taxonomy.subtype_id == sid) & (taxonomy.reference == reference)] if not taxonomy.empty else pd.DataFrame()
            profiles[reference] = {str(row.reference_label): float(row.fraction) for row in selected.itertuples()}
        cards.append({"subtype_id": sid, "member_count": int(rec["member_count"]), "common_accept_set_count": int(rec["common_accept_set_count"]), "supporting_run_count": int(rec["supporting_run_count"]), "supporting_k_count": int(rec["supporting_k_count"]), "supporting_ks": rec["supporting_ks"], "mean_pair_recurrence_similarity": rec["mean_pair_recurrence_similarity"], "min_pair_recurrence_similarity": rec["min_pair_recurrence_similarity"], "top_wsi_enriched_features": wu, "top_wsi_depleted_features": wd, "top_ct_enriched_features": cu, "top_ct_depleted_features": cd, "top_hallmark_up": pathways["HALLMARK"]["up"], "top_hallmark_down": pathways["HALLMARK"]["down"], "top_reactome_up": pathways["REACTOME"]["up"], "top_reactome_down": pathways["REACTOME"]["down"], "top_kegg_up": pathways["KEGG_MEDICUS"]["up"], "top_kegg_down": pathways["KEGG_MEDICUS"]["down"], "significant_driver_mutations": x[x.driver_panel_member].gene.head(20).tolist(), "significant_other_mutations": x[~x.driver_panel_member].gene.head(20).tolist(), "clearcode34_profile": profiles["clearcode34"], "tcga_mrna_profile": profiles["tcga_m1_m4"], "representative_patient": reps.loc[reps.subtype_id == sid, "case_id"].iloc[0]})
    json_dump(out / "subtype_identity_card.json", cards)
    pd.DataFrame(cards).to_csv(out / "subtype_identity_card.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Characterize frozen patient-recurrence subtypes from production artifacts.")
    parser.add_argument("--output-root", default="output_kirc")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--analysis-root", default=None)
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
    rna = rna_analysis(context, output_root, config_dir, out)
    print("[final_subtype_analysis] WXS", flush=True)
    wxs = wxs_analysis(context, output_root, config_dir, out, args.permutations)
    print("[final_subtype_analysis] taxonomy/confounders", flush=True)
    taxonomy = known_and_confounders(context, output_root, config_dir, out)
    print("[final_subtype_analysis] representatives and identity cards", flush=True)
    reps = representatives(context, output_root, out)
    identity_cards(context, recurrence, wsi_rest, ct_rest, rna, wxs, taxonomy, reps, out)
    subtype_cfg = config_dir / "subtype_review.yaml"
    wxs_cfg = config_dir / "wxs.yaml"
    candidate_cfg = config_dir / "candidate_proposer.yaml"
    source_files = list(context["source_paths"].values()) + [output_root / "rna/case_raw_counts.csv", output_root / "wxs/wxs_discovery_features.csv", output_root / "wxs/wxs_interpretation_features.csv", subtype_cfg, wxs_cfg, candidate_cfg, Path(__file__)]
    try:
        subtype_yaml = yaml.safe_load(subtype_cfg.read_text(encoding="utf-8"))
        source_files.extend(Path(subtype_yaml["rna"][key]) for key in ("hallmark_gene_sets_path", "reactome_gene_sets_path", "kegg_medicus_gene_sets_path"))
    except (KeyError, TypeError):
        pass
    manifest = {"analysis_type": "final_stable_subtype_characterization", "analysis_role": "post_discovery_in_sample_characterization", "final_subtype_source": str(context["source_paths"]["subtypes"].resolve()), "candidate_patient_order": str(context["source_paths"]["order"].resolve()), "core_patient_count": len(context["core_patient_ids"]), "noncore_patient_count": len(context["noncore_patient_ids"]), "subtype_count": len(context["subtypes"]), "subtype_sizes": {k: len(v) for k, v in context["subtypes"].items()}, "active_modalities": list(MODALITIES), "one_vs_rest_reference": "all_other_candidate_cohort_patients", "scripts_dependency": False, "legacy_output_dependency": False, "input_sha256": {str(p): sha256(p) for p in source_files if p.is_file()}}
    json_dump(out / "source_manifest.json", manifest)
    summary = {"status": "complete", "subtype_count": len(context["subtypes"]), "core_patient_count": len(context["core_patient_ids"]), "candidate_patient_count": len(context["candidate_patient_ids"]), "noncore_patient_count": len(context["noncore_patient_ids"]), "subtype_sizes": {k: len(v) for k, v in context["subtypes"].items()}, "analyses": {k: "complete" for k in ("recurrence", "representation", "wsi", "ct", "rna", "wxs", "known_taxonomy", "technical_confounders")}}
    json_dump(out / "analysis_summary.json", summary)
    print("[final_subtype_analysis]")
    for key, value in (("candidate cohort", summary["candidate_patient_count"]), ("core patients", summary["core_patient_count"]), ("noncore patients", summary["noncore_patient_count"]), ("final subtypes", summary["subtype_count"])): print(f"{key}: {value}")
    print("sizes: " + ", ".join(f"{k}={v}" for k, v in summary["subtype_sizes"].items()))
    print(f"output: {out}")


if __name__ == "__main__":
    main()
