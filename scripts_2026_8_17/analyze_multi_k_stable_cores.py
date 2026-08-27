#!/usr/bin/env python3
"""Offline characterization of recurrent multi-K stable cores.

This module reads existing feature, affinity, main-review, and multi-K artifacts.
It never invokes an Agent, changes a partition, retrains a model, or writes PDF.
All interpretations remain descriptive; no core is renamed as a new subtype.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
import sys
import zlib
from itertools import combinations
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))


MODALITIES = ("ct", "wsi", "rna", "genomic", "fused")
CORE_MODALITIES = ("ct", "wsi", "rna", "genomic")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(key for row in rows for key in row)) if rows else fields or []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def rounded(value):
    value = finite(value)
    return round(value, 8) if value is not None else None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bh(values: list[float | None]) -> list[float | None]:
    indexed = [(i, float(v)) for i, v in enumerate(values) if finite(v) is not None]
    result = [None] * len(values)
    if not indexed:
        return result
    indexed.sort(key=lambda item: item[1])
    n = len(indexed)
    running = 1.0
    for rank, (index, p_value) in reversed(list(enumerate(indexed, 1))):
        running = min(running, p_value * n / rank)
        result[index] = running
    return result


def smd(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(right) < 2:
        return None
    a, b = np.asarray(left, float), np.asarray(right, float)
    pooled = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)
    return rounded((a.mean() - b.mean()) / math.sqrt(pooled)) if pooled > 0 else None


def cliffs_delta(left: list[float], right: list[float]) -> float | None:
    if not left or not right:
        return None
    a, b = np.asarray(left), np.asarray(right)
    return rounded((np.greater.outer(a, b).sum() - np.less.outer(a, b).sum()) / (len(a) * len(b)))


def numeric_test(left: list[float], right: list[float]) -> tuple[float | None, float | None]:
    if not left or not right or len(set(left + right)) < 2:
        return None, None
    from scipy.stats import mannwhitneyu
    return smd(left, right), rounded(mannwhitneyu(left, right, alternative="two-sided").pvalue)


def odds_ratio_ci(a: int, b: int, c: int, d: int) -> tuple[float, list[float] | None]:
    odds = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
    se = math.sqrt(sum(1.0 / (x + 0.5) for x in (a, b, c, d)))
    return rounded(odds), [rounded(math.exp(math.log(odds) - 1.96 * se)), rounded(math.exp(math.log(odds) + 1.96 * se))]


def load_states(data_root: Path) -> tuple[dict[str, dict], list[str]]:
    path = data_root / "storage" / "patient_states" / "patient_states.jsonl"
    states = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                state = json.loads(line)
                if state.get("qc") == "success":
                    states[str(state["case_id"])] = state
    return states, sorted(states)


def load_cores(experiment_root: Path) -> tuple[dict[str, list[str]], list[str]]:
    rows = read_csv(experiment_root / "stable_core_membership.csv")
    cores: dict[str, list[str]] = {}
    for row in rows:
        cores.setdefault(str(row["core_id"]), []).append(str(row["patient_id"]))
    cores = {key: sorted(set(value)) for key, value in sorted(cores.items())}
    return cores, sorted({case_id for members in cores.values() for case_id in members})


def load_main_sets(data_root: Path) -> list[dict]:
    summary = json.loads((data_root / "subtype_review" / "final_review_summary.json").read_text(encoding="utf-8"))
    accepted = {str(row["set_id"]) for row in summary.get("accepted_subtype_sets", [])}
    dropped = {str(row["set_id"]) for row in summary.get("dropped_set_registry", [])}
    return [
        {"set_id": str(row["set_id"]), "member_ids": sorted(map(str, row.get("member_ids", []))),
         "decision": "accept" if str(row["set_id"]) in accepted else "drop" if str(row["set_id"]) in dropped else "unknown"}
        for row in summary.get("partition_sets", [])
    ]


def main_mapping(cores: dict[str, set[str] | list[str]], main_sets: list[dict]) -> tuple[list[dict], dict[str, dict[str, int]]]:
    rows, composition = [], {}
    for core_id, core_members in cores.items():
        core_members = set(core_members)
        composition[core_id] = {}
        for main in main_sets:
            members = set(main["member_ids"])
            intersection = len(core_members & members)
            composition[core_id][main["set_id"]] = intersection
            union = len(core_members | members)
            rows.append({
                "core_id": core_id, "core_size": len(core_members), "main_set_id": main["set_id"],
                "main_set_size": len(members), "main_decision": main["decision"], "intersection_n": intersection,
                "core_fraction_in_main": intersection / len(core_members) if core_members else None,
                "main_fraction_captured": intersection / len(members) if members else None,
                "jaccard": intersection / union if union else None,
                "overlap_coefficient": intersection / min(len(core_members), len(members)) if core_members and members else None,
            })
    return rows, composition


def load_table(path: Path) -> tuple[list[str], dict[str, dict[str, float]]]:
    rows = read_csv(path)
    if not rows:
        return [], {}
    features = [key for key in rows[0] if key != "case_id"]
    table = {row["case_id"]: {feature: finite(row.get(feature)) for feature in features} for row in rows}
    return features, table


def values(table: dict[str, dict[str, float]], ids: list[str], feature: str) -> list[float]:
    return [value for case_id in ids if (value := table.get(case_id, {}).get(feature)) is not None]


def core_rest_rows(table: dict[str, dict[str, float]], features: list[str], cores: dict[str, list[str]], all_ids: list[str], name_key: str) -> list[dict]:
    rows = []
    for core_id, members in cores.items():
        rest = [case_id for case_id in all_ids if case_id not in members]
        for feature in features:
            left, right = values(table, members, feature), values(table, rest, feature)
            effect, p_value = numeric_test(left, right)
            rows.append({name_key: feature, "core_id": core_id, "n_core": len(left), "n_rest": len(right),
                         "median_core": rounded(np.median(left)) if left else None, "median_rest": rounded(np.median(right)) if right else None,
                         "mean_core": rounded(np.mean(left)) if left else None, "mean_rest": rounded(np.mean(right)) if right else None,
                         "smd": effect, "direction": "up" if effect is not None and effect > 0 else "down" if effect is not None and effect < 0 else None,
                         "p_value": p_value, "q_value": None, "available_n_core": len(left), "available_n_rest": len(right)})
        q_values = bh([row["p_value"] for row in rows if row["core_id"] == core_id])
        current = [row for row in rows if row["core_id"] == core_id]
        for row, q_value in zip(current, q_values):
            row["q_value"] = rounded(q_value)
    return rows


def pairwise_numeric_rows(table: dict[str, dict[str, float]], cores: dict[str, list[str]], feature: str) -> list[dict]:
    rows = []
    for core_a, core_b in combinations(sorted(cores), 2):
        left, right = values(table, cores[core_a], feature), values(table, cores[core_b], feature)
        effect, p_value = numeric_test(left, right)
        rows.append({"core_a": core_a, "core_b": core_b, "feature": feature, "smd_a_vs_b": effect,
                     "p_value": p_value, "q_value": None, "median_a": rounded(np.median(left)) if left else None,
                     "median_b": rounded(np.median(right)) if right else None})
    q_values = bh([row["p_value"] for row in rows])
    for row, q_value in zip(rows, q_values):
        row["q_value"] = rounded(q_value)
    return rows


def test_rows(groups: dict[str, list[float]], all_values: list[float], feature: str) -> list[dict]:
    table = {"core": {feature: value} for value in []}
    rows = []
    for core_id, left in groups.items():
        right = list(all_values)
        for value in left:
            right.remove(value)
        effect, p_value = numeric_test(left, right)
        rows.append({"core_id": core_id, "feature": feature, "smd": effect, "p_value": p_value, "q_value": None})
    q_values = bh([row["p_value"] for row in rows])
    for row, q_value in zip(rows, q_values):
        row["q_value"] = q_value
    return rows


def load_expression_scores(data_root: Path, config_dir: Path) -> tuple[list[str], dict[str, dict[str, float]]]:
    features, table = load_table(data_root / "rna" / "case_pathway_features.csv")
    gmt_path = Path()
    min_overlap = 15
    try:
        import yaml
        config = yaml.safe_load((config_dir / "subtype_review.yaml").read_text()) or {}
        gmt_path = Path(config["rna"]["pathway_gene_sets_path"])
        min_overlap = int(config.get("rna", {}).get("min_pathway_overlap", 15))
    except Exception:
        gmt_path = config_dir.parent / "tools" / "msigdb" / "h.all.v2026.1.Hs.symbols.gmt"
    pathways = {}
    if gmt_path.is_file():
        for line in gmt_path.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) >= 3:
                genes = [gene for gene in parts[2:] if gene in features]
                if len(genes) >= min_overlap:
                    pathways[parts[0]] = genes
    try:
        from tools.pathway_enrichment import ssgsea_scores
        import pandas as pd
        frame = pd.DataFrame.from_dict(table, orient="index").reindex(columns=features)
        score_frame = ssgsea_scores(frame, pathways, min_overlap)
        if not score_frame.empty:
            return sorted(pathways), {case_id: {pathway: finite(score_frame.loc[case_id, pathway]) for pathway in pathways if case_id in score_frame.index and pathway in score_frame.columns} for case_id in table}
    except Exception:
        pass
    scores: dict[str, dict[str, float]] = {case_id: {} for case_id in table}
    for case_id, row in table.items():
        ranked = {gene: rank for rank, gene in enumerate(sorted(features, key=lambda gene: (row.get(gene) is None, row.get(gene) or 0)), 1)}
        for pathway, genes in pathways.items():
            if genes:
                scores[case_id][pathway] = float(np.mean([ranked[gene] / len(features) for gene in genes]) - 0.5)
    return sorted(pathways), scores


def rna_analysis(data_root: Path, config_dir: Path, cores: dict[str, list[str]], all_ids: list[str]) -> tuple[list[dict], list[dict]]:
    pathways, scores = load_expression_scores(data_root, config_dir)
    rows = []
    for core_id, members in cores.items():
        rest = [case_id for case_id in all_ids if case_id not in members]
        for pathway in pathways:
            left, right = values(scores, members, pathway), values(scores, rest, pathway)
            effect, p_value = numeric_test(left, right)
            rows.append({"core_id": core_id, "pathway": pathway, "n_core": len(left), "n_rest": len(right),
                         "median_core": rounded(np.median(left)) if left else None, "median_rest": rounded(np.median(right)) if right else None,
                         "mean_core": rounded(np.mean(left)) if left else None, "mean_rest": rounded(np.mean(right)) if right else None,
                         "smd": effect, "direction": "up" if effect is not None and effect > 0 else "down" if effect is not None and effect < 0 else None,
                         "p_value": p_value, "q_value": None, "available_n_core": len(left), "available_n_rest": len(right)})
        current = [row for row in rows if row["core_id"] == core_id]
        for row, q_value in zip(current, bh([item["p_value"] for item in current])):
            row["q_value"] = rounded(q_value)
    pair_rows = []
    for core_a, core_b in combinations(sorted(cores), 2):
        current = []
        for pathway in pathways:
            left, right = values(scores, cores[core_a], pathway), values(scores, cores[core_b], pathway)
            effect, p_value = numeric_test(left, right)
            current.append({"core_a": core_a, "core_b": core_b, "pathway": pathway, "smd_a_vs_b": effect,
                            "p_value": p_value, "q_value": None, "median_a": rounded(np.median(left)) if left else None,
                            "median_b": rounded(np.median(right)) if right else None})
        for row, q_value in zip(current, bh([item["p_value"] for item in current])):
            row["q_value"] = rounded(q_value)
        pair_rows.extend(current)
    return rows, pair_rows


def binary_analysis(table: dict[str, dict[str, float]], features: list[str], cores: dict[str, list[str]], all_ids: list[str], prefix: str) -> tuple[list[dict], list[dict]]:
    rest_by_core = {core: [case for case in all_ids if case not in members] for core, members in cores.items()}
    rows = []
    for core_id, members in cores.items():
        current = []
        for feature in features:
            left, right = values(table, members, feature), values(table, rest_by_core[core_id], feature)
            if not left or not right:
                continue
            a, c = int(sum(value > 0 for value in left)), int(sum(value > 0 for value in right))
            b, d = len(left) - a, len(right) - c
            from scipy.stats import fisher_exact
            odds, p_value = fisher_exact([[a, b], [c, d]])
            odds, ci = odds_ratio_ci(a, b, c, d)
            current.append({"core_id": core_id, "gene": feature.removeprefix("mutation::"), "feature": feature,
                            "core_mutated_n": a, "core_n": len(left), "core_frequency": a / len(left),
                            "rest_mutated_n": c, "rest_n": len(right), "rest_frequency": c / len(right),
                            "frequency_difference": a / len(left) - c / len(right), "odds_ratio": odds, "ci_low": ci[0], "ci_high": ci[1],
                            "fisher_p": rounded(p_value), "q_value": None, "wxs_block": prefix})
            if prefix == "cnv_gain_loss":
                current[-1].update({"alteration_frequency": current[-1]["core_frequency"], "rest_alteration_frequency": current[-1]["rest_frequency"],
                                    "core_altered_n": a, "core_total_n": len(left), "rest_altered_n": c, "rest_total_n": len(right)})
        for row, q_value in zip(current, bh([item["fisher_p"] for item in current])):
            row["q_value"] = rounded(q_value)
        rows.extend(current)
    pair_rows = []
    for core_a, core_b in combinations(sorted(cores), 2):
        current = []
        for feature in features:
            left, right = values(table, cores[core_a], feature), values(table, cores[core_b], feature)
            if not left or not right:
                continue
            a, c = int(sum(value > 0 for value in left)), int(sum(value > 0 for value in right))
            b, d = len(left) - a, len(right) - c
            from scipy.stats import fisher_exact
            _, p_value = fisher_exact([[a, b], [c, d]])
            odds, ci = odds_ratio_ci(a, b, c, d)
            current.append({"core_a": core_a, "core_b": core_b, "gene": feature.removeprefix("mutation::"), "feature": feature,
                            "core_a_mutated_n": a, "core_a_n": len(left), "core_a_frequency": a / len(left),
                            "core_b_mutated_n": c, "core_b_n": len(right), "core_b_frequency": c / len(right),
                            "frequency_difference": a / len(left) - c / len(right), "odds_ratio": odds, "ci_low": ci[0], "ci_high": ci[1],
                            "fisher_p": rounded(p_value), "q_value": None, "wxs_block": prefix})
        for row, q_value in zip(current, bh([item["fisher_p"] for item in current])):
            row["q_value"] = rounded(q_value)
        pair_rows.extend(current)
    return rows, pair_rows


def cnv_analysis(table: dict[str, dict[str, float]], features: list[str], cores: dict[str, list[str]], all_ids: list[str], threshold: float = 0.2) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    continuous = core_rest_rows(table, features, cores, all_ids, "feature")
    for row in continuous:
        row["cliffs_delta"] = cliffs_delta(values(table, cores[row["core_id"]], row["feature"]), values(table, [case for case in all_ids if case not in cores[row["core_id"]]], row["feature"]))
        row["direction"] = "higher_in_core" if row["cliffs_delta"] is not None and row["cliffs_delta"] > 0 else "lower_in_core" if row["cliffs_delta"] is not None and row["cliffs_delta"] < 0 else None
    event_features, event_table = [], {case_id: {} for case_id in table}
    for feature in features:
        if feature.startswith("chr") or feature.startswith("locus::"):
            for event, predicate in (("loss", lambda value: value <= -threshold), ("gain", lambda value: value >= threshold)):
                event_feature = f"{feature}::{event}"; event_features.append(event_feature)
                for case_id, row in table.items():
                    if row.get(feature) is not None:
                        event_table[case_id][event_feature] = float(predicate(row[feature]))
    gain_loss, _ = binary_analysis(event_table, event_features, cores, all_ids, "cnv_gain_loss")
    pair_continuous = []
    for core_a, core_b in combinations(sorted(cores), 2):
        for feature in features:
            left, right = values(table, cores[core_a], feature), values(table, cores[core_b], feature)
            effect, p_value = numeric_test(left, right)
            pair_continuous.append({"core_a": core_a, "core_b": core_b, "feature": feature, "cliffs_delta": cliffs_delta(left, right), "p_value": p_value, "q_value": None,
                                    "direction": "higher_in_core_a" if effect is not None and effect > 0 else "lower_in_core_a" if effect is not None and effect < 0 else None})
        current = pair_continuous[-len(features):]
        for row, q_value in zip(current, bh([item["p_value"] for item in current])):
            row["q_value"] = rounded(q_value)
    pair_gain_loss = []
    for core_a, core_b in combinations(sorted(cores), 2):
        for feature in features:
            if not (feature.startswith("chr") or feature.startswith("locus::")):
                continue
            for event, predicate in (("loss", lambda value: value <= -threshold), ("gain", lambda value: value >= threshold)):
                left = [float(predicate(table[case_id][feature])) for case_id in cores[core_a] if table.get(case_id, {}).get(feature) is not None]
                right = [float(predicate(table[case_id][feature])) for case_id in cores[core_b] if table.get(case_id, {}).get(feature) is not None]
                if left and right:
                    from scipy.stats import fisher_exact
                    a, c = int(sum(left)), int(sum(right)); b, d = len(left) - a, len(right) - c
                    _, p_value = fisher_exact([[a, b], [c, d]])
                    odds, ci = odds_ratio_ci(a, b, c, d)
                    pair_gain_loss.append({"core_a": core_a, "core_b": core_b, "feature": f"{feature}::{event}", "core_a_altered_n": a, "core_a_n": len(left), "core_a_frequency": a / len(left),
                                           "core_b_altered_n": c, "core_b_n": len(right), "core_b_frequency": c / len(right), "frequency_difference": a / len(left) - c / len(right),
                                           "odds_ratio": odds, "ci_low": ci[0], "ci_high": ci[1], "fisher_p": rounded(p_value), "q_value": None})
                    pair_gain_loss[-1].update({"alteration_frequency_a": pair_gain_loss[-1]["core_a_frequency"], "alteration_frequency_b": pair_gain_loss[-1]["core_b_frequency"]})
        current = pair_gain_loss[-2 * len([f for f in features if f.startswith("chr") or f.startswith("locus::")]):]
        for row, q_value in zip(current, bh([item["fisher_p"] for item in current])):
            row["q_value"] = rounded(q_value)
    return continuous, gain_loss, pair_continuous, pair_gain_loss


def normalize_affinity(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, float)
    matrix = matrix / np.sqrt(np.outer(np.diag(matrix), np.diag(matrix)))
    matrix = np.clip((matrix + matrix.T) / 2, 0, 1)
    np.fill_diagonal(matrix, 1)
    return matrix


def affinity_characterization(similarity: np.ndarray, ids: list[str], cores: dict[str, list[str]]) -> dict:
    from tools.multimodal_consistency_check import fixed_membership_silhouettes
    similarity = normalize_affinity(similarity)
    distance = 1 - similarity
    labels = {case_id: core_id for core_id, members in cores.items() for case_id in members}
    ordered = [case_id for case_id in ids if case_id in labels]
    label_array = np.asarray([labels[case_id] for case_id in ordered])
    silhouettes = fixed_membership_silhouettes(distance[np.ix_([ids.index(x) for x in ordered], [ids.index(x) for x in ordered])], label_array)
    per_core = {}
    for core_id, members in cores.items():
        values_for_core = [silhouettes[str(i)] for i, case_id in enumerate(ordered) if case_id in members and silhouettes[str(i)] is not None]
        per_core[core_id] = {"silhouette": rounded(np.median(values_for_core)) if values_for_core else None, "available_n": len(values_for_core)}
    core_ids = sorted(cores)
    distances = {a: {b: None for b in core_ids} for a in core_ids}
    for a, b in combinations(core_ids, 2):
        ia, ib = [ordered.index(x) for x in cores[a] if x in ordered], [ordered.index(x) for x in cores[b] if x in ordered]
        value = float(distance[np.ix_(ia, ib)].mean()) if ia and ib else None
        distances[a][b] = distances[b][a] = rounded(value)
    return {"per_core": per_core, "distance_matrix": distances, "ordered_ids": ordered, "similarity": similarity}


def core_embedding_analysis(affinities: dict[str, np.ndarray], ids: list[str], cores: dict[str, list[str]], permutations: int = 999) -> tuple[list[dict], dict, dict]:
    from tools.multimodal_consistency_check import permanova_metrics, permdisp_metrics
    separation, distance_matrices, tests = [], {}, {}
    for modality, matrix in affinities.items():
        result = affinity_characterization(matrix, ids, cores)
        distance_matrices[modality] = result["distance_matrix"]
        for core_id, row in result["per_core"].items():
            separation.append({"modality": modality, "core_id": core_id, **row})
        ordered = result["ordered_ids"]
        labels = np.asarray([next(core for core, members in cores.items() if case_id in members) for case_id in ordered])
        sub = result["similarity"][np.ix_([ids.index(x) for x in ordered], [ids.index(x) for x in ordered])]
        tests[modality] = {"permanova": permanova_metrics(sub, labels, permutations=permutations, seed=42),
                           "permdisp": permdisp_metrics(sub, labels, permutations=permutations, seed=142)}
    return separation, distance_matrices, tests


def load_core_runs(experiment_root: Path) -> list[dict]:
    runs = []
    for path in sorted(experiment_root.glob("run*/K*/final_review_summary.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        accepted = {str(row["set_id"]): set(map(str, row.get("member_ids", []))) for row in report.get("accepted_subtype_sets", [])}
        runs.append({"run_id": f"{path.parents[1].name}_{path.parents[0].name}", "initial_k": int(path.parents[0].name[1:]), "sets": accepted})
    return runs


def core_cooccurrence(experiment_root: Path, cores: dict[str, list[str]]) -> tuple[list[dict], list[dict]]:
    rows, by_k = [], []
    for run in load_core_runs(experiment_root):
        for core_a, core_b in combinations(sorted(cores), 2):
            details = {}
            for core_id in (core_a, core_b):
                candidates = [(set_id, len(set(cores[core_id]) & members)) for set_id, members in run["sets"].items()]
                dominant, count = max(candidates, key=lambda item: (item[1], item[0]), default=(None, 0))
                details[core_id] = (dominant, count / len(cores[core_id]) if cores[core_id] else None)
            a_set, a_fraction = details[core_a]; b_set, b_fraction = details[core_b]
            valid = a_fraction is not None and b_fraction is not None and a_fraction >= 0.5 and b_fraction >= 0.5
            rows.append({"core_a": core_a, "core_b": core_b, "run_id": run["run_id"], "K": run["initial_k"], "dominant_set_a": a_set,
                         "dominant_fraction_a": rounded(a_fraction), "dominant_set_b": b_set, "dominant_fraction_b": rounded(b_fraction),
                         "same_parent_set": int(valid and a_set == b_set)})
    for core_a, core_b in combinations(sorted(cores), 2):
        current = [row for row in rows if row["core_a"] == core_a and row["core_b"] == core_b]
        for k in sorted({row["K"] for row in current}):
            subset = [row for row in current if row["K"] == k]
            by_k.append({"core_a": core_a, "core_b": core_b, "K": k, "same_parent_fraction": rounded(np.mean([row["same_parent_set"] for row in subset])) if subset else None,
                         "valid_repeat_count": sum(row["dominant_fraction_a"] is not None and row["dominant_fraction_b"] is not None and row["dominant_fraction_a"] >= .5 and row["dominant_fraction_b"] >= .5 for row in subset)})
    return rows, by_k


def core_confounds(data_root: Path, config_dir: Path, states: dict[str, dict], cores: dict[str, list[str]]) -> list[dict]:
    from tools.confound import CATEGORICAL_FIELDS, NUMERIC_FIELDS, confounder_values, global_categorical, global_numeric, set_categorical, set_numeric, assign_q_values
    memberships = {key: members for key, members in cores.items()}
    values_by_case = confounder_values(states, str(data_root))
    global_rows = {field: global_categorical(field, memberships, values_by_case, 999) for field in CATEGORICAL_FIELDS}
    global_rows.update({field: global_numeric(field, memberships, values_by_case) for field in NUMERIC_FIELDS})
    set_rows = {core: {**{field: set_categorical(core, field, memberships, values_by_case) for field in CATEGORICAL_FIELDS}, **{field: set_numeric(core, field, memberships, values_by_case) for field in NUMERIC_FIELDS}} for core in cores}
    assign_q_values(global_rows, set_rows)
    rows = []
    for core_id, fields in set_rows.items():
        for field, metric in fields.items():
            if field in CATEGORICAL_FIELDS:
                for level, item in metric.items():
                    rows.append({"core_id": core_id, "field": field, "level": level, **{key: item.get(key) for key in ("set_count", "set_total", "rest_count", "rest_total", "set_fraction", "rest_fraction", "frequency_difference", "odds_ratio", "q_value")}})
            else:
                rows.append({"core_id": core_id, "field": field, "level": "", **{key: metric.get(key) for key in ("set_mean", "rest_mean", "cliffs_delta", "q_value", "direction")}})
    for field, metric in global_rows.items():
        rows.append({"core_id": "GLOBAL", "field": field, "level": "", "cramers_v": metric.get("cramers_v"), "epsilon_squared": metric.get("epsilon_squared"), "q_value": metric.get("q_value"), "available_n": metric.get("available_n"), "missing_n": metric.get("missing_n")})
    return rows


FONT = {"0": "111101101101111", "1": "010110010010111", "2": "111001111100111", "3": "111001111001111", "4": "101101111001001", "5": "111100111001111", "6": "111100111101111", "7": "111001001001001", "8": "111101111101111", "9": "111101111001111", "-": "000000111000000", ".": "000000000000010"}


class Canvas:
    def __init__(self, width: int, height: int):
        self.data = np.full((height, width, 3), 255, dtype=np.uint8)
        self.width, self.height = width, height

    def rectangle(self, x0, y0, x1, y1, color):
        self.data[max(0, int(y0)):min(self.height, int(y1) + 1), max(0, int(x0)):min(self.width, int(x1) + 1)] = color

    def line(self, x0, y0, x1, y1, color, width=1):
        n = max(abs(int(x1) - int(x0)), abs(int(y1) - int(y0)), 1)
        for t in np.linspace(0, 1, n + 1):
            x, y = int(x0 + t * (x1 - x0)), int(y0 + t * (y1 - y0))
            self.rectangle(x - width, y - width, x + width, y + width, color)

    def circle(self, x, y, radius, color):
        yy, xx = np.ogrid[:self.height, :self.width]
        mask = (xx - int(x)) ** 2 + (yy - int(y)) ** 2 <= radius ** 2
        self.data[mask] = color

    def text(self, x, y, text, scale=1, color=(0, 0, 0)):
        for char in str(text):
            bitmap = FONT.get(char.upper())
            if bitmap:
                for index, value in enumerate(bitmap):
                    if value == "1":
                        px, py = index % 3, index // 3
                        self.rectangle(x + px * scale, y + py * scale, x + (px + 1) * scale - 1, y + (py + 1) * scale - 1, color)
            x += 4 * scale


def write_png(path: Path, width: int, height: int, draw) -> Path:
    canvas = Canvas(width, height)
    draw(canvas)
    raw = b"".join(b"\x00" + canvas.data[y].tobytes() for y in range(height))
    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff)
    payload = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(payload)
    return path


def plot_heatmap(path: Path, matrix: np.ndarray, row_labels: list[str], col_labels: list[str], title: str = "") -> None:
    matrix = np.asarray(matrix, float); matrix = np.nan_to_num(matrix, nan=0.0)
    h, w = max(100, 30 + 12 * len(row_labels)), max(180, 50 + 45 * len(col_labels))
    def draw(canvas):
        lo, hi = float(matrix.min(initial=0)), float(matrix.max(initial=0)); scale = max(abs(lo), abs(hi), 1e-9)
        left, top, cell_w, cell_h = 65, 20, max(8, (w - 80) // max(1, len(col_labels))), max(5, (h - 30) // max(1, len(row_labels)))
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                t = max(-1, min(1, value / scale)); color = (int(255 * (1 - max(t, 0))), int(255 * (1 - abs(t))), int(255 * (1 + min(t, 0))))
                canvas.rectangle(left + j * cell_w, top + i * cell_h, left + (j + 1) * cell_w - 1, top + (i + 1) * cell_h - 1, color)
        for j, label in enumerate(col_labels): canvas.text(left + j * cell_w, 4, label[-4:], 1)
        for i, label in enumerate(row_labels): canvas.text(2, top + i * cell_h, label[:10], 1)
    write_png(path, w, h, draw)


def plot_stacked(path: Path, matrix: np.ndarray, row_labels: list[str], col_labels: list[str]) -> None:
    matrix = np.nan_to_num(np.asarray(matrix, float)); width, height = 260, 170
    colors = [(50, 100, 190), (60, 160, 90), (220, 150, 50), (180, 70, 120), (120, 120, 120)]
    def draw(canvas):
        left, top, bar_w, bar_h = 35, 20, max(12, (width - 50) // max(1, len(row_labels)) - 4), 110
        for i, row in enumerate(matrix):
            total = row.sum() or 1; cursor = top + bar_h
            for j, value in enumerate(row):
                segment = int(bar_h * value / total); canvas.rectangle(left + i * (bar_w + 4), cursor - segment, left + i * (bar_w + 4) + bar_w, cursor, colors[j % len(colors)]); cursor -= segment
            canvas.text(left + i * (bar_w + 4), top + bar_h + 6, row_labels[i][-2:], 1)
        for j, label in enumerate(col_labels): canvas.text(3, top + j * 9, label[-4:], 1, colors[j % len(colors)])
    write_png(path, width, height, draw)


def plot_scatter(path: Path, points: list[tuple[float, float, tuple[int, int, int]]], title: str = "") -> None:
    def draw(canvas):
        if not points: return
        xs, ys = np.asarray([p[0] for p in points]), np.asarray([p[1] for p in points]); x0, x1 = xs.min(), xs.max(); y0, y1 = ys.min(), ys.max()
        for x, y, color in points:
            px = 20 + int(180 * (x - x0) / (x1 - x0 + 1e-9)); py = 180 - int(160 * (y - y0) / (y1 - y0 + 1e-9)); canvas.circle(px, py, 3, color)
    write_png(path, 210, 200, draw)


def plot_bubbles(path: Path, rows: list[dict], cores: list[str], pathways: list[str]) -> None:
    values = {(row["core_id"], row["pathway"]): row for row in rows}
    width, height = 70 + 35 * len(cores), 30 + 12 * len(pathways)
    def draw(canvas):
        max_effect = max((abs(row.get("smd") or 0) for row in rows), default=1.0) or 1.0
        for i, pathway in enumerate(pathways):
            canvas.text(2, 20 + i * 12, pathway[:10], 1)
            for j, core in enumerate(cores):
                row = values.get((core, pathway), {})
                effect = float(row.get("smd") or 0); radius = max(1, int(6 * abs(effect) / max_effect))
                color = (210, 50, 50) if effect > 0 else (50, 80, 210)
                canvas.circle(60 + j * 35, 24 + i * 12, radius, color)
                if row.get("q_value") is not None and row["q_value"] <= .05:
                    canvas.line(60 + j * 35 - radius - 1, 24 + i * 12 - radius - 1, 60 + j * 35 + radius + 1, 24 + i * 12 - radius - 1, (0, 0, 0))
        for j, core in enumerate(cores): canvas.text(52 + j * 35, 4, core[-2:], 1)
    write_png(path, width, height, draw)


def plot_oncoplot(path: Path, table: dict[str, dict[str, float]], genes: list[str], cores: dict[str, list[str]], all_ids: list[str]) -> None:
    ordered = [case_id for core in sorted(cores) for case_id in cores[core]]
    width, height = max(180, 45 + 7 * len(ordered)), max(100, 25 + 10 * len(genes))
    def draw(canvas):
        for i, gene in enumerate(genes):
            canvas.text(2, 25 + i * 10, gene.removeprefix("mutation::")[:8], 1)
            for j, case_id in enumerate(ordered):
                value = table.get(case_id, {}).get(gene)
                color = (50, 150, 70) if value and value > 0 else (235, 235, 235)
                canvas.rectangle(42 + j * 7, 23 + i * 10, 47 + j * 7, 29 + i * 10, color)
        offset = 42
        for core in sorted(cores):
            canvas.text(offset, 5, core[-2:], 1); offset += max(1, 7 * len(cores[core]))
    write_png(path, width, height, draw)


def plot_network(path: Path, cores: dict[str, list[str]], cooccurrence: list[dict], rna_corr: dict[tuple[str, str], float | None]) -> None:
    core_ids = sorted(cores); center = (130, 95); positions = {}
    for i, core in enumerate(core_ids):
        angle = 2 * math.pi * i / max(len(core_ids), 1)
        positions[core] = (center[0] + int(75 * math.cos(angle)), center[1] + int(65 * math.sin(angle)))
    def draw(canvas):
        for row in cooccurrence:
            a, b = row["core_a"], row["core_b"]
            if row.get("same_parent_fraction") is not None and row["same_parent_fraction"] > 0:
                canvas.line(*positions[a], *positions[b], (130, 130, 130), max(1, int(5 * row["same_parent_fraction"])))
        for core, (x, y) in positions.items():
            canvas.circle(x, y, max(5, min(16, len(cores[core]) // 2)), (50, 100, 190)); canvas.text(x - 7, y - 3, core[-2:], 1, (255, 255, 255))
    write_png(path, 260, 190, draw)


def feature_matrix(states: dict[str, dict], ids: list[str], modality: str) -> tuple[list[str], np.ndarray]:
    if modality == "ct":
        names = sorted({str(name) for case_id in ids for name in dict(states[case_id].get("ct_evidence", {}).get("features", {}) or {})})
        return names, np.asarray([[finite(dict(states[case_id].get("ct_evidence", {}).get("features", {}) or {}).get(name)) or 0 for name in names] for case_id in ids], float)
    if modality == "wsi":
        vectors = [np.asarray(states[case_id].get("wsi_evidence", {}).get("features", []), float).reshape(-1) for case_id in ids]
        width = max((len(vector) for vector in vectors), default=0)
        return [f"wsi_{i}" for i in range(width)], np.asarray([np.pad(vector, (0, width - len(vector))) for vector in vectors], float)
    if modality == "genomic":
        wxs_names, wxs = load_table(Path(str(states[ids[0]].get("omics_evidence", {}).get("wxs_discovery_feature_path", "")))) if ids else ([], {})
        cnv_names, cnv = load_table(Path(str(states[ids[0]].get("omics_evidence", {}).get("cnv_feature_path", "")))) if ids else ([], {})
        names = [f"wxs::{name}" for name in wxs_names] + [f"cnv::{name}" for name in cnv_names]
        return names, np.asarray([[wxs.get(case_id, {}).get(name, 0) or 0 for name in wxs_names] + [cnv.get(case_id, {}).get(name, 0) or 0 for name in cnv_names] for case_id in ids], float)
    path = Path(str(states[ids[0]].get("omics_evidence", {}).get("rna_feature_path", ""))) if ids else Path()
    names, table = load_table(path)
    return names, np.asarray([[table.get(case_id, {}).get(name) or 0 for name in names] for case_id in ids], float)


def plot_projection(path: Path, points: list[tuple[float, float, tuple[int, int, int]]]) -> None:
    plot_scatter(path, points)


def projection_points(matrix: np.ndarray, ids: list[str], cores: dict[str, list[str]], random_state: int) -> tuple[list[tuple[float, float, tuple[int, int, int]]], str]:
    if len(matrix) < 2 or matrix.shape[1] == 0:
        return [], "unavailable"
    values = np.nan_to_num(np.asarray(matrix, float)); values -= values.mean(axis=0, keepdims=True); values /= np.where(values.std(axis=0, keepdims=True) > 0, values.std(axis=0, keepdims=True), 1)
    method = "pca_fallback"
    try:
        import umap
        coordinates = umap.UMAP(n_components=2, random_state=random_state).fit_transform(values); method = "umap"
    except Exception:
        from sklearn.decomposition import PCA
        coordinates = PCA(n_components=2, random_state=random_state).fit_transform(values)
    colors = {core: ((40 * (index + 1)) % 220, 70 + 25 * (index % 4), 180 - 20 * (index % 5)) for index, core in enumerate(sorted(cores))}
    labels = {case_id: core for core, members in cores.items() for case_id in members}
    return [(float(x), float(y), colors.get(labels.get(case_id), (160, 160, 160))) for case_id, (x, y) in zip(ids, coordinates)], method


def pairwise_similarity(cores, rna_rows, wxs_rows, cnv_rows, distance_matrices, co_run):
    from scipy.stats import pearsonr, spearmanr
    result = []
    core_ids = sorted(cores)
    rna_profiles = {core: {row["pathway"]: row.get("smd") for row in rna_rows if row["core_id"] == core} for core in core_ids}
    wxs_profiles = {core: {row["gene"]: row.get("core_frequency") for row in wxs_rows if row["core_id"] == core} for core in core_ids}
    cnv_profiles = {core: {row["feature"]: row.get("cliffs_delta") for row in cnv_rows if row["core_id"] == core} for core in core_ids}
    for a, b in combinations(core_ids, 2):
        def corr(left, right, method):
            keys = sorted(set(left) & set(right)); x = [left[k] for k in keys if left[k] is not None and right[k] is not None]; y = [right[k] for k in keys if left[k] is not None and right[k] is not None]
            return rounded(method(x, y).statistic) if len(x) >= 3 else None
        same = [row["same_parent_set"] for row in co_run if row["core_a"] == a and row["core_b"] == b]
        pair_rna = [row for row in []]
        result.append({"core_a": a, "core_b": b, "same_parent_fraction": rounded(np.mean(same)) if same else None, "k_same_parent_count": sum(same),
                       "rna_hallmark_profile_pearson": corr(rna_profiles[a], rna_profiles[b], pearsonr), "rna_hallmark_profile_spearman": corr(rna_profiles[a], rna_profiles[b], spearmanr),
                       "rna_pairwise_significant_pathway_count": None, "rna_median_abs_pairwise_smd": None,
                       "mutation_frequency_pearson": corr(wxs_profiles[a], wxs_profiles[b], pearsonr), "mutation_jaccard_high_frequency": None, "wxs_pairwise_fdr_gene_count": None,
                       "cnv_effect_profile_pearson": corr(cnv_profiles[a], cnv_profiles[b], pearsonr), "cnv_pairwise_fdr_feature_count": None,
                       "ct_centroid_distance": distance_matrices.get("ct", {}).get(a, {}).get(b), "wsi_centroid_distance": distance_matrices.get("wsi", {}).get(a, {}).get(b),
                       "genomic_centroid_distance": distance_matrices.get("genomic", {}).get(a, {}).get(b), "fused_centroid_distance": distance_matrices.get("fused", {}).get(a, {}).get(b)})
    return result


def update_pairwise_similarity(rows, rna_pair, wxs_pair, cnv_pair, cores):
    for row in rows:
        a, b = row["core_a"], row["core_b"]
        rna = [item for item in rna_pair if item["core_a"] == a and item["core_b"] == b]
        wxs = [item for item in wxs_pair if item["core_a"] == a and item["core_b"] == b]
        cnv = [item for item in cnv_pair if item["core_a"] == a and item["core_b"] == b]
        row["rna_pairwise_significant_pathway_count"] = sum(item.get("q_value") is not None and item["q_value"] <= .05 for item in rna)
        values = [abs(item["smd_a_vs_b"]) for item in rna if item.get("smd_a_vs_b") is not None]
        row["rna_median_abs_pairwise_smd"] = rounded(np.median(values)) if values else None
        row["wxs_pairwise_fdr_gene_count"] = sum(item.get("q_value") is not None and item["q_value"] <= .05 for item in wxs)
        af_a = {item["gene"]: item.get("core_a_frequency") for item in wxs}; af_b = {item["gene"]: item.get("core_b_frequency") for item in wxs}
        high_a = {gene for gene, value in af_a.items() if value is not None and value >= .5}; high_b = {gene for gene, value in af_b.items() if value is not None and value >= .5}
        row["mutation_jaccard_high_frequency"] = len(high_a & high_b) / len(high_a | high_b) if high_a | high_b else None
        row["cnv_pairwise_fdr_feature_count"] = sum(item.get("q_value") is not None and item["q_value"] <= .05 for item in cnv_pair)
    return rows


def summary_rows(cores, main_sets, composition, rna_rows, wxs_rows, cnv_rows, separation_rows, confounds):
    rows = []
    for core, members in cores.items():
        main = max(main_sets, key=lambda item: composition[core].get(item["set_id"], 0), default={})
        core_rna = sorted([row for row in rna_rows if row["core_id"] == core], key=lambda row: (row["q_value"] is None, row["q_value"] or 1, -abs(row["smd"] or 0)))
        core_wxs = sorted([row for row in wxs_rows if row["core_id"] == core], key=lambda row: (row["q_value"] is None, row["q_value"] or 1, -abs(row["frequency_difference"] or 0)))
        core_cnv = sorted([row for row in cnv_rows if row["core_id"] == core], key=lambda row: (row["q_value"] is None, row["q_value"] or 1, -abs(row.get("cliffs_delta") or 0)))
        sep = {row["modality"]: row.get("silhouette") for row in separation_rows if row["core_id"] == core}
        tss = sorted([row for row in confounds if row["core_id"] == core and row["field"] == "tissue_source_site"], key=lambda row: (row.get("q_value") is None, row.get("q_value") or 1))
        item = {"core_id": core, "n": len(members), "main_dominant_candidate": main.get("set_id"), "main_candidate_fraction": composition[core].get(main.get("set_id"), 0) / len(members) if members else None}
        for prefix, source, name_key, effect_key in (("top_rna_pathway", core_rna, "pathway", "smd"), ("top_wxs_gene", core_wxs, "gene", "odds_ratio"), ("top_cnv_feature", core_cnv, "feature", "cliffs_delta")):
            for index, row in enumerate(source[:5], 1): item.update({f"{prefix}_{index}": row.get(name_key), f"{prefix}_{index}_effect": row.get(effect_key), f"{prefix}_{index}_q": row.get("q_value")})
        item.update({f"{modality}_silhouette": sep.get(modality) for modality in MODALITIES}); item.update({"tss_top_association": tss[0].get("level") if tss else None, "tss_q": tss[0].get("q_value") if tss else None})
        rows.append(item)
    return rows


def build_summary(cores, all_ids, mapping_rows, separation_rows, confound_rows):
    return {"patient_count": len(all_ids), "stable_core_patient_count": sum(len(v) for v in cores.values()), "non_core_patient_count": len(set(all_ids) - set().union(*(set(v) for v in cores.values()))), "core_count": len(cores), "mapping_rows": len(mapping_rows), "embedding_rows": len(separation_rows), "confound_rows": len(confound_rows)}


def run(data_root: Path, experiment_root: Path, output_root: Path, config_dir: Path, top_pathways: int = 25, top_cnv: int = 25, random_state: int = 42, force: bool = False) -> dict:
    if output_root.exists() and any(output_root.iterdir()) and not force:
        raise FileExistsError(f"Output exists; pass --force to overwrite: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    figure_root = output_root / "figures"; figure_root.mkdir(exist_ok=True)
    states, all_ids = load_states(data_root)
    cores, core_ids = load_cores(experiment_root)
    main_sets = load_main_sets(data_root)
    mapping_rows, composition = main_mapping(cores, main_sets)
    write_csv(output_root / "stable_core_main_mapping.csv", mapping_rows)
    write_csv(output_root / "stable_core_main_composition.csv", [{"core_id": core, **counts} for core, counts in composition.items()])

    rna_rows, rna_pair_rows = rna_analysis(data_root, config_dir, cores, all_ids)
    write_csv(output_root / "rna_hallmark_core_vs_rest.csv", rna_rows)
    write_csv(output_root / "rna_hallmark_core_pairwise.csv", rna_pair_rows)
    wxs_features, wxs_table = load_table(data_root / "wxs" / "wxs_discovery_features.csv")
    wxs_rows, wxs_pair_rows = binary_analysis(wxs_table, wxs_features, cores, all_ids, "mutation")
    write_csv(output_root / "wxs_core_vs_rest.csv", wxs_rows); write_csv(output_root / "wxs_core_pairwise.csv", wxs_pair_rows)
    cnv_features, cnv_table = load_table(data_root / "cnv" / "case_features.csv")
    cnv_cont, cnv_event, cnv_pair_cont, cnv_pair_event = cnv_analysis(cnv_table, cnv_features, cores, all_ids)
    write_csv(output_root / "cnv_continuous_core_vs_rest.csv", cnv_cont); write_csv(output_root / "cnv_gain_loss_core_vs_rest.csv", cnv_event)
    write_csv(output_root / "cnv_continuous_core_pairwise.csv", cnv_pair_cont); write_csv(output_root / "cnv_gain_loss_core_pairwise.csv", cnv_pair_event)

    affinity_paths = {"ct": data_root / "candidate_subtype" / "ct_affinity.npy", "wsi": data_root / "candidate_subtype" / "wsi_affinity.npy", "rna": data_root / "candidate_subtype" / "rna_affinity.npy", "genomic": data_root / "wxs" / "genomic_affinity.npy", "fused": data_root / "candidate_subtype" / "fused_similarity.npy"}
    affinity_ids = json.loads((data_root / "candidate_subtype" / "affinity_patient_order.json").read_text(encoding="utf-8"))
    affinities = {name: np.load(path) for name, path in affinity_paths.items() if path.is_file()}
    separation_rows, distance_matrices, tests = core_embedding_analysis(affinities, affinity_ids, cores)
    write_csv(output_root / "embedding_core_separation.csv", separation_rows)
    for modality, matrix in distance_matrices.items(): write_csv(output_root / f"{modality}_core_distance_matrix.csv", [{"core_id": a, **values} for a, values in matrix.items()])
    write_csv(output_root / "core_permanova.csv", [{"modality": m, **tests[m]["permanova"]} for m in tests]); write_csv(output_root / "core_permdisp.csv", [{"modality": m, **tests[m]["permdisp"]} for m in tests])

    co_run, co_k = core_cooccurrence(experiment_root, cores)
    write_csv(output_root / "stable_core_cooccurrence_by_run.csv", co_run); write_csv(output_root / "stable_core_cooccurrence_by_k.csv", co_k)
    confounds = core_confounds(data_root, config_dir, states, cores); write_csv(output_root / "stable_core_confounders.csv", confounds)
    similarity_rows = update_pairwise_similarity(pairwise_similarity(cores, rna_rows, wxs_rows, cnv_cont, distance_matrices, co_run), rna_pair_rows, wxs_pair_rows, cnv_pair_cont, cores)
    write_csv(output_root / "stable_core_pairwise_similarity.csv", similarity_rows)
    overview_rows = summary_rows(cores, main_sets, composition, rna_rows, wxs_rows, cnv_cont, separation_rows, confounds)
    write_csv(output_root / "stable_core_summary_multimodal.csv", overview_rows)

    core_order = sorted(cores); main_order = [row["set_id"] for row in main_sets]; matrix = np.asarray([[composition[c].get(s, 0) / len(cores[c]) for s in main_order] for c in core_order])
    plot_stacked(figure_root / "core_main_composition.png", np.asarray([[composition[c].get(s, 0) for s in main_order] for c in core_order]), core_order, main_order)
    pathways = sorted({row["pathway"] for row in rna_rows}); selected = sorted(pathways, key=lambda p: min((abs(row["smd"] or 0) for row in rna_rows if row["pathway"] == p), default=0), reverse=True)[:top_pathways]
    plot_heatmap(figure_root / "pathway_smd_heatmap.png", np.asarray([[next((row["smd"] or 0 for row in rna_rows if row["core_id"] == core and row["pathway"] == pathway), 0) for core in core_order] for pathway in selected]), selected, core_order)
    cnv_selected = sorted(cnv_cont, key=lambda row: abs(row.get("cliffs_delta") or 0), reverse=True)[:top_cnv]
    plot_heatmap(figure_root / "cnv_effect_heatmap.png", np.asarray([[next((row.get("cliffs_delta") or 0 for row in cnv_cont if row["core_id"] == core and row["feature"] == item["feature"]), 0) for core in core_order] for item in cnv_selected]), [row["feature"] for row in cnv_selected], core_order)
    plot_bubbles(figure_root / "pathway_bubble_plot.png", rna_rows, core_order, selected)
    driver_genes = [feature for feature in wxs_features if feature.startswith("mutation::")]
    plot_oncoplot(figure_root / "driver_mutation_oncoplot.png", wxs_table, driver_genes, cores, all_ids)
    parent_global = {(a, b): float(np.mean([row["same_parent_set"] for row in co_run if row["core_a"] == a and row["core_b"] == b])) if any(row["core_a"] == a and row["core_b"] == b for row in co_run) else 0 for a, b in combinations(core_order, 2)}
    parent_matrix = np.asarray([[1 if a == b else parent_global.get((a, b), parent_global.get((b, a), 0)) for b in core_order] for a in core_order])
    plot_heatmap(figure_root / "core_same_parent_heatmap.png", parent_matrix, core_order, core_order)
    pair_lookup = {(row["core_a"], row["core_b"]): row for row in similarity_rows}
    rna_matrix = np.eye(len(core_order)); cnv_matrix = np.eye(len(core_order));
    for i, a in enumerate(core_order):
        for j, b in enumerate(core_order):
            item = pair_lookup.get((a, b)) or pair_lookup.get((b, a))
            if item: rna_matrix[i, j] = item.get("rna_hallmark_profile_pearson") or 0; cnv_matrix[i, j] = item.get("cnv_effect_profile_pearson") or 0
    plot_heatmap(figure_root / "core_rna_similarity_heatmap.png", rna_matrix, core_order, core_order)
    plot_heatmap(figure_root / "core_cnv_similarity_heatmap.png", cnv_matrix, core_order, core_order)
    plot_heatmap(figure_root / "core_pairwise_similarity_heatmap.png", rna_matrix, core_order, core_order)
    plot_network(figure_root / "core_relationship_network.png", cores, [{"core_a": a, "core_b": b, "same_parent_fraction": parent_global.get((a, b), 0)} for a, b in combinations(core_order, 2)], {(row["core_a"], row["core_b"]): row.get("rna_hallmark_profile_pearson") for row in similarity_rows})
    projection_methods = {}
    for modality in MODALITIES:
        if modality == "fused":
            matrix = affinities[modality]
            plot_ids = affinity_ids
        elif modality in CORE_MODALITIES:
            _, matrix = feature_matrix(states, affinity_ids, modality)
            plot_ids = affinity_ids
        else:
            continue
        points, method = projection_points(matrix, plot_ids, cores, random_state)
        projection_methods[modality] = method
        plot_projection(figure_root / f"{modality}_core_umap.png", points)
    generated = sorted(str(path.relative_to(output_root)) for path in output_root.rglob("*") if path.is_file() and path.suffix != ".pdf")
    manifest = {"core_count": len(cores), "patient_count": len(all_ids), "core_patient_count": len(core_ids), "non_core_patient_count": len(set(all_ids) - set(core_ids)),
                "source_multi_k_summary_sha256": file_sha256(experiment_root / "stable_core_summary.csv"), "source_main_partition_sha256": file_sha256(data_root / "subtype_review" / "final_partition_sets.json"),
                "analysis_parameters": {"top_pathways": top_pathways, "top_cnv": top_cnv, "random_state": random_state, "png_only": True, "projection_methods": projection_methods}, "generated_files": generated}
    write_json(output_root / "stable_core_analysis_manifest.json", manifest)
    summary = build_summary(cores, all_ids, mapping_rows, separation_rows, confounds); write_json(output_root / "stable_core_analysis_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("output_kirc"))
    parser.add_argument("--experiment-root", type=Path, default=Path("output_kirc_v11/experiment_multi_k_accepted_core_stability"))
    parser.add_argument("--output-root", type=Path, default=Path("output_kirc_v11/stable_core_multimodal_analysis"))
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--top-pathways", type=int, default=25); parser.add_argument("--top-cnv", type=int, default=25)
    parser.add_argument("--random-state", type=int, default=42); parser.add_argument("--force", action="store_true")
    args = parser.parse_args(); print(json.dumps(run(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
