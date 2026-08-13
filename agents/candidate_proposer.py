from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from utils.candidate_clustering_outputs import (
    canonical_partition,
    cluster_sizes_from_labels,
    consensus_matrix_from_partitions,
    consensus_silhouette_score,
    fit_kmedoids,
    save_candidate_clustering_outputs,
)
from utils.cluster_store import save_candidate_clusters
from utils.llm_utils import load_yaml_file, resolve_api_key
from utils.patient_store import save_patient_states


def snf_fuse(
    feature_matrices: list[np.ndarray],
    *,
    k: int,
    iterations: int,
    mu: float,
    alpha: float,
) -> np.ndarray:
    from tools.evidence_features import snf_fuse_feature_matrices

    return snf_fuse_feature_matrices(
        feature_matrices, k=k, iterations=iterations, mu=mu, alpha=alpha
    )


def calculate_consensus_cdf(
    consensus_values: np.ndarray,
    *,
    thresholds: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    values = np.nan_to_num(np.asarray(consensus_values, dtype=float), nan=0.0)
    threshold_values = (
        np.asarray(thresholds, dtype=float)
        if thresholds is not None
        else np.linspace(0.0, 1.0, 101)
    )
    cdf_values = np.asarray(
        [np.mean(values <= threshold) for threshold in threshold_values],
        dtype=float,
    )
    cdf_area = float(
        np.sum((cdf_values[:-1] + cdf_values[1:]) * 0.5 * np.diff(threshold_values))
    )
    return threshold_values, cdf_values, cdf_area


def calculate_delta_area(
    cdf_area: float,
    *,
    previous_cdf_area: float,
) -> tuple[float, float]:
    delta_area = float(cdf_area) - float(previous_cdf_area)
    relative_delta_area = (
        delta_area / max(abs(float(previous_cdf_area)), 1e-8)
        if previous_cdf_area
        else delta_area
    )
    return float(delta_area), float(relative_delta_area)


def calculate_pac(consensus_values: np.ndarray, *, lower: float, upper: float) -> float:
    values = np.nan_to_num(np.asarray(consensus_values, dtype=float), nan=0.0)
    return float(np.mean((values > float(lower)) & (values < float(upper))))


# ── LLM K selection ──────────────────────────────────────────────────────


def build_k_selection_evidence(
    consensus_records: list[dict[str, Any]],
    min_cluster_size: int,
) -> dict[str, Any]:
    """Build K-selection evidence from consensus records for LLM review."""
    k_evidence = []
    eligible_ks = []
    candidate_ks = sorted(
        int(record.get("n_clusters", 0) or 0) for record in consensus_records
    )
    baseline_k = min(candidate_ks)

    for record in consensus_records:
        k = int(record.get("n_clusters", 0) or 0)
        sizes = list(record.get("cluster_sizes", []) or [])
        eligible = min(sizes) >= min_cluster_size if sizes else False
        if eligible:
            eligible_ks.append(k)

        matrix = np.asarray(record["consensus"], dtype=float)
        labels = np.asarray(record["labels"], dtype=int)

        cluster_scores = []
        for label in sorted(set(labels)):
            indexes = np.where(labels == label)[0]
            block = matrix[np.ix_(indexes, indexes)]
            upper = block[np.triu_indices(len(indexes), k=1)]
            if len(upper) > 0:
                cluster_scores.append(float(np.mean(upper)))
        cluster_consensus = {
            "min": round(float(np.min(cluster_scores)), 6) if cluster_scores else 0.0
        }

        item_scores = []
        for index, label in enumerate(labels):
            members = np.where(labels == label)[0]
            others = members[members != index]
            if len(others) > 0:
                item_scores.append(float(np.mean(matrix[index, others])))
        item_consensus = {
            "p10": round(float(np.quantile(item_scores, 0.1)), 6)
            if item_scores
            else 0.0
        }

        k_evidence.append(
            {
                "k": k,
                "eligible": eligible,
                "relative_delta_area": None
                if k == baseline_k
                else round(
                    float(record.get("relative_delta_area", 0.0) or 0.0), 6
                ),
                "pac": round(float(record.get("pac", 0.0) or 0.0), 6),
                "cluster_sizes": sizes,
                "cluster_consensus": cluster_consensus,
                "item_consensus": item_consensus,
            }
        )

    return {
        "task": "select_initial_candidate_k",
        "selection_scope": "candidate_proposer_only",
        "min_cluster_size": min_cluster_size,
        "candidate_k_values": candidate_ks,
        "eligible_k_values": eligible_ks,
        "k_evidence": k_evidence,
    }


def select_k_with_llm(
    consensus_records: list[dict[str, Any]],
    *,
    output_root: str,
    config_dir: str,
    min_cluster_size: int,
    llm_client=None,
) -> dict[str, Any]:
    """Use LLM to select exactly one K from consensus records.

    LLM failure → RuntimeError. No PAC/silhouette fallback.
    """
    from openai import OpenAI

    evidence = build_k_selection_evidence(consensus_records, min_cluster_size)
    eligible = list(evidence.get("eligible_k_values", []))
    candidate = list(evidence.get("candidate_k_values", []))

    if not eligible:
        raise RuntimeError("No candidate K satisfies min_cluster_size.")

    config_path = Path(config_dir) / "candidate_k_selector.yaml"
    if not config_path.exists():
        config_path = (
            Path(__file__).resolve().parent.parent
            / "configs"
            / "candidate_k_selector.yaml"
        )
    selector_config = load_yaml_file(str(config_path))
    llm_config = dict(selector_config["llm"])

    prompt_path = Path(selector_config["prompt_path"])
    if not prompt_path.is_absolute():
        prompt_path = Path(__file__).resolve().parent.parent / prompt_path
    system_prompt = prompt_path.read_text(encoding="utf-8")

    if llm_client is None:
        llm_client = OpenAI(
            api_key=resolve_api_key(llm_config),
            base_url=str(llm_config["base_url"]),
            timeout=120.0,
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(evidence, ensure_ascii=False, indent=2)},
    ]

    json_retries = int(llm_config["json_retries"])
    max_attempts = json_retries + 1
    last_error = ""
    raw_response = ""
    attempts = []
    audit_dir = (
        Path(output_root)
        / "candidate_subtype"
        / "consensus_cluster"
    )
    audit_dir.mkdir(parents=True, exist_ok=True)

    for attempt in range(max_attempts):
        attempt_audit = {"attempt": attempt + 1}
        try:
            request = {
                "model": str(llm_config["model_name"]),
                "messages": messages,
                "temperature": float(llm_config["temperature"]),
                "max_tokens": int(llm_config["max_new_tokens"]),
                "response_format": {"type": "json_object"},
            }
            if str(llm_config["provider"]).lower() == "deepseek":
                request["extra_body"] = {
                    "thinking": {"type": "disabled"}
                }
            response = llm_client.chat.completions.create(
                **request
            )
            choice = response.choices[0]
            message = choice.message
            raw_response = str(message.content or "")
            reasoning_content = str(
                getattr(message, "reasoning_content", "") or ""
            )
            attempt_audit.update(
                {
                    "response_id": str(getattr(response, "id", "") or ""),
                    "response_model": str(
                        getattr(response, "model", "") or ""
                    ),
                    "finish_reason": str(
                        getattr(choice, "finish_reason", "") or ""
                    ),
                    "content_empty": not bool(raw_response.strip()),
                    "content_length": len(raw_response),
                    "reasoning_content_present": bool(reasoning_content),
                    "reasoning_content_length": len(reasoning_content),
                }
            )
            if not raw_response.strip():
                raise ValueError("LLM returned empty message.content")
            decision = json.loads(raw_response)
        except Exception as e:
            last_error = str(e)
            attempt_audit["error"] = last_error
            attempts.append(attempt_audit)
            if attempt < max_attempts - 1:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your previous response was not valid JSON: {last_error}\n"
                            "Return one corrected JSON object only."
                        ),
                    }
                )
            continue

        # Validate decision
        errors = []
        required_keys = {
            "selected_k",
            "confidence",
            "reasoning_summary",
            "evidence_refs",
        }
        missing = required_keys - set(decision.keys())
        if missing:
            errors.append(f"Missing required keys: {sorted(missing)}")
        if not isinstance(decision.get("selected_k"), int):
            errors.append("selected_k must be an integer")
        if decision.get("confidence") not in {"high", "medium", "low"}:
            errors.append("confidence must be high|medium|low")
        if (
            not isinstance(decision.get("reasoning_summary"), str)
            or not decision["reasoning_summary"].strip()
        ):
            errors.append("reasoning_summary must be a non-empty string")
        ev_refs = decision.get("evidence_refs")
        if not isinstance(ev_refs, list) or len(ev_refs) == 0:
            errors.append("evidence_refs must be a non-empty list")
        elif not all(isinstance(r, str) and r.strip() for r in ev_refs):
            errors.append("each evidence_ref must be a non-empty string")
        if decision.get("selected_k") not in candidate:
            errors.append(
                f"selected_k {decision['selected_k']} not in candidate K values {candidate}"
            )
        if decision.get("selected_k") not in eligible:
            errors.append(
                f"selected_k {decision['selected_k']} is not eligible "
                f"(min_cluster_size={min_cluster_size})"
            )

        if errors:
            last_error = "; ".join(errors)
            attempt_audit["error"] = last_error
            attempts.append(attempt_audit)
            if attempt < max_attempts - 1:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your previous response violated the JSON or K-selection contract: {last_error}\n"
                            "Return one corrected JSON object only."
                        ),
                    }
                )
            continue

        # Success
        attempt_audit["error"] = ""
        attempts.append(attempt_audit)
        audit_path = audit_dir / "k_selection_llm_audit.json"
        audit = {
            "selection_metric": "llm_consensus_review",
            "model": {
                "provider": str(llm_config["provider"]),
                "model_name": str(llm_config["model_name"]),
                "temperature": float(llm_config["temperature"]),
            },
            "prompt_path": str(prompt_path),
            "evidence_input": evidence,
            "raw_response": raw_response,
            "decision": decision,
            "attempts": attempts,
        }
        with audit_path.open("w", encoding="utf-8") as f:
            json.dump(audit, f, indent=2, ensure_ascii=False)

        return {
            **decision,
            "audit_path": str(audit_path),
            "diagnostics": evidence["k_evidence"],
        }

    failure_audit_path = audit_dir / "k_selection_llm_failure_audit.json"
    failure_audit_path.write_text(
        json.dumps(
            {
                "selection_metric": "llm_consensus_review",
                "model": {
                    "provider": str(llm_config["provider"]),
                    "model_name": str(llm_config["model_name"]),
                    "temperature": float(llm_config["temperature"]),
                },
                "prompt_path": str(prompt_path),
                "evidence_input": evidence,
                "attempts": attempts,
                "last_error": last_error,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    raise RuntimeError(
        f"LLM K selection failed after {max_attempts} attempts: {last_error}; "
        f"audit={failure_audit_path}"
    )


def build_feature_store_payload(
    patient_states: list[Mapping[str, Any]],
    *,
    config_dir: str = "",
    output_root: str = "",
) -> dict[str, Any]:
    """Read evidence-stage affinity artifacts for candidate generation.

    Raw-feature compatibility is retained only for legacy fixtures;
    candidate generation itself never performs modality preprocessing.
    """
    if not output_root:
        from tools.evidence_features import build_legacy_feature_payload

        return build_legacy_feature_payload(patient_states, config_dir=config_dir)
    eligible = [dict(item) for item in patient_states if item.get("qc") == "success"]
    if not eligible:
        empty = np.zeros((0, 0), dtype=float)
        return {"eligible_patient_ids": [], "z_snf": empty, "modality_affinities": {}, "feature_engineering_audit": {}}
    from utils.llm_utils import load_yaml_file

    snf_config = load_yaml_file(Path(config_dir or "configs") / "snf.yaml")
    paths = dict(eligible[0].get("omics_evidence", {}).get("modality_affinity_paths", {}))
    required = {"ct", "wsi", "rna", "genomic"}
    if not required.issubset(paths):
        raise ValueError("Evidence-stage modality affinity artifacts are incomplete.")
    order_path = Path(str(eligible[0]["omics_evidence"]["modality_affinity_patient_order_path"]))
    order = json.loads(order_path.read_text(encoding="utf-8"))
    patient_ids = [str(item.get("case_id", "")) for item in eligible]
    if [str(value) for value in order] != patient_ids:
        raise ValueError("Precomputed modality affinity patient order mismatch.")
    modality_affinities = {name: np.asarray(np.load(Path(paths[name])), dtype=float) for name in required}
    expected_shape = (len(patient_ids), len(patient_ids))
    if any(matrix.shape != expected_shape for matrix in modality_affinities.values()):
        raise ValueError("Evidence-stage modality affinity shapes do not match the patient cohort.")
    audit_path = Path(str(eligible[0]["omics_evidence"].get("multimodal_audit_path", "")))
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else {}
    empty = np.zeros((len(patient_ids), 0), dtype=float)
    return {
        "eligible_patient_ids": patient_ids,
        "z_ct": empty,
        "z_wsi": empty,
        "z_rna": empty,
        "z_wxs": empty,
        "z_snf": fuse_affinities([modality_affinities[name] for name in ("ct", "wsi", "rna", "genomic")], snf_config),
        "snf_config": snf_config,
        "ct_feature_names": [],
        "ct_ccc_filter": dict(audit.get("ct", {}) or {}),
        "modality_affinities": modality_affinities,
        "feature_engineering_audit": audit,
    }


def candidate_proposer(
    patient_states: list[dict[str, Any]],
    *,
    output_root: str,
    config_dir: str = "",
) -> dict[str, Any]:
    feature_payload = build_feature_store_payload(
        patient_states,
        config_dir=config_dir,
        output_root=output_root,
    )
    ccc_audit_path = Path(output_root) / "candidate_subtype" / "ct_ccc_filter.json"
    ccc_audit_path.parent.mkdir(parents=True, exist_ok=True)
    ccc_audit_path.write_text(
        json.dumps(
            dict(feature_payload.get("ct_ccc_filter", {}) or {}),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    feature_audit_path = (
        Path(output_root)
        / "candidate_subtype"
        / "feature_engineering_audit.json"
    )
    feature_audit_path.write_text(
        json.dumps(
            feature_payload.get(
                "feature_engineering_audit",
                {},
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    affinity_order_path = (
        Path(output_root) / "candidate_subtype" / "affinity_patient_order.json"
    )
    affinity_order_path.write_text(
        json.dumps(
            feature_payload.get("eligible_patient_ids", []),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    fused_similarity_path = (
        Path(output_root) / "candidate_subtype" / "fused_similarity.npy"
    )
    if "z_snf" in feature_payload:
        np.save(fused_similarity_path, feature_payload["z_snf"])
    eligible_states = [
        dict(item) for item in patient_states if item.get("qc") == "success"
    ]
    patient_ids = [str(item.get("case_id", "")) for item in eligible_states]
    n_cases = len(patient_ids)
    candidate_clusters: list[dict[str, Any]] = []
    if not eligible_states:
        print(
            "[candidate_cluster_generator] No QC-passed cases; skip candidate clustering.",
            flush=True,
        )
    elif n_cases == 1:
        print(
            "[candidate_cluster_generator] Only one QC-passed case; create one cluster.",
            flush=True,
        )
        candidate_clusters = [
            {
                "cluster_id": "C1",
                "member_ids": patient_ids,
                "source_views": ["snf"],
                "status": "under_review",
                "generator": {
                    "algorithm": "consensus_hierarchical",
                    "n_clusters": 1,
                    "seed": None,
                    "partition_id": "consensus_hierarchical_K1",
                    "cluster_label": 0,
                },
                "consensus": {
                    "partition_count": 0,
                    "secondary_algorithm": "hierarchical",
                },
            }
        ]
    else:
        z_snf = np.nan_to_num(
            np.asarray(feature_payload.get("z_snf", []), dtype=float),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if z_snf.shape != (n_cases, n_cases):
            print(
                "[candidate_cluster_generator] Missing SNF matrix; skip candidate clustering.",
                flush=True,
            )
        else:
            from sklearn.cluster import AgglomerativeClustering, SpectralClustering

            z_snf = (z_snf + z_snf.T) / 2.0
            z_snf = np.clip(z_snf, 0.0, 1.0)
            np.fill_diagonal(z_snf, 1.0)
            snf_distance = 1.0 - z_snf
            np.fill_diagonal(snf_distance, 0.0)
            config_path = (
                Path(config_dir).expanduser() if config_dir else Path("configs")
            )
            cluster_config = load_yaml_file(config_path / "candidate_clustering.yaml")
            repeat_count = int(cluster_config["repeat_count"])
            max_clusters = int(cluster_config["max_clusters"])
            if repeat_count < 1:
                raise ValueError("candidate_clustering.repeat_count must be >= 1")
            if max_clusters < 2:
                raise ValueError("candidate_clustering.max_clusters must be >= 2")
            max_cluster_value = min(max_clusters, n_cases)
            algorithms_config = dict(cluster_config["algorithms"])
            consensus_linkage = str(cluster_config["consensus_linkage"])
            pac_lower = float(cluster_config["pac_lower"])
            pac_upper = float(cluster_config["pac_upper"])
            selection_method = str(cluster_config["selection_method"])
            min_cluster_size = int(cluster_config["min_cluster_size"])
            random_seed = int(cluster_config["random_seed"])
            configured_algorithms = [
                (algorithm, dict(algorithms_config[algorithm]))
                for algorithm in ("hierarchical", "spectral", "kmedoids")
                if algorithm in algorithms_config
            ]
            rng = np.random.default_rng(random_seed)
            clustering_jobs = []
            for algorithm, algorithm_config in configured_algorithms:
                for n_clusters in range(2, max_cluster_value + 1):
                    for run_index in range(1, repeat_count + 1):
                        run_config = dict(algorithm_config)
                        if algorithm == "hierarchical":
                            linkage_options = list(run_config.pop("linkage_options"))
                            run_config["linkage"] = str(rng.choice(linkage_options))
                        elif algorithm == "spectral":
                            assign_labels_options = list(
                                run_config.pop("assign_labels_options")
                            )
                            run_config["assign_labels"] = str(
                                rng.choice(assign_labels_options)
                            )
                        elif algorithm == "kmedoids":
                            init_options = list(run_config.pop("init_options"))
                            run_config["init"] = str(rng.choice(init_options))
                        clustering_jobs.append(
                            {
                                "algorithm": algorithm,
                                "run_index": run_index,
                                "n_clusters": n_clusters,
                                "seed": int(rng.integers(0, np.iinfo(np.int32).max)),
                                "algorithm_config": run_config,
                            }
                        )
            n_cluster_values = sorted(
                {int(job["n_clusters"]) for job in clustering_jobs}
            )
            print(
                "[candidate_cluster_generator] "
                f"Start multi-candidate clustering: cases={n_cases}, "
                f"repeat_count={repeat_count}, max_clusters={max_cluster_value}, "
                f"algorithms={[name for name, _ in configured_algorithms]}.",
                flush=True,
            )

            partition_records: list[dict[str, Any]] = []
            for job in clustering_jobs:
                algorithm = str(job["algorithm"])
                run_index = int(job["run_index"])
                n_clusters = int(job["n_clusters"])
                seed = int(job["seed"])
                algorithm_config = dict(job.get("algorithm_config", {}) or {})
                partition_id = f"{algorithm}_run{run_index:03d}_K{n_clusters}_S{seed}"
                base_record = {
                    "algorithm": algorithm,
                    "run_index": run_index,
                    "n_clusters": n_clusters,
                    "seed": seed,
                    "partition_id": partition_id,
                    "algorithm_config": algorithm_config,
                }
                try:
                    if algorithm == "hierarchical":
                        linkage = str(algorithm_config["linkage"])
                        labels = AgglomerativeClustering(
                            n_clusters=n_clusters,
                            metric="precomputed",
                            linkage=linkage,
                        ).fit_predict(snf_distance)
                    elif algorithm == "spectral":
                        labels = SpectralClustering(
                            n_clusters=n_clusters,
                            affinity="precomputed",
                            assign_labels=str(algorithm_config["assign_labels"]),
                            random_state=seed,
                        ).fit_predict(z_snf)
                    else:
                        labels = fit_kmedoids(
                            snf_distance,
                            n_clusters=n_clusters,
                            init=str(algorithm_config["init"]),
                            seed=seed,
                        )
                    labels = np.asarray(labels, dtype=int)
                    partition_key = (
                        canonical_partition(labels)
                        if labels.shape == (n_cases,)
                        else tuple()
                    )
                    actual_cluster_count = len(set(partition_key))
                    valid_partition = (
                        labels.shape == (n_cases,)
                        and actual_cluster_count == n_clusters
                    )
                    partition_records.append(
                        {
                            **base_record,
                            "labels": partition_key,
                            "valid": valid_partition,
                            "skip_reason": ""
                            if valid_partition
                            else "cluster_count_mismatch",
                        }
                    )
                except Exception as exc:
                    partition_records.append(
                        {
                            **base_record,
                            "labels": tuple(),
                            "valid": False,
                            "skip_reason": f"{type(exc).__name__}: {exc}",
                        }
                    )

            valid_partition_records = [
                record
                for record in partition_records
                if bool(record.get("valid", True))
            ]
            valid_partitions_by_k: dict[int, list[tuple[int, ...]]] = {}
            for record in valid_partition_records:
                n_clusters = int(record.get("n_clusters", 0) or 0)
                valid_partitions_by_k.setdefault(n_clusters, []).append(
                    tuple(int(item) for item in tuple(record.get("labels", ())))
                )
            print(
                "[candidate_cluster_generator] "
                f"Kept {len(valid_partition_records)} valid candidate partitions "
                f"from {len(clustering_jobs)} attempted partitions.",
                flush=True,
            )

            consensus_records: list[dict[str, Any]] = []
            previous_cdf_area = 0.0
            for n_clusters in n_cluster_values:
                k_partitions = valid_partitions_by_k.get(n_clusters, [])
                if not k_partitions:
                    continue
                consensus = consensus_matrix_from_partitions(k_partitions)
                consensus_values = np.nan_to_num(
                    consensus[np.triu_indices(n_cases, k=1)], nan=0.0
                )
                pac = calculate_pac(consensus_values, lower=pac_lower, upper=pac_upper)
                cdf_thresholds_array, cdf_array, cdf_area = calculate_consensus_cdf(
                    consensus_values
                )
                delta_area, relative_delta_area = calculate_delta_area(
                    cdf_area, previous_cdf_area=previous_cdf_area
                )
                previous_cdf_area = cdf_area
                consensus_distance = 1.0 - consensus
                np.fill_diagonal(consensus_distance, 0.0)
                labels = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    metric="precomputed",
                    linkage=consensus_linkage,
                ).fit_predict(consensus_distance)
                partition_key = canonical_partition(labels)
                cluster_sizes = cluster_sizes_from_labels(partition_key)
                silhouette = consensus_silhouette_score(consensus, partition_key)
                consensus_records.append(
                    {
                        "n_clusters": int(n_clusters),
                        "partition_count": len(k_partitions),
                        "consensus": consensus,
                        "cdf_thresholds": cdf_thresholds_array.tolist(),
                        "cdf_values": cdf_array.astype(float).tolist(),
                        "cdf_area": float(cdf_area),
                        "delta_area": float(delta_area),
                        "relative_delta_area": float(relative_delta_area),
                        "pac": float(pac),
                        "cluster_sizes": cluster_sizes,
                        "consensus_silhouette": silhouette,
                        "labels": partition_key,
                    }
                )

            if consensus_records:
                if selection_method != "llm_consensus_review":
                    raise ValueError(
                        "candidate_clustering.selection_method must be "
                        "'llm_consensus_review'."
                    )

                k_decision = select_k_with_llm(
                    consensus_records,
                    output_root=output_root,
                    config_dir=config_dir,
                    min_cluster_size=min_cluster_size,
                )

                selected_k = int(k_decision["selected_k"])
                best_record = next(
                    record
                    for record in consensus_records
                    if int(record["n_clusters"]) == selected_k
                )
                best_record = dict(best_record)
                best_record["selection_metric"] = "llm_consensus_review"
                best_record["selected_k_reason"] = str(k_decision["reasoning_summary"])
                best_record["llm_confidence"] = str(k_decision["confidence"])
                best_record["llm_evidence_refs"] = list(k_decision["evidence_refs"])
                best_record["k_selection_audit_path"] = str(k_decision["audit_path"])
                best_record["candidate_k_diagnostics"] = k_decision.get("diagnostics", [])

                best_n_clusters = selected_k
                best_delta_area = float(best_record.get("delta_area", 0.0) or 0.0)
                best_cdf_area = float(best_record.get("cdf_area", 0.0) or 0.0)
                best_pac = float(best_record.get("pac", 0.0))
                selection_metric = str(best_record["selection_metric"])
                selected_k_reason = str(best_record.get("selected_k_reason", "") or "")
                best_partition = tuple(
                    int(item) for item in tuple(best_record.get("labels", ()))
                )
                print(
                    "[consensus_clustering] "
                    f"Selected best K by {selection_metric}: K={best_n_clusters}, "
                    f"confidence={k_decision['confidence']}.",
                    flush=True,
                )
                source_algorithms = sorted(
                    {str(record["algorithm"]) for record in valid_partition_records}
                )
                consensus_paths = save_candidate_clustering_outputs(
                    output_root,
                    patient_ids,
                    partition_records,
                    consensus_records,
                    best_record,
                    snf_matrix=z_snf,
                )
                for label in sorted(set(best_partition)):
                    members = [
                        patient_ids[index]
                        for index, value in enumerate(best_partition)
                        if value == label
                    ]
                    if members:
                        candidate_clusters.append(
                            {
                                "cluster_id": f"C{len(candidate_clusters) + 1:04d}",
                                "member_ids": members,
                                "source_views": ["snf"],
                                "status": "under_review",
                                "generator": {
                                    "algorithm": "consensus_hierarchical",
                                    "n_clusters": int(best_n_clusters),
                                    "seed": None,
                                    "partition_id": f"consensus_hierarchical_K{best_n_clusters}",
                                    "cluster_label": int(label),
                                    "selection_metric": selection_metric,
                                },
                                "consensus": {
                                    "partition_count": int(
                                        best_record["partition_count"]
                                    ),
                                    "attempted_partition_count": len(clustering_jobs),
                                    "saved_partition_count": len(partition_records),
                                    "source_algorithms": source_algorithms,
                                    "secondary_algorithm": "hierarchical",
                                    "secondary_n_clusters": int(best_n_clusters),
                                    "best_n_clusters": int(best_n_clusters),
                                    "selection_metric": selection_metric,
                                    "pac": float(best_pac),
                                    "pac_lower": float(pac_lower),
                                    "pac_upper": float(pac_upper),
                                    "cdf_area": float(best_cdf_area),
                                    "delta_area": float(best_delta_area),
                                    "cluster_sizes": list(
                                        best_record.get("cluster_sizes", []) or []
                                    ),
                                    "consensus_silhouette": best_record.get(
                                        "consensus_silhouette"
                                    ),
                                    "selected_k_reason": selected_k_reason,
                                    "llm_confidence": k_decision.get("confidence", ""),
                                    "llm_evidence_refs": k_decision.get(
                                        "evidence_refs", []
                                    ),
                                    "k_selection_audit_path": k_decision.get(
                                        "audit_path", ""
                                    ),
                                    **consensus_paths,
                                },
                            }
                        )
                print(
                    "[consensus_clustering] "
                    f"Generated {len(candidate_clusters)} candidate clusters; "
                    f"sizes={[len(cluster['member_ids']) for cluster in candidate_clusters]}.",
                    flush=True,
                )

    cluster_ids_by_case = {
        str(item.get("case_id", "") or ""): [] for item in patient_states
    }
    for cluster in candidate_clusters:
        cluster_id = str(cluster.get("cluster_id", "") or "")
        for member_id in list(cluster.get("member_ids", []) or []):
            member_key = str(member_id)
            if member_key in cluster_ids_by_case and cluster_id:
                cluster_ids_by_case[member_key].append(cluster_id)
    patient_states = [
        {
            **dict(patient_state),
            "candidate_cluster_ids": cluster_ids_by_case.get(
                str(dict(patient_state).get("case_id", "") or ""), []
            ),
        }
        for patient_state in patient_states
    ]
    return {
        "patient_states": patient_states,
        "patient_states_by_id": {
            str(patient_state["case_id"]): dict(patient_state)
            for patient_state in patient_states
        },
        "candidate_clusters": candidate_clusters,
        "patient_store_paths": save_patient_states(output_root, patient_states),
        "candidate_clusters_path": save_candidate_clusters(
            output_root, candidate_clusters
        ),
        "ct_ccc_filter_path": str(ccc_audit_path),
        "feature_engineering_audit_path": str(feature_audit_path),
        "affinity_patient_order_path": str(affinity_order_path),
        "fused_similarity_path": str(fused_similarity_path),
    }
def fuse_affinities(networks: list[np.ndarray], snf_config: Mapping[str, Any]) -> np.ndarray:
    import snf

    n_cases = len(networks[0])
    fused = snf.snf(
        *networks,
        K=min(max(int(snf_config["neighbor_count"]), 1), n_cases - 1),
        t=int(snf_config["iterations"]),
        alpha=float(snf_config["alpha"]),
    )
    fused = np.maximum((np.asarray(fused) + np.asarray(fused).T) / 2.0, 0.0)
    off_diagonal = ~np.eye(n_cases, dtype=bool)
    maximum = float(fused[off_diagonal].max()) if off_diagonal.any() else 0.0
    if maximum > 1:
        fused[off_diagonal] /= maximum
    np.fill_diagonal(fused, 1.0)
    return fused
