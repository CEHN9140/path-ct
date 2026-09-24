#!/usr/bin/env python3
"""Compare the same review graph with three high-capability LLM backbones."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.io import write_json
from utils.llm_utils import load_yaml_file
from agents.subtype_review.multi_k import run_multi_k_aggregation
from agents.subtype_review.runner import run_single_review_job

ACTIVE_MODALITIES = ("ct", "wsi", "rna", "wxs")
MODEL_CONFIGS = {
    "gpt56sol": {
        "model_name": "gpt-5.6-sol",
        "base_url": "https://api.chatanywhere.org/v1",
        "api_key_env": "CHATANYWHERE_API_KEY",
        "extra_body": {},
        "reasoning_effort": "none",
        # Previous unified ChatAnywhere settings retained for reference:
        # "base_url": "https://api.chatanywhere.org/v1",
        # "api_key_env": "CHATANYWHERE_API_KEY",
    },
    "deepseek_v4_pro": {
        "model_name": "deepseek-v4-pro",
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "extra_body": {"thinking": {"type": "disabled"}},
        # Previous unified ChatAnywhere settings retained for reference:
        # "base_url": "https://api.chatanywhere.org/v1",
        # "api_key_env": "CHATANYWHERE_API_KEY",
    },
    "qwen38max": {
        "model_name": "qwen3.8-max",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
        "extra_body": {},
        "enable_thinking": False,
        # Previous unified ChatAnywhere settings retained for reference:
        # "base_url": "https://api.chatanywhere.org/v1",
        # "api_key_env": "CHATANYWHERE_API_KEY",
    },
}


def build_model_config(config_dir: Path, model_key: str) -> dict:
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model key: {model_key}")
    config = load_yaml_file(config_dir / "subtype_review.yaml")
    config["llm"] = {
        **config["llm"],
        **MODEL_CONFIGS[model_key],
        "temperature": 0,
    }
    return config


def write_model_config(config: dict, root: Path, source_config_dir: Path) -> Path:
    import yaml

    root.mkdir(parents=True, exist_ok=True)
    config = dict(config)
    config["prompt_dir"] = str(ROOT / "agents" / "subtype_review" / "prompts")
    path = root / "subtype_review.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    shutil.copy2(source_config_dir / "wxs.yaml", root / "wxs.yaml")
    return root


def load_patient_states(patient_states_root: Path) -> dict[str, dict]:
    path = patient_states_root / "storage" / "patient_states" / "patient_states.jsonl"
    states = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                state = json.loads(line)
                states[str(state["case_id"])] = state
    return states


def load_candidate_partition(
    data_root: Path, initial_k: int, patient_ids: list[str]
) -> list[dict]:
    path = (
        data_root / "candidate_subtype" / "consensus_cluster"
        / f"consensus_hierarchical_K{initial_k}.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "candidate_sets" in payload:
        if int(payload.get("initial_k", -1)) != initial_k:
            raise ValueError(f"Saved K={initial_k} partition has a mismatched initial_k.")
        candidate_sets = payload["candidate_sets"]
        if len(candidate_sets) != initial_k:
            raise ValueError(f"Saved K={initial_k} partition has the wrong set count.")
        observed_ids = {
            str(member_id)
            for candidate_set in candidate_sets
            for member_id in candidate_set["member_ids"]
        }
        if observed_ids != set(patient_ids):
            raise ValueError(f"Saved K={initial_k} partition does not match patient order.")
        return [
            {
                **candidate_set,
                "set_id": str(candidate_set["set_id"]),
                "member_ids": [str(member_id) for member_id in candidate_set["member_ids"]],
                "source_views": list(ACTIVE_MODALITIES),
                "status": "under_review",
            }
            for candidate_set in candidate_sets
        ]
    labels = {str(case_id): int(label) for case_id, label in payload["labels"].items()}
    expected_clusters = payload.get("n_clusters", payload.get("initial_k"))
    if set(labels) != set(patient_ids) or int(expected_clusters) != initial_k:
        raise ValueError(f"Saved K={initial_k} partition does not match patient order.")
    return [
        {
            "set_id": f"C{index:04d}",
            "member_ids": [case_id for case_id in patient_ids if labels[case_id] == label],
            "source_views": list(ACTIVE_MODALITIES),
            "status": "under_review",
            "generator": {
                "algorithm": "consensus_hierarchical",
                "n_clusters": initial_k,
                "partition_id": f"consensus_hierarchical_K{initial_k}",
                "cluster_label": label,
            },
        }
        for index, label in enumerate(sorted(set(labels.values())), 1)
    ]


def candidate_signature(partitions: dict[int, list[dict]]) -> str:
    payload = json.dumps(partitions, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_model(
    model_key: str,
    data_root: Path,
    patient_states_root: Path,
    base_config_dir: Path,
    output_root: Path,
    initial_ks: tuple[int, ...],
    repeats: tuple[int, ...],
    parallel_runs: int,
    force: bool,
) -> dict:
    model_root = output_root / model_key
    if force and model_root.exists():
        shutil.rmtree(model_root)
    config = build_model_config(base_config_dir, model_key)
    config_dir = write_model_config(config, model_root / "config", base_config_dir)
    patient_states = load_patient_states(patient_states_root)
    patient_ids = [
        str(item) for item in json.loads(
            (data_root / "candidate_subtype" / "affinity_patient_order.json")
            .read_text(encoding="utf-8")
        )
    ]
    partitions = {
        initial_k: load_candidate_partition(data_root, initial_k, patient_ids)
        for initial_k in initial_ks
    }
    selected_states = {
        case_id: state for case_id, state in patient_states.items()
        if case_id in set(patient_ids) and state.get("qc") == "success"
    }
    if set(selected_states) != set(patient_ids):
        raise ValueError("Frozen patient states do not cover the candidate patient order.")
    model_output = model_root / "subtype_review"
    candidate_output = model_root / "candidate_subtype"
    candidate_output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        data_root / "candidate_subtype" / "affinity_patient_order.json",
        candidate_output / "affinity_patient_order.json",
    )
    jobs = []
    for repeat in repeats:
        for initial_k in initial_ks:
            run_root = model_output / "runs" / f"K{initial_k}" / f"repeat{repeat}"
            if run_root.exists():
                if not force:
                    raise FileExistsError(f"Run directory exists: {run_root}; use --force to rerun it.")
                shutil.rmtree(run_root)
            run_root.mkdir(parents=True)
            write_json(
                run_root / "initial_partition.json",
                {"initial_k": initial_k, "repeat": repeat, "views": list(ACTIVE_MODALITIES),
                 "candidate_sets": partitions[initial_k]},
            )
            jobs.append((initial_k, repeat, partitions[initial_k], str(run_root)))

    input_signature = candidate_signature(partitions)
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(parallel_runs, len(jobs)), mp_context=ctx
    ) as executor:
        futures = {
            executor.submit(
                run_single_review_job, initial_k, repeat, partition,
                patient_states, str(data_root), str(config_dir), run_root,
                input_signature, input_signature,
            ): (initial_k, repeat)
            for initial_k, repeat, partition, run_root in jobs
        }
        results = [future.result() for future in as_completed(futures)]
    results.sort(key=lambda item: (item["repeat"], item["initial_k"]))
    complete_run_count = sum(item["status"] == "complete" for item in results)
    if complete_run_count:
        aggregation = run_multi_k_aggregation(str(model_root), config)
    else:
        aggregation = {
            "status": "pending",
            "complete_run_count": complete_run_count,
            "expected_run_count": len(results),
        }
    result = {
        "output_root": str(model_root),
        "initial_k": list(initial_ks),
        "repeats": list(repeats),
        "parallel_runs": parallel_runs,
        "run_count": len(results),
        "complete_run_count": complete_run_count,
        "failed_runs": [item for item in results if item["status"] == "failed"],
        "multi_k_aggregation": aggregation,
    }
    manifest = {
        "experiment": "llm_backbone_ablation",
        "model_key": model_key,
        "model_name": MODEL_CONFIGS[model_key]["model_name"],
        "active_modalities": list(ACTIVE_MODALITIES),
        "initial_ks": list(initial_ks),
        "repeats": list(repeats),
        "data_root": str(data_root),
        "patient_states_root": str(patient_states_root),
        "result": result,
    }
    write_json(model_root / "ablation_manifest.json", manifest)
    return manifest


def run(
    data_root: Path,
    patient_states_root: Path,
    config_dir: Path,
    output_root: Path,
    model_keys: tuple[str, ...],
    initial_ks: tuple[int, ...],
    repeats: tuple[int, ...],
    parallel_runs: int = 4,
    force: bool = False,
) -> dict:
    if parallel_runs < 1:
        raise ValueError("parallel_runs must be >= 1")
    output_root.mkdir(parents=True, exist_ok=True)
    results = {}
    for key in model_keys:
        results[key] = run_model(
            key,
            data_root,
            patient_states_root,
            config_dir,
            output_root,
            initial_ks,
            repeats,
            parallel_runs,
            force,
        )
    summary = {
        "experiment": "llm_backbone_ablation",
        "models": results,
        "shared_active_modalities": list(ACTIVE_MODALITIES),
        "patient_states_root": str(patient_states_root),
        "initial_ks": list(initial_ks),
        "repeats": list(repeats),
        "parallel_runs": parallel_runs,
        "model_parallelism": 1,
    }
    write_json(output_root / "llm_backbone_ablation_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs")
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "ablation/results/01_llm_backbone"
    )
    parser.add_argument(
        "--model", dest="model_keys", choices=tuple(MODEL_CONFIGS), action="append"
    )
    parser.add_argument(
        "--initial-k", dest="initial_ks", type=int, choices=range(2, 9), action="append"
    )
    parser.add_argument(
        "--repeat", dest="repeats", type=int, choices=(1, 2, 3), action="append"
    )
    parser.add_argument("--parallel-runs", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main_output_root = ROOT / "output_kirc"
    result = run(
        main_output_root,
        main_output_root,
        args.config_dir,
        args.output_root,
        tuple(args.model_keys or MODEL_CONFIGS),
        tuple(sorted(set(args.initial_ks or range(2, 9)))),
        tuple(sorted(set(args.repeats or (1, 2, 3)))),
        args.parallel_runs,
        args.force,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
