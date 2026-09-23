from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping

from agents.candidate_proposer import candidate_proposer
from agents.evidence_builder import (
    build_evidence_states,
    build_wsi_embeddings_cohort,
    evidence_builder,
)
from agents.inventory import inventory_case
from agents.quality_control import ct_qc, wsi_qc
from agents.subtype_review.runner import run_review_grid
from utils.patient_store import save_patient_states
from utils.io import write_json
from utils.tool_utils import safe_identifier, to_jsonable

DEFAULT_DATA_JSON_PATH = "/data/qijun/path-ct/data/tcga_kirc_data.json"
DEFAULT_OUTPUT_ROOT = "/data/qijun/path-ct/output_kirc"
DEFAULT_CONFIG_DIR = "/data/qijun/path-ct/configs"
CASE_TOOL_OUTPUTS = {
    "ct_qc",
    "ct_tumor_seg",
    "ct_radiomics",
    "wsi_qc",
    "wsi_patch",
    "wsi_tumor_seg",
    "wsi_embeddings",
    "rna",
    "wxs",
}


def sync_case_membership(output_root: str, cases: list[Mapping[str, Any]]) -> None:
    root = Path(output_root)
    manifest_path = root / "storage" / "active_cases.json"
    current_ids = sorted(
        str(case.get("Case_ID", "") or case.get("case_id", "")) for case in cases
    )
    previous_ids = []
    if manifest_path.is_file():
        previous_ids = list(
            json.loads(manifest_path.read_text(encoding="utf-8")).get("case_ids", [])
        )
    for case_id in sorted(set(previous_ids) - set(current_ids)):
        identifier = safe_identifier(case_id)
        for tool_name in CASE_TOOL_OUTPUTS:
            case_dir = root / tool_name / identifier
            case_json = root / tool_name / f"{identifier}.json"
            if case_dir.is_dir():
                shutil.rmtree(case_dir)
            if case_json.is_file():
                case_json.unlink()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"case_ids": current_ids}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_pipeline(
    args: argparse.Namespace, cases: list[Mapping[str, Any]]
) -> dict[str, Any]:
    cases = [dict(item) for item in cases]
    sync_case_membership(args.output_root, cases)
    patient_states = [inventory_case(case_payload) for case_payload in cases]
    patient_states = ct_qc(
        patient_states,
        output_root=args.output_root,
        config_dir=args.config_dir,
    )
    patient_states = wsi_qc(
        patient_states,
        output_root=args.output_root,
        config_dir=args.config_dir,
    )
    save_patient_states(args.output_root, patient_states)
    from utils.llm_utils import load_yaml_file

    radiomics_workers = int(
        load_yaml_file(Path(args.config_dir) / "ct_radiomics.yaml")["num_workers"]
    )
    case_indices = [
        index
        for index, state in enumerate(patient_states)
        if state.get("qc") == "success"
    ]
    if radiomics_workers == 1:
        for index in case_indices:
            patient_states[index] = dict(
                evidence_builder(
                    patient_states[index],
                    output_root=args.output_root,
                    config_dir=args.config_dir,
                )
            )
    else:
        previous_itk_threads = os.environ.get("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS")
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
        try:
            with ProcessPoolExecutor(
                max_workers=radiomics_workers,
                mp_context=multiprocessing.get_context("spawn"),
            ) as executor:
                futures = {
                    index: executor.submit(
                        evidence_builder,
                        patient_states[index],
                        output_root=args.output_root,
                        config_dir=args.config_dir,
                    )
                    for index in case_indices
                }
                for index, future in futures.items():
                    patient_states[index] = dict(future.result())
        finally:
            if previous_itk_threads is None:
                os.environ.pop("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", None)
            else:
                os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = previous_itk_threads
    save_patient_states(args.output_root, patient_states)
    patient_states = build_wsi_embeddings_cohort(
        patient_states,
        output_root=args.output_root,
        config_dir=args.config_dir,
    )
    save_patient_states(args.output_root, patient_states)
    patient_states = build_evidence_states(
        patient_states,
        output_root=args.output_root,
        config_dir=args.config_dir,
    )
    save_patient_states(args.output_root, patient_states)

    candidate_output = candidate_proposer(
        patient_states,
        output_root=args.output_root,
        config_dir=args.config_dir,
    )
    print(
        f"[candidate_cluster_generator] Generated candidate partitions for "
        f"K={','.join(map(str, sorted(candidate_output['candidate_partitions'])))}.",
        flush=True,
    )

    patient_states_by_id = {
        str(case_id): dict(state)
        for case_id, state in candidate_output["patient_states_by_id"].items()
    }
    candidate_patient_ids = {
        str(member_id)
        for partition in candidate_output["candidate_partitions"].values()
        for candidate_set in partition
        for member_id in candidate_set["member_ids"]
    }
    if set(patient_states_by_id) != candidate_patient_ids:
        missing = sorted(candidate_patient_ids - set(patient_states_by_id))
        extra = sorted(set(patient_states_by_id) - candidate_patient_ids)
        raise ValueError(
            "Agent patient states do not match the candidate cohort: "
            f"missing={missing}, extra={extra}"
        )
    review_grid = run_review_grid(
        candidate_output["candidate_partitions"],
        patient_states_by_id,
        str(args.output_root),
        str(args.config_dir),
        tuple(args.initial_ks),
        tuple(args.repeats),
        candidate_output["candidate_signature"],
        force=args.force,
        parallel_runs=getattr(args, "parallel_runs", None),
    )
    print(
        f"[subtype_review] completed={review_grid['complete_run_count']}/"
        f"{review_grid['requested_run_count']}, "
        f"failed={len(review_grid['failed_runs'])}, "
        f"incomplete={len(review_grid['incomplete_runs'])}",
        flush=True,
    )
    result = {
        "agent_run_count": len(review_grid["runs"]),
        "agent_runs_root": review_grid["run_root"],
        "requested_run_count": review_grid["requested_run_count"],
        "complete_run_count": review_grid["complete_run_count"],
        "failed_runs": review_grid["failed_runs"],
        "incomplete_runs": review_grid["incomplete_runs"],
        "input_signature": review_grid["input_signature"],
    }
    write_json(
        Path(args.output_root) / "subtype_review" / "agent_grid_summary.json",
        result,
    )
    write_json(
        Path(args.output_root) / "storage" / "reports" / "final_output.json",
        result,
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Four-view patient-resampled multi-K subtype review pipeline."
    )
    parser.add_argument("--data-json-path", type=str, default=DEFAULT_DATA_JSON_PATH)
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config-dir", type=str, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--initial-k", dest="initial_ks", type=int, action="append")
    parser.add_argument("--repeat", dest="repeats", type=int, action="append")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--parallel-runs", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.config_dir = str(Path(args.config_dir).expanduser().resolve())
    from utils.llm_utils import load_yaml_file

    multi_k = load_yaml_file(Path(args.config_dir) / "subtype_review.yaml")["multi_k"]
    args.initial_ks = sorted(set(args.initial_ks or multi_k["initial_ks"]))
    args.repeats = sorted(set(args.repeats or multi_k["repeats"]))
    case = json.loads(Path(args.data_json_path).read_text(encoding="utf-8"))
    if not isinstance(case, list):
        raise TypeError("data json must contain a JSON list.")
    final_output = run_pipeline(args, case)
    print(json.dumps(to_jsonable(dict(final_output)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
