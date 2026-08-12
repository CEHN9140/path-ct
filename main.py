from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

from agents.candidate_proposer import candidate_proposer
from agents.evidence_builder import (
    build_ct_tumor_seg_cohort,
    build_evidence_states,
    build_wsi_embeddings_cohort,
    build_wsi_tumor_seg_cohort,
    evidence_builder,
)
from agents.inventory import inventory_case
from agents.quality_control import run_ct_qc_cohort, run_wsi_qc_cohort
from agents.subtype_review.graph import build_review_graph, initial_review_state, save_review_outputs
from agents.subtype_review.llm import build_default_reviser, build_default_router, build_default_verifier
from agents.subtype_review.tools import load_available_tool_functions, normalize_tool_definitions
from utils.patient_store import save_patient_states
from utils.report_store import save_final_output
from utils.llm_utils import load_yaml_file
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
    "cnv",
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
    patient_states = run_ct_qc_cohort(
        patient_states,
        output_root=args.output_root,
        config_dir=args.config_dir,
    )
    patient_states = run_wsi_qc_cohort(
        patient_states,
        output_root=args.output_root,
        config_dir=args.config_dir,
    )
    patient_states = build_ct_tumor_seg_cohort(
        patient_states,
        output_root=args.output_root,
        config_dir=args.config_dir,
    )
    save_patient_states(args.output_root, patient_states)
    for index, patient_state in enumerate(patient_states):
        if patient_state.get("qc") == "success":
            patient_state = evidence_builder(
                patient_state,
                output_root=args.output_root,
                config_dir=args.config_dir,
            )
        patient_states[index] = dict(patient_state)
    save_patient_states(args.output_root, patient_states)
    patient_states = build_wsi_tumor_seg_cohort(
        patient_states,
        output_root=args.output_root,
        config_dir=args.config_dir,
    )
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
        f"[candidate_cluster_generator] Generated "
        f"{len(candidate_output.get('candidate_clusters', []))} candidate clusters.",
        flush=True,
    )

    review_config = load_yaml_file(Path(args.config_dir) / "subtype_review.yaml")
    tool_definitions = normalize_tool_definitions(review_config.get("tools", {}))
    tool_functions, tool_load_errors = load_available_tool_functions(tool_definitions)
    review_patients = list(candidate_output.get("patient_states", patient_states))
    patient_states_by_id = {
        str(item.get("case_id", item.get("Case_ID", ""))): dict(item)
        for item in review_patients
        if str(item.get("case_id", item.get("Case_ID", "")))
    }
    review_runtime = {
        "patient_states_by_id": patient_states_by_id,
        "output_root": str(args.output_root),
        "config_dir": str(args.config_dir),
        "tool_functions": tool_functions,
        "tool_load_errors": tool_load_errors,
    }
    verifier_model = build_default_verifier(review_config, args.config_dir)
    reviser_model = build_default_reviser(review_config, args.config_dir)
    router_model = build_default_router(review_config, args.config_dir, lambda *_args, **_kwargs: {})
    review_graph = build_review_graph(
        verifier_model=verifier_model,
        router_model=router_model,
        reviser_model=reviser_model,
        tool_functions=tool_functions,
        runtime=review_runtime,
    )
    review_state = initial_review_state(candidate_output.get("candidate_clusters", []))
    review_state["control"]["max_rounds"] = int(
        dict(review_config.get("budget", {}) or {}).get("max_rounds", 12) or 12
    )
    review_state["control"]["max_failures"] = int(
        dict(review_config.get("budget", {}) or {}).get("max_failures", 3) or 3
    )
    final_state = review_graph.invoke(
        review_state,
        context=review_runtime,
        config={"recursion_limit": int(review_state["control"]["max_rounds"]) * 3 + 10},
    )
    final_review_summary = save_review_outputs(final_state, args.output_root)
    save_final_output(args.output_root, {"final_review_summary": final_review_summary})
    return {"final_review_summary": final_review_summary}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Logic-V-Sub V3 pipeline with LLM K-selection and global subtype review."
    )
    parser.add_argument("--data-json-path", type=str, default=DEFAULT_DATA_JSON_PATH)
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config-dir", type=str, default=DEFAULT_CONFIG_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.config_dir = str(Path(args.config_dir).expanduser().resolve())
    case = json.loads(Path(args.data_json_path).read_text(encoding="utf-8"))
    if not isinstance(case, list):
        raise TypeError("data json must contain a JSON list.")
    final_output = run_pipeline(args, case)
    print(json.dumps(to_jsonable(dict(final_output)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
