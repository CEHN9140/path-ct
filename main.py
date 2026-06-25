from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import traceback
from pathlib import Path
from typing import Any, Mapping, TypedDict

try:
    from langgraph.graph import END, START, StateGraph
except ModuleNotFoundError:
    START = "__start__"
    END = "__end__"

    class SimpleCompiledGraph:
        def __init__(self, nodes, edges, conditional_edges):
            self.nodes = dict(nodes)
            self.edges = dict(edges)
            self.conditional_edges = dict(conditional_edges)

        def invoke(self, state, config=None):
            current = self.edges[START]
            payload = dict(state)
            limit = int(dict(config or {}).get("recursion_limit", 25) or 25)
            for _ in range(limit):
                update = self.nodes[current](payload)
                if isinstance(update, dict):
                    payload.update(update)
                if current in self.conditional_edges:
                    route_func, route_map = self.conditional_edges[current]
                    next_key = route_func(payload)
                    current = route_map[next_key]
                else:
                    current = self.edges.get(current, END)
                if current == END:
                    return payload
            raise RuntimeError("SimpleCompiledGraph recursion limit reached.")

    class StateGraph:
        def __init__(self, state_type):
            self.nodes = {}
            self.edges = {}
            self.conditional_edges = {}

        def add_node(self, name, func):
            self.nodes[name] = func

        def add_edge(self, source, target):
            self.edges[source] = target

        def add_conditional_edges(self, source, route_func, route_map):
            self.conditional_edges[source] = (route_func, dict(route_map))

        def compile(self):
            return SimpleCompiledGraph(self.nodes, self.edges, self.conditional_edges)

from agents.candidate_proposer import candidate_proposer
from agents.evidence_builder import build_evidence_states, evidence_builder
from agents.inventory import inventory_case
from agents.quality_control import run_ct_qc_cohort, run_wsi_qc_cohort
from agents.subtype_review import *
from utils.cluster_flow import REVIEW_UNAVAILABLE_STATUS, build_pipeline_output
from utils.cluster_store import save_cluster_states
from utils.io import ensure_dir, write_json
from utils.patient_store import save_patient_states
from utils.report_store import save_final_output, save_graph_pngs
from utils.subtype_review_runtime import (
    attach_global_figures_to_reports,
    build_review_global_figures,
)
from utils.tool_utils import JSONValue, to_jsonable

DEFAULT_DATA_JSON_PATH = "/data/qijun/path-ct/data/tcga_kirc_data.json"
DEFAULT_OUTPUT_ROOT = "/data/qijun/path-ct/output_kirc"
DEFAULT_CONFIG_DIR = "/data/qijun/path-ct/configs"
EVIDENCE_CHECKPOINT_STAGE = "evidence_ready"
EVIDENCE_CHECKPOINT_CONFIGS = {
    "ct_qc.yaml",
    "ct_radiomics.yaml",
    "ct_tumor_seg.yaml",
    "ct_tumor_seg_lung.yaml",
    "rna.yaml",
    "wsi_embeddings.yaml",
    "wsi_patch.yaml",
    "wsi_qc.yaml",
    "wxs.yaml",
}


def save_review_summary_figures(
    output_root: str, final_review_summary: Mapping[str, Any]
) -> dict[str, str]:
    return {}


class ProjectState(TypedDict, total=False):
    case_id: str
    inventory: dict[str, JSONValue]
    qc: str
    ct_evidence: dict[str, JSONValue]
    wsi_evidence: dict[str, JSONValue]
    text_evidence: dict[str, JSONValue]
    omics_evidence: dict[str, JSONValue]
    candidate_cluster_ids: list[str]


def build_subtype_review_graph() -> Any:
    workflow = StateGraph(ProjectState)
    workflow.add_node("init_subtype_review", init_subtype_review)
    workflow.add_node("compute_verification_vector", compute_verification_vector_node)
    workflow.add_node("verifier", verifier_node)
    workflow.add_node("router_planner", router_planner_node)
    workflow.add_node("revision_engine_node", revision_engine_node)
    workflow.add_edge(START, "init_subtype_review")
    workflow.add_edge("init_subtype_review", "compute_verification_vector")
    workflow.add_edge("compute_verification_vector", "verifier")
    workflow.add_edge("verifier", "router_planner")
    workflow.add_conditional_edges(
        "router_planner",
        route_cluster_action,
        {
            "revision_engine_node": "revision_engine_node",
            "end": END,
        },
    )
    workflow.add_conditional_edges(
        "revision_engine_node",
        route_after_revision,
        {
            "compute_verification_vector": "compute_verification_vector",
            "end": END,
        },
    )
    return workflow.compile()


def pipeline_signature(cases: list[Mapping[str, Any]], config_dir: str) -> str:
    payload: dict[str, Any] = {"cases": to_jsonable(cases), "configs": {}}
    config_path = Path(config_dir).expanduser()
    if config_path.exists():
        payload["configs"] = {
            str(path.relative_to(config_path)): path.read_text(encoding="utf-8")
            for path in sorted(config_path.glob("*.yaml"))
            if path.name in EVIDENCE_CHECKPOINT_CONFIGS
        }
    content = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def load_evidence_checkpoint(
    output_root: str,
    cases: list[Mapping[str, Any]],
    config_dir: str,
) -> list[dict[str, Any]] | None:
    checkpoint_path = (
        Path(output_root)
        / "storage"
        / "pipeline_checkpoints"
        / f"{EVIDENCE_CHECKPOINT_STAGE}.json"
    )
    if not checkpoint_path.exists():
        return None
    try:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if str(checkpoint.get("stage", "") or "") != EVIDENCE_CHECKPOINT_STAGE:
        return None
    if str(checkpoint.get("signature", "") or "") != pipeline_signature(
        cases, config_dir
    ):
        return None
    patient_states = checkpoint.get("patient_states", [])
    if not isinstance(patient_states, list):
        return None
    return [dict(item) for item in patient_states if isinstance(item, Mapping)]


def save_evidence_checkpoint(
    output_root: str,
    cases: list[Mapping[str, Any]],
    config_dir: str,
    patient_states: list[Mapping[str, Any]],
) -> str:
    checkpoint_dir = Path(output_root) / "storage" / "pipeline_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{EVIDENCE_CHECKPOINT_STAGE}.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "stage": EVIDENCE_CHECKPOINT_STAGE,
                "signature": pipeline_signature(cases, config_dir),
                "patient_count": len(patient_states),
                "patient_states": to_jsonable(patient_states),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return str(checkpoint_path)


def subtype_review_agent(
    state: Mapping[str, Any],
    subtype_review_graph: Any,
    output_root: str,
    config_dir: str,
) -> dict[str, Any]:
    inventory = dict(state.get("inventory", {}) or {})
    patient_states_by_id = dict(inventory["patient_states_by_id"])
    candidate_clusters = list(inventory["candidate_clusters"])
    shutil.rmtree(Path(output_root) / "subtype_review", ignore_errors=True)
    review_dir = ensure_dir(Path(output_root) / "subtype_review")
    write_json(
        review_dir / "run_status.json",
        {
            "stage": "subtype_review_started",
            "candidate_cluster_count": len(candidate_clusters),
        },
    )
    cluster_states = []
    absorbed_records = []
    absorbed_cluster_ids = set()
    pending_clusters = [dict(cluster) for cluster in candidate_clusters]
    known_clusters = {
        str(cluster.get("cluster_id", "")): dict(cluster)
        for cluster in pending_clusters
        if str(cluster.get("cluster_id", ""))
    }
    pending_index = 0
    while pending_index < len(pending_clusters):
        cluster = dict(pending_clusters[pending_index])
        pending_index += 1
        cluster_id = str(cluster.get("cluster_id", "unknown_cluster"))
        if cluster_id in absorbed_cluster_ids:
            continue
        try:
            cluster_result = subtype_review_graph.invoke(
                {
                    "case_id": cluster_id,
                    "qc": "success",
                    "inventory": {
                        "cluster": dict(cluster),
                        "patient_states_by_id": patient_states_by_id,
                        "output_root": output_root,
                        "config_dir": config_dir,
                        "all_cluster_states": list(known_clusters.values())
                        + cluster_states,
                    },
                    "ct_evidence": {},
                    "wsi_evidence": {},
                    "text_evidence": {},
                    "omics_evidence": {},
                    "candidate_cluster_ids": [],
                },
                config={"recursion_limit": 25},
            )
        except Exception as exc:
            error_dir = ensure_dir(review_dir / cluster_id)
            write_json(
                error_dir / "error.json",
                {
                    "cluster_id": cluster_id,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            raise
        cluster_inventory = dict(cluster_result.get("inventory", {}) or {})
        patient_states_by_id = {
            str(patient_id): dict(patient_state)
            for patient_id, patient_state in dict(
                cluster_inventory.get("patient_states_by_id", patient_states_by_id)
            ).items()
        }
        cluster_state = dict(cluster_inventory.get("cluster_state", {}) or {})
        if cluster_state:
            generated_clusters = [
                dict(item)
                for item in list(cluster_state.get("generated_clusters", []) or [])
            ]
            if generated_clusters:
                cluster_states.append(cluster_state)
                for item in generated_clusters:
                    generated_id = str(item.get("cluster_id", ""))
                    if generated_id:
                        known_clusters[generated_id] = dict(item)
                        pending_clusters.append(dict(item))
                absorbed_into = str(generated_clusters[0].get("cluster_id", "") or "")
                for absorbed_id in [
                    str(item)
                    for item in list(
                        cluster_state.get("absorbed_cluster_ids", []) or []
                    )
                    if str(item)
                ]:
                    absorbed_cluster_ids.add(absorbed_id)
                    source = dict(
                        known_clusters.get(absorbed_id, {"cluster_id": absorbed_id})
                    )
                    absorbed_records = [
                        item
                        for item in absorbed_records
                        if str(item.get("cluster_id", "")) != absorbed_id
                    ]
                    cluster_states = [
                        item
                        for item in cluster_states
                        if str(item.get("cluster_id", "")) != absorbed_id
                    ]
                    absorbed_records.append(
                        {
                            "cluster_id": absorbed_id,
                            "member_ids": list(source.get("member_ids", []) or []),
                            "parent_cluster_ids": list(
                                source.get("parent_cluster_ids", []) or [absorbed_id]
                            ),
                            "final_action": "merge",
                            "final_decision": "merge",
                            "status": "merge",
                            "absorbed_into": absorbed_into,
                            "absorbed_by": cluster_id,
                            "revision_history": list(
                                cluster_state.get("revision_history", []) or []
                            ),
                        }
                    )
                current_patient_states = [
                    dict(patient_states_by_id[patient_id])
                    for patient_id in sorted(patient_states_by_id)
                ]
                current_cluster_snapshot = cluster_states + [
                    dict(item)
                    for item in pending_clusters[pending_index:]
                    if str(item.get("cluster_id", "") or "")
                    not in absorbed_cluster_ids
                ]
                global_figures = build_review_global_figures(
                    output_root,
                    current_cluster_snapshot,
                    current_patient_states,
                )
                attach_global_figures_to_reports(cluster_states, global_figures)
                continue
            cluster_states.append(cluster_state)
    cluster_states = cluster_states + absorbed_records
    patient_states = [
        dict(patient_states_by_id[patient_id])
        for patient_id in sorted(patient_states_by_id)
    ]
    inventory.update(
        {
            "patient_states": patient_states,
            "patient_states_by_id": patient_states_by_id,
            "patient_store_paths": save_patient_states(output_root, patient_states),
            "cluster_states": cluster_states,
            "cluster_store_paths": save_cluster_states(output_root, cluster_states),
        }
    )
    review_dir = ensure_dir(Path(output_root) / "subtype_review")
    inventory["revised_clusters_path"] = write_json(
        review_dir / "revised_clusters.json",
        [
            {
                "cluster_id": str(item.get("cluster_id", "")),
                "member_ids": list(item.get("member_ids", []) or []),
                "parent_cluster_ids": list(item.get("parent_cluster_ids", []) or []),
                "review_status": str(item.get("status", "") or ""),
                "final_action": str(
                    ""
                    if str(item.get("status", "") or "") == REVIEW_UNAVAILABLE_STATUS
                    else item.get("final_action", "")
                    or item.get("final_decision", "")
                    or ""
                ),
                "drop_reason": str(item.get("drop_reason", "") or ""),
                "confidence_level": str(
                    dict(dict(item.get("report_draft", {}) or {}).get("metadata", {}) or {}).get("confidence_level")
                    or dict(dict(item.get("report_draft", {}) or {}).get("verifier_decision", {}) or {}).get("confidence_level")
                    or dict(item.get("verifier_decision", {}) or {}).get("confidence_level", "")
                    or ""
                ),
                "review_unavailable_reason": str(
                    item.get("review_unavailable_reason", "") or ""
                ),
                "primary_reason_code": (
                    list(
                        dict(item.get("report_draft", {}) or {})
                        .get("verifier_decision", {})
                        .get("reason_codes", [])
                        or ([str(item.get("drop_reason", "") or "")]
                            if str(item.get("drop_reason", "") or "")
                            else [])
                    )[0]
                    if (
                        list(
                            dict(item.get("report_draft", {}) or {})
                            .get("verifier_decision", {})
                            .get("reason_codes", [])
                            or ([str(item.get("drop_reason", "") or "")]
                                if str(item.get("drop_reason", "") or "")
                                else [])
                        )
                    )
                    else ""
                ),
                "reason_codes": list(
                    dict(item.get("report_draft", {}) or {})
                    .get("verifier_decision", {})
                    .get("reason_codes", [])
                    or ([str(item.get("drop_reason", "") or "")]
                        if str(item.get("drop_reason", "") or "")
                        else [])
                ),
                "absorbed_into": str(item.get("absorbed_into", "") or ""),
                "revision_history": list(item.get("revision_history", []) or []),
            }
            for item in cluster_states
        ],
    )
    final_review_summary = build_pipeline_output(
        patient_states=patient_states,
        candidate_clusters=candidate_clusters,
        cluster_states=cluster_states,
        patient_store_paths={},
        cluster_store_paths={},
        graph_paths={},
        output_root=output_root,
    )["final_review_summary"]
    inventory["final_review_summary_path"] = write_json(
        review_dir / "final_review_summary.json",
        final_review_summary,
    )
    inventory["review_summary_figures"] = save_review_summary_figures(
        output_root, final_review_summary
    )
    inventory["cluster_store_paths"]["candidate_clusters"] = inventory[
        "candidate_clusters_path"
    ]
    inventory["cluster_store_paths"]["revised_clusters"] = inventory[
        "revised_clusters_path"
    ]
    inventory["cluster_store_paths"]["final_review_summary"] = inventory[
        "final_review_summary_path"
    ]
    return {"inventory": inventory}


def run_pipeline(args, case):
    subtype_review_graph = build_subtype_review_graph()
    graph_paths = save_graph_pngs(
        args.output_root,
        {"subtype_review_graph": subtype_review_graph},
    )

    cases = [dict(item) for item in case]
    patient_states = load_evidence_checkpoint(args.output_root, cases, args.config_dir)
    if patient_states is not None:
        print(
            "[pipeline_checkpoint] Reuse evidence_ready checkpoint; skip QC and evidence builder.",
            flush=True,
        )
    else:
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
        patient_states_after_evidence = []
        for patient_state in patient_states:
            if patient_state.get("qc") == "success":
                patient_state = evidence_builder(
                    patient_state,
                    output_root=args.output_root,
                    config_dir=args.config_dir,
                )
            patient_states_after_evidence.append(dict(patient_state))
            save_patient_states(args.output_root, patient_states_after_evidence)

        patient_states = patient_states_after_evidence
        patient_states = build_evidence_states(
            patient_states,
            output_root=args.output_root,
            config_dir=args.config_dir,
        )
        save_patient_states(args.output_root, patient_states)
        checkpoint_path = save_evidence_checkpoint(
            args.output_root,
            cases,
            args.config_dir,
            patient_states,
        )
        print(
            f"[pipeline_checkpoint] Saved evidence_ready checkpoint: {checkpoint_path}",
            flush=True,
        )
    inventory = {
        "cases": cases,
        "graph_paths": graph_paths,
        **candidate_proposer(
            patient_states,
            output_root=args.output_root,
            config_dir=args.config_dir,
        ),
    }
    result = subtype_review_agent(
        {"inventory": inventory},
        subtype_review_graph,
        args.output_root,
        args.config_dir,
    )
    inventory = dict(result.get("inventory", {}) or inventory)
    final_output = build_pipeline_output(
        patient_states=list(inventory["patient_states"]),
        candidate_clusters=list(inventory["candidate_clusters"]),
        cluster_states=list(inventory.get("cluster_states", []) or []),
        patient_store_paths=dict(inventory["patient_store_paths"]),
        cluster_store_paths=dict(inventory["cluster_store_paths"]),
        graph_paths=graph_paths,
        output_root=args.output_root,
    )
    save_final_output(args.output_root, final_output)
    return final_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Logic-V-Sub V2 coordinator with fixed workflow and subtype review agent."
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
