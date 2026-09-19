from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import ToolMessage

from agents.subtype_review.llm import parse_json_content
from agents.subtype_review.schemas import EvidenceReportBatch, EvidenceRequest, RouterAction, RouterPlan, set_id
from agents.subtype_review.state import append_trace, context_values, current_sets, partition_artifact_id, partition_signature, registry
from agents.subtype_review.tools import compact_tool_result


def report_tool_refs(rows: list[Mapping[str, Any]]) -> dict[tuple[str, str, str], list[str]]:
    """Index tool provenance by dimension, scope and individual report target."""
    grouped: dict[tuple[str, str, str], set[str]] = {}
    for row in rows:
        targets = [""] if row.get("scope") == "partition" else row.get("target_ids", []) or []
        for target in targets:
            grouped.setdefault(
                (str(row["dimension"]), str(row["scope"]), str(target)), set()
            ).add(str(row["tool_name"]))
    return {key: sorted(value) for key, value in grouped.items()}


def required_reports_for_round(
    state: Mapping[str, Any], runtime: Mapping[str, Any]
) -> list[dict[str, Any]]:
    return [
        {
            "dimension": dimension,
            "scope": scope,
            "target_ids": [] if not target else [target],
            "tool_names": tool_names,
        }
        for (dimension, scope, target), tool_names in sorted(
            report_tool_refs(state.get("round_evidence", []) or []).items()
        )
    ]


def expected_tool_refs_for_round(
    state: Mapping[str, Any], runtime: Mapping[str, Any]
) -> dict[tuple[str, str, str], list[str]]:
    return report_tool_refs([
        row for row in current_partition_evidence(state)
        if row.get("status") in {"success", "scientific_unavailable"}
    ])


def eligible_tools_for_requests(
    state: Mapping[str, Any], runtime: Mapping[str, Any], requests: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    values = context_values(runtime)
    tools = registry(values)
    signature = partition_signature(current_sets(state))
    completed = completed_tool_keys(state)
    eligible = {}
    targets = [set_id(item) for item in current_sets(state)]
    for name, metadata in sorted(tools.items()):
        if not metadata.get("verifier_selectable"):
            continue
        matching = [item for item in requests if item["dimension"] == metadata["dimension"]]
        if not matching:
            continue
        if metadata["scope"] == "partition":
            if not any(not item["target_ids"] for item in matching) or (name, signature, "") in completed:
                continue
            allowed = set()
        else:
            allowed = {
                target
                for item in matching
                for target in item["target_ids"]
                if target in targets and (name, signature, target) not in completed
            }
            if not allowed:
                continue
        eligible[name] = {
            "dimension": metadata["dimension"],
            "scope": metadata["scope"],
            "target_ids": sorted(allowed),
            "description": metadata["description"],
        }
    return eligible


def completed_tool_keys(state: Mapping[str, Any]) -> set[tuple[str, str, str]]:
    return {
        (str(row.get("tool_name", "")), str(row["partition_signature"]), str(target))
        for row in current_partition_evidence(state)
        if row.get("status") in {"success", "scientific_unavailable"}
        for target in row.get("target_ids", []) or [""]
    }


def current_partition_evidence(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    signature = partition_signature(current_sets(state))
    rows = [
        row
        for entry in state.get("history", []) or []
        for row in entry.get("round_evidence", []) or []
    ]
    rows.extend(state.get("round_evidence", []) or [])
    return [row for row in rows if row.get("partition_signature") == signature]


def available_evidence_requests(
    state: Mapping[str, Any], runtime: Mapping[str, Any]
) -> list[dict[str, Any]]:
    tools = registry(runtime)
    signature = partition_signature(current_sets(state))
    completed = completed_tool_keys(state)
    targets = [set_id(item) for item in current_sets(state)]
    requests = {}
    for name, metadata in sorted(tools.items()):
        if not metadata.get("verifier_selectable"):
            continue
        for target in [""] if metadata["scope"] == "partition" else targets:
            if (name, signature, target) in completed:
                continue
            question = f"Clarify the current {metadata['dimension']} evidence"
            requests[(metadata["dimension"], target)] = {
                "dimension": metadata["dimension"],
                "target_ids": [target] if target else [],
                "question": f"{question} for {target}." if target else f"{question}.",
            }
    return [requests[key] for key in sorted(requests)]


def normalize_tool_call(call: Any) -> dict[str, Any]:
    name = str(call.get("name", "") if isinstance(call, Mapping) else getattr(call, "name", ""))
    args = call.get("args", {}) if isinstance(call, Mapping) else getattr(call, "args", {})
    return {
        "name": name,
        "args": parse_json_content(args) if isinstance(args, str) else dict(args or {}),
        "id": str(call.get("id", name) if isinstance(call, Mapping) else getattr(call, "id", name)),
    }


def validate_selected_tool_coverage(
    calls: list[Any],
    requests: list[dict[str, Any]],
    registry: Mapping[str, Mapping[str, Any]],
) -> None:
    selected = [normalize_tool_call(call) for call in calls]
    for request in requests:
        targets = set(request["target_ids"])
        covered = set()
        partition_covered = False
        for call in selected:
            name, args = call["name"], call["args"]
            metadata = registry.get(name, {})
            if metadata.get("dimension") != request["dimension"]:
                continue
            call_targets = {str(target) for target in args.get("target_ids", []) or []}
            if not targets:
                if metadata.get("scope") == "partition" and not call_targets:
                    partition_covered = True
            elif metadata.get("scope") == "set_identity":
                covered.update(args.get("target_ids", []) or [])
        if not targets:
            if not partition_covered:
                raise ValueError(
                    "Verifier tool calls do not cover partition EvidenceRequest "
                    f"{request['dimension']}"
                )
            continue
        if not targets.issubset(covered):
            raise ValueError(
                "Verifier tool calls do not cover EvidenceRequest "
                f"{request['dimension']}:{sorted(targets)}"
            )


def execute_tool_calls(
    state: dict[str, Any], message: Any, runtime: Mapping[str, Any]
) -> None:
    raw_calls = message.get("tool_calls", []) if isinstance(message, Mapping) else getattr(message, "tool_calls", [])
    calls = [normalize_tool_call(call) for call in raw_calls or []]
    names = [call["name"] for call in calls]
    if len(names) != len(set(names)):
        raise ValueError("Verifier cannot call the same registered tool twice")
    sets = current_sets(state)
    signature = partition_signature(sets)
    tools = registry(runtime)
    eligible = dict(state["control"].get("eligible_tools", {}) or {})
    requests = state["control"].get("pending_evidence_requests", []) or []
    cluster_state = {
        "cluster_id": partition_artifact_id(signature),
        "member_ids": sorted(member for item in sets for member in item["member_ids"]),
        "active_modalities": tuple(runtime.get("active_modalities") or ("ct", "wsi", "rna", "wxs", "cnv")),
    }
    results = []
    messages = [*state.get("messages", []), message]
    validate_selected_tool_coverage(calls, requests, tools)
    known = {set_id(item) for item in sets}
    completed = completed_tool_keys(state)
    for call in calls:
        name, args = call["name"], call["args"]
        if name not in tools:
            raise ValueError(f"Verifier called an unregistered tool: {name}")
        metadata = tools[name]
        if not metadata.get("verifier_selectable"):
            raise ValueError(f"Tool {name} is not selectable by Verifier")
        if name not in eligible:
            raise ValueError(f"Verifier called tool not listed in eligible_tools: {name}")
        target_ids = sorted({str(target) for target in args.get("target_ids", []) or [] if str(target)})
        if not set(target_ids).issubset(known):
            raise ValueError(f"Verifier tool {name} targeted a non-current set")
        if metadata["scope"] == "partition" and target_ids:
            raise ValueError(f"Partition tool {name} cannot target sets")
        if metadata["scope"] == "set_identity" and not target_ids:
            raise ValueError(f"Set tool {name} requires target_ids")
        available_targets = set(eligible[name].get("target_ids", []) or [])
        if metadata["scope"] == "set_identity" and not set(target_ids).issubset(available_targets):
            raise ValueError(f"Verifier tool {name} targeted an ineligible set")
        matching = [request for request in requests if request["dimension"] == metadata["dimension"]]
        if metadata["scope"] == "partition":
            if not any(not request["target_ids"] for request in matching):
                raise ValueError(f"Verifier tool {name} does not answer a pending EvidenceRequest")
        else:
            pending_targets = {
                target
                for request in matching
                for target in request["target_ids"]
            }
            if not set(target_ids).issubset(pending_targets):
                raise ValueError(f"Verifier tool {name} does not answer the selected EvidenceRequest")
        if metadata["scope"] == "partition":
            if (name, signature, "") in completed:
                raise ValueError(f"Verifier repeated completed tool {name}")
        elif any((name, signature, target) in completed for target in target_ids):
            raise ValueError(f"Verifier repeated completed tool {name} for target")
        raw = metadata["function"](
            cluster_state,
            dict(runtime.get("patient_states_by_id", {}) or {}),
            str(runtime.get("data_root", "")),
            config_dir=str(runtime.get("config_dir", "")),
            all_cluster_states=sets,
            scope=str(metadata["scope"]),
            target_ids=target_ids,
            artifact_root=str(runtime.get("artifact_root", runtime.get("data_root", ""))),
        )
        compact = compact_tool_result(raw, name)
        if compact["status"] == "runtime_failure":
            raise RuntimeError(
                f"Scientific tool {name} failed at runtime: "
                f"{'; '.join(compact.get('errors', []))}"
            )
        compact.update({
            "dimension": metadata["dimension"],
            "scope": metadata["scope"],
            "target_ids": target_ids,
            "partition_signature": signature,
        })
        results.append(compact)
        tool_message_payload = {
            "tool_name": name,
            "dimension": metadata["dimension"],
            "scope": metadata["scope"],
            "target_ids": target_ids,
            "status": compact["status"],
            "metrics": compact["metrics"],
            "warnings": list(compact.get("warnings", []) or []),
            "missing_reason": compact.get("missing_reason", ""),
            "errors": list(compact.get("errors", []) or []),
        }
        messages.append(ToolMessage(
            content=json.dumps(tool_message_payload, ensure_ascii=False),
            tool_call_id=call["id"],
        ))
    state["round_evidence"] = results
    state["messages"] = messages
    control = dict(state["control"])
    control["next"] = "verifier_audit"
    state["control"] = control
    append_trace(state, {"node": "verifier", "event": "tool_selection", "tools": names})


def raw_metric_refs(rows: list[Mapping[str, Any]]) -> set[str]:
    return {
        str(ref)
        for row in rows
        for ref in row.get("metric_refs", []) or []
    }


def available_revision_metric_refs(
    state: Mapping[str, Any], router_plan: Mapping[str, Any] | RouterPlan
) -> list[str]:
    plan = router_plan.model_dump() if hasattr(router_plan, "model_dump") else router_plan
    refs = raw_metric_refs(current_partition_evidence(state))
    prefix = "tool_results.multimodal_consistency_check.metrics.cross_modal_consistency.per_set."
    selected = set()
    for action in plan.get("actions", []) or []:
        targets = [str(target) for target in action.get("target_ids", []) or []]
        if action.get("action") == "split" and targets:
            path = f"{prefix}{targets[0]}.internal_structure."
            selected.update(ref for ref in refs if ref.startswith(path))
        elif action.get("action") == "merge" and len(targets) == 2:
            left, right = sorted(targets)
            paths = (
                f"{prefix}{left}.boundary_to_other_sets.{right}.",
                f"{prefix}{right}.boundary_to_other_sets.{left}.",
            )
            selected.update(ref for ref in refs if ref.startswith(paths))
    return sorted(selected)


def metric_refs_for_report(
    report: Any, state: Mapping[str, Any]
) -> list[str]:
    rows = {
        str(row.get("tool_name", "")): row
        for row in current_partition_evidence(state)
    }
    target = next(iter(report.target_ids), "")
    refs = []
    for tool_name in report.tool_refs:
        row = rows.get(tool_name, {})
        metrics = row.get("metrics", {}) or {}
        leaves = [str(ref) for ref in row.get("metric_refs", []) or []]
        base = f"tool_results.{tool_name}.metrics"
        if report.scope == "partition":
            refs.extend(leaves)
            continue
        prefixes = []
        for key, value in metrics.items():
            if not isinstance(value, Mapping):
                continue
            if key == "cross_modal_consistency":
                if target in value.get("per_set", {}):
                    prefixes.append(f"{base}.{key}.per_set.{target}")
                if "partition" in value:
                    prefixes.append(f"{base}.{key}.partition")
            elif target in value:
                prefixes.append(f"{base}.{key}.{target}")
            elif f"{target}_vs_rest" in value:
                prefixes.append(f"{base}.{key}.{target}_vs_rest")
        refs.extend(
            ref for ref in leaves
            if any(ref.startswith(f"{prefix}.") or ref.startswith(f"{prefix}[") for prefix in prefixes)
        )
    return sorted(set(refs))


def validate_reports(
    batch: EvidenceReportBatch, state: Mapping[str, Any], runtime: Mapping[str, Any]
) -> None:
    sets = {set_id(item) for item in current_sets(state)}
    expected = set(report_tool_refs(state.get("round_evidence", []) or []))
    prior_keys = {
        (str(report["dimension"]), str(report["scope"]), next(iter(report.get("target_ids", [])), ""))
        for report in state.get("reports", []) or []
    }
    allowed_keys = expected | prior_keys
    expected_refs = expected_tool_refs_for_round(state, runtime)
    reported = set()
    for report in batch.reports:
        targets = set(report.target_ids)
        if targets - sets:
            raise ValueError("Evidence Report references a non-current set")
        if report.scope == "set_identity" and len(targets) != 1:
            raise ValueError("set Evidence Reports must target exactly one current set")
        key = (
            report.dimension,
            report.scope,
            next(iter(targets)) if targets else "",
        )
        if key in reported:
            raise ValueError("Duplicate Evidence Report")
        if key not in allowed_keys:
            raise ValueError("Evidence Report does not match current evidence")
        if not report.tool_refs:
            raise ValueError("Evidence Report must cite tool_refs")
        allowed = set(expected_refs.get(key, []))
        if set(report.tool_refs) != allowed:
            raise ValueError(
                f"Evidence Report tool_refs mismatch for {key}: "
                f"expected={sorted(allowed)} got={sorted(report.tool_refs)}"
            )
        reported.add(key)
    if not expected.issubset(reported):
        raise ValueError(f"Evidence Reports must cover exactly current targets: {expected - reported}")
    for report in batch.reports:
        report.metric_refs = metric_refs_for_report(report, state)


def validate_evidence_request(
    request: EvidenceRequest, state: Mapping[str, Any], runtime: Mapping[str, Any]
) -> None:
    available = available_evidence_requests(state, runtime)
    same_dimension = [
        item for item in available if item["dimension"] == request.dimension
    ]
    targets = set(request.target_ids)
    if not targets:
        valid = any(not item["target_ids"] for item in same_dimension)
    else:
        available_targets = {
            target
            for item in same_dimension
            for target in item["target_ids"]
        }
        valid = targets.issubset(available_targets)
    if not valid:
        raise ValueError(
            "EvidenceRequest is not currently available "
            "for the requested dimension/targets"
        )


def evidence_coverage(state: Mapping[str, Any], runtime: Mapping[str, Any] | None = None) -> dict[str, Any]:
    scope_dimensions = {"set_identity": set(), "partition": set()}
    for metadata in registry(context_values(runtime or {})).values():
        if metadata.get("verifier_selectable"):
            scope_dimensions[metadata["scope"]].add(metadata["dimension"])
    coverage = {
        "set_identity": {
            set_id(item): {
                dimension: "unassessed"
                for dimension in sorted(scope_dimensions["set_identity"])
            }
            for item in current_sets(state)
        },
        "partition": {
            dimension: "unassessed"
            for dimension in sorted(scope_dimensions["partition"])
        },
    }
    for report in state.get("reports", []) or []:
        dimension = str(report.get("dimension", ""))
        if report.get("scope") == "partition":
            if dimension in coverage["partition"]:
                coverage["partition"][dimension] = "assessed"
        else:
            for target in report.get("target_ids", []) or []:
                target = str(target)
                if dimension in coverage["set_identity"].get(target, {}):
                    coverage["set_identity"][target][dimension] = "assessed"
    return coverage


def validate_decision_state_evidence(
    action: RouterAction, state: Mapping[str, Any], runtime: Mapping[str, Any]
) -> None:
    coverage = evidence_coverage(state, runtime)
    for target in action.target_ids:
        current = coverage["set_identity"].get(target, {})
        if action.decision_state.identity != "unassessed" and current.get("biological_support") != "assessed":
            raise ValueError("identity assessment requires biological_support evidence")
        if action.decision_state.structure != "unassessed" and current.get("cross_modal_consistency") != "assessed":
            raise ValueError("structure assessment requires cross_modal_consistency evidence")
        if (
            action.decision_state.alternative_explanation != "unassessed"
            and current.get("confounder_exclusion") != "assessed"
        ):
            raise ValueError("alternative-explanation assessment requires confounder evidence")


def structural_evidence_available(
    state: Mapping[str, Any], targets: list[str], pair: bool = False
) -> bool:
    target_set = set(targets)
    for row in current_partition_evidence(state):
        if row.get("tool_name") != "multimodal_consistency_check" or row.get("status") != "success":
            continue
        structural = dict(row.get("full_metrics", {}) or {}).get("structural_characterization", {}) or {}
        if pair:
            for item in dict(structural.get("boundary_by_pair", {}) or {}).values():
                if set(item.get("targets", []) or []) == target_set:
                    return True
        elif any(
            target in dict(structural.get("internal_structure_by_set", {}) or {})
            for target in target_set
        ):
            return True
    return False


def validate_action_contract(
    action: RouterAction, state: Mapping[str, Any], runtime: Mapping[str, Any]
) -> None:
    known = {set_id(item) for item in current_sets(state)}
    if not set(action.target_ids).issubset(known):
        raise ValueError("Router referenced a non-current set")
    validate_decision_state_evidence(action, state, runtime)
    if action.action == "split":
        if action.decision_state.structure != "incompatible":
            raise ValueError("split requires structure=incompatible")
        if not structural_evidence_available(state, action.target_ids):
            raise ValueError("split requires assessed structural evidence")
    if action.action == "merge":
        if action.decision_state.structure != "incompatible":
            raise ValueError("merge requires structure=incompatible")
        if not structural_evidence_available(state, action.target_ids, pair=True):
            raise ValueError("merge requires assessed pairwise structural evidence")


def validate_router_plan(
    plan: RouterPlan, state: Mapping[str, Any], runtime: Mapping[str, Any]
) -> None:
    if plan.evidence_requests:
        seen = set()
        for request in plan.evidence_requests:
            validate_evidence_request(request, state, runtime)
            targets = request.target_ids or ["__partition__"]
            for target in targets:
                key = (request.dimension, target)
                if key in seen:
                    raise ValueError("Duplicate or overlapping EvidenceRequest coverage")
                seen.add(key)
        return
    known = {set_id(item) for item in current_sets(state)}
    occupied = set()
    for action in plan.actions:
        targets = set(action.target_ids)
        if occupied.intersection(targets):
            raise ValueError("Current sets cannot occur in multiple Router actions")
        occupied.update(targets)
    if occupied != known:
        raise ValueError("Router plan must cover every current set exactly once")
    for action in plan.actions:
        validate_action_contract(action, state, runtime)


def structural_metrics(state: Mapping[str, Any]) -> dict[str, Any]:
    row = next(
        (
            item for item in state.get("round_evidence", []) or []
            if item.get("tool_name") == "multimodal_consistency_check"
        ),
        {},
    )
    if not row:
        row = next(
            (item for item in reversed(current_partition_evidence(state))
             if item.get("tool_name") == "multimodal_consistency_check"),
            {},
        )
    return dict(dict(row.get("full_metrics", {}) or {}).get("structural_characterization", {}) or {})


def revision_metrics(state: Mapping[str, Any]) -> dict[str, Any]:
    return {"partition": state["partition"], "structural_characterization": structural_metrics(state)}


def compact_structural_index(state: Mapping[str, Any]) -> dict[str, Any]:
    structural = structural_metrics(state)
    per_set = {}
    for item in current_sets(state):
        target = set_id(item)
        metrics = dict(structural.get("internal_structure_by_set", {}).get(target, {}) or {})
        probe = dict(metrics.get("fused_binary_probe", {}) or {})
        resampling = dict(probe.get("resampling", {}) or {})
        modality_probe = dict(metrics.get("probe_support_by_modality", {}) or {})
        per_set[target] = {
            "member_n": len(item.get("member_ids", []) or []),
            "binary_probe": {
                "child_sizes": probe.get("child_sizes"),
                "fused_silhouette": probe.get("median_silhouette"),
                "normalized_cut": probe.get("normalized_cut"),
                "median_ari": resampling.get("median_resample_ari"),
                "consensus_separation": resampling.get("consensus_separation"),
                "pac": resampling.get("pac"),
                "degenerate_fraction": resampling.get("degenerate_resample_fraction"),
                "modality_probe_silhouettes": {
                    name: dict(modality_probe.get(name, {}) or {}).get("median_silhouette")
                    for name in ("ct", "wsi", "rna", "wxs", "cnv")
                },
            },
        }
    boundaries = []
    for pair_key, pair_metrics in sorted(
        dict(structural.get("boundary_by_pair", {}) or {}).items()
    ):
        pair = dict(pair_metrics or {})
        targets = list(pair.get("targets", []) or [])
        if len(targets) != 2:
            targets = str(pair_key).split("+", 1)
        fused = dict(pair.get("fused", {}) or {})
        boundaries.append({
            "pair": sorted(str(target) for target in targets),
            "fused": {
                key: fused.get(key)
                for key in (
                    "pair_median_silhouette", "left_median_margin", "right_median_margin",
                    "left_boundary_separation", "right_boundary_separation",
                )
            },
            "modalities": {
                name: {
                    key: dict(pair.get("modalities", {}).get(name, {}) or {}).get(key)
                    for key in (
                        "pair_median_silhouette", "left_median_margin", "right_median_margin",
                        "left_boundary_separation", "right_boundary_separation",
                    )
                }
                for name in ("ct", "wsi", "rna", "wxs", "cnv")
            },
        })
    return {
        "per_set": per_set,
        "boundary_by_pair": boundaries,
        "limitations": list(structural.get("limitations", []) or []),
    }
