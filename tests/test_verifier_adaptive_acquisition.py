from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agents.subtype_review.graph import (
    evidence_request_ref,
    evidence_report_ref,
    initial_review_state,
    latest_acquisition_context,
    partition_signature,
    reports_for_request,
    router_node,
    terminal_closure_refs,
    validate_router_plan,
    verifier_node,
)
from agents.subtype_review.llm import VerifierChatModel
from agents.subtype_review.schemas import EvidenceReport, EvidenceRequest, RouterAction, RouterPlan
from agents.subtype_review.tools import TOOL_REGISTRY, build_selection_tools


def make_registry(*names):
    registry = {}
    for name, dimension, scopes in names:
        registry[name] = {
            "aspect": name,
            "dimension": dimension,
            "scopes": scopes,
            "description": f"Evidence from {name}.",
            "selection_guidance": f"Use {name} for its relevant question.",
            "function": lambda **kwargs: {
                "status": "success",
                "metrics": {kwargs["scope"]: {
                    target: {"value": 1} for target in kwargs["target_ids"]
                } or {"partition": {"value": 1}}},
            },
        }
    return registry


def context(registry, model, **extra):
    return {
        "tool_registry": registry,
        "patient_states_by_id": {},
        "data_root": "/tmp",
        "config_dir": "/tmp",
        "verifier_model": model,
        **extra,
    }


def valid_reports(payload):
    return {"reports": [
        {
            "dimension": item["dimension"],
            "aspect": item["aspect"],
            "scope": item["scope"],
            "target_ids": item["target_ids"],
            "observations": [],
            "dimension_interpretation": "The measured evidence describes this request.",
            "cross_evidence_context": "No relevant prior report is available.",
            "limitations": [],
            "tool_refs": [],
            "metric_refs": [],
        }
        for item in payload["required_reports"]
    ]}


def request_state(requests, sets=(("C1", ["P1", "P2"]), ("C2", ["P3", "P4"]))):
    state = initial_review_state([
        {"set_id": name, "member_ids": members} for name, members in sets
    ])
    state["control"]["pending_evidence_requests"] = requests
    return state


def test_verifier_can_stop_after_first_evidence_report_and_selector_sees_report_only(tmp_path):
    registry = make_registry(
        ("rna", "biological_support", ("set",)),
        ("wxs", "biological_support", ("set",)),
    )
    payloads = []
    executed = []
    for name in registry:
        registry[name]["function"] = lambda _name=name, **kwargs: (
            executed.append(_name) or {"status": "success", "metrics": {"set": {"C1": {"signal": _name}}}}
        )
    request = {"dimension": "biological_support", "scope": "set", "target_ids": ["C1"], "question": "Is the phenotype coherent?"}

    class Verifier:
        def invoke(self, payload):
            payloads.append(payload)
            if payload["mode"] == "select":
                if payload["require_tool"]:
                    return {"selected_tool": "rna"}
                assert payload["current_evidence"]
                assert "round_evidence" not in payload
                assert "metric_refs" not in str(payload)
                assert "signal" not in str(payload)
                return {"selected_tool": None, "stop_reason": "verifier_no_further_tool_call"}
            return valid_reports(payload)

    state = request_state([request])
    trace_path = tmp_path / "runtime_trace.jsonl"
    result = verifier_node(state, context(registry, Verifier(), runtime_trace_path=str(trace_path)))
    selections = [item for item in payloads if item["mode"] == "select"]
    assert [item["require_tool"] for item in selections] == [True, False]
    assert selections[1]["current_evidence"][0]["aspect"] == "rna"
    assert selections[1]["remaining_tools"] == ["wxs"]
    assert executed == ["rna"]
    assert len([item for item in payloads if item["mode"] == "audit"]) == 1
    assert len(result["round_evidence"]) == 1
    assert result["control"]["pending_evidence_requests"] == []
    stopped = [
        json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["event"] == "verifier_request_stopped"
    ]
    assert len(stopped) == 1
    assert stopped[0]["stop_reason"] == "verifier_no_further_tool_call"
    assert "reports" not in json.dumps(stopped[0])
    assert "NES" not in json.dumps(stopped[0])
    assert "EvidenceReport" not in json.dumps(stopped[0])


def test_wave_selection_uses_same_report_snapshot_and_requests_stop_independently():
    registry = make_registry(
        ("rna", "biological_support", ("set",)),
        ("wxs", "biological_support", ("set",)),
    )
    timeline = []
    selection_calls = []
    for name in registry:
        registry[name]["function"] = lambda _name=name, **kwargs: (
            timeline.append(("execute", _name, kwargs["target_ids"][0]))
            or {"status": "success", "metrics": {"set": {kwargs["target_ids"][0]: {"signal": _name}}}}
        )
    requests = [
        {"dimension": "biological_support", "scope": "set", "target_ids": [target], "question": "Is the phenotype coherent?"}
        for target in ("C1", "C2")
    ]

    class Verifier:
        audits = 0

        def invoke(self, payload):
            if payload["mode"] == "select":
                target = payload["evidence_request"]["target_ids"][0]
                timeline.append(("select", payload["wave"], target))
                selection_calls.append((payload, self.audits))
                if payload["wave"] == 1:
                    return {"selected_tool": "rna"}
                if target == "C1":
                    return {"selected_tool": None, "stop_reason": "No further evidence needed."}
                return {"selected_tool": "wxs"}
            self.audits += 1
            timeline.append(("audit", payload["wave"]))
            return valid_reports(payload)

    verifier = Verifier()
    result = verifier_node(request_state(requests), context(registry, verifier))

    assert timeline[0][0] == timeline[1][0] == "select"
    assert timeline[2][0] == timeline[3][0] == "execute"
    wave_two = [(payload, audits) for payload, audits in selection_calls if payload["wave"] == 2]
    assert len(wave_two) == 2 and all(audits == 1 for _, audits in wave_two)
    assert all(payload["current_evidence"] for payload, _ in wave_two)
    assert sorted(row["tool_name"] for row in result["round_evidence"]) == ["rna", "rna", "wxs"]
    assert {(row["tool_name"], row["target_ids"][0]) for row in result["tool_evidence"]} == {
        ("rna", "C1"), ("rna", "C2"), ("wxs", "C2"),
    }


def test_mandatory_partition_screen_bypasses_selection():
    registry = make_registry(("structural_diagnostics", "cross_modal_consistency", ("partition",)))
    calls = []
    registry["structural_diagnostics"]["function"] = lambda **kwargs: (
        calls.append((kwargs["scope"], kwargs["target_ids"]))
        or {"status": "success", "metrics": {"partition": {"screen": 1}}}
    )

    class Verifier:
        def invoke(self, payload):
            assert payload["mode"] == "audit"
            return valid_reports(payload)

    result = verifier_node(request_state([{
        "dimension": "cross_modal_consistency", "scope": "partition", "target_ids": [],
        "question": "Screen current partition structure.",
    }]), context(registry, Verifier()))
    assert calls == [("partition", [])]
    assert result["round_evidence"][0]["tool_name"] == "structural_diagnostics"


def test_router_payload_hides_aspects_but_python_validation_keeps_them():
    registry = make_registry(("rna", "biological_support", ("set",)))
    state = request_state([], sets=(("C1", ["P1", "P2"]),))
    state["tool_evidence"].append({
        "partition_signature": partition_signature(state["partition"]["sets"]),
        "tool_name": "structural_diagnostics", "scope": "partition", "target_ids": [],
    })
    captured = {}

    class Router:
        def invoke(self, payload):
            captured.update(payload)
            return {"evidence_requests": [{
                "dimension": "biological_support", "scope": "set", "target_ids": ["C1"],
                "question": "Is this candidate biologically coherent?",
            }], "actions": []}

    state.update(router_node(state, context(registry, None, router_model=Router())))
    option = captured["available_evidence_requests"][0]
    assert set(option) == {"dimension", "scope", "target_ids"}
    assert "available_aspects" not in str(captured)
    assert "rna" not in str(captured)
    validate_router_plan(RouterPlan(evidence_requests=[{
        "dimension": "biological_support", "scope": "set", "target_ids": ["C1"],
        "question": "Is this candidate biologically coherent?",
    }]), state, {
        ("biological_support", "set", ("C1",)),
    })
    with pytest.raises(ValueError, match="unavailable"):
        validate_router_plan(RouterPlan(evidence_requests=[{
            "dimension": "known_label_echo", "scope": "set", "target_ids": ["C1"],
            "question": "Is it known?",
        }]), state, {("biological_support", "set", ("C1",))})


def test_selection_tools_have_empty_argument_schema():
    tools = {item.name: item for item in build_selection_tools()}
    assert tools
    assert all(not item.args_schema.model_fields for item in tools.values())


def test_latest_acquisition_uses_current_round_evidence_even_when_report_ref_is_stable():
    registry = make_registry(
        ("structural_diagnostics", "cross_modal_consistency", ("partition",)),
        ("tool_a", "biological_support", ("set",)),
        ("tool_b", "biological_support", ("set",)),
    )
    state = request_state([])
    signature = partition_signature(state["partition"]["sets"])
    request = EvidenceRequest(
        dimension="biological_support", scope="set", target_ids=["C1"], question="Follow up?",
    )
    report_ref = evidence_report_ref(signature, "biological_support", "shared", "set", ("C1",))
    old_report = {
        "report_ref": report_ref, "tool_refs": ["tool_a"], "dimension": "biological_support",
        "aspect": "shared", "scope": "set", "target_ids": ["C1"], "observations": [],
        "dimension_interpretation": "Prior", "cross_evidence_context": "Prior", "limitations": [],
    }
    partition_ref = "ER:partition"
    state["reports"] = [
        {**old_report, "tool_refs": ["tool_a", "tool_b"]},
        {"report_ref": partition_ref, "dimension": "cross_modal_consistency",
         "aspect": "structural_diagnostics", "scope": "partition", "target_ids": []},
    ]
    state["tool_evidence"] = [{
        "partition_signature": signature, "tool_name": "structural_diagnostics",
        "scope": "partition", "target_ids": [], "metrics": {"partition": {}},
    }]
    state["round_evidence"] = [{
        "partition_signature": signature, "dimension": "biological_support",
        "aspect": "shared", "scope": "set", "target_ids": ["C1"], "tool_name": "tool_b",
    }]
    state["history"] = [{
        "round": 4, "partition_signature": signature, "evidence_reports": [old_report],
        "router_plan": RouterPlan(evidence_requests=[request]).model_dump(),
    }]

    captured = {}

    class Router:
        def invoke(self, payload):
            captured.update(payload)
            return {"actions": [
                {"action": "accept", "target_ids": [target], "evidence_report_refs": [
                    partition_ref, *([report_ref] if target == "C1" else []),
                ], "reason": "Retain."}
                for target in ("C1", "C2")
            ], "evidence_requests": []}

    router_node(state, context(registry, None, router_model=Router()))
    assert captured["latest_acquisition_closure"]["source_round"] == 4
    assert captured["latest_acquisition_closure"]["new_report_refs"] == [report_ref]
    assert captured["latest_acquisition_closure"]["required_terminal_report_refs_by_target"] == {
        "C1": [report_ref], "C2": [],
    }


def test_latest_acquisition_excludes_old_reports_and_includes_all_new_aspects():
    state = request_state([])
    signature = partition_signature(state["partition"]["sets"])
    requests = [
        EvidenceRequest(dimension="biological_support", scope="set", target_ids=["C1"], question="Set?"),
        EvidenceRequest(dimension="cross_modal_consistency", scope="pair", target_ids=["C1", "C2"], question="Pair?"),
    ]
    old_reports = [
        {"report_ref": "ER:old-1", "dimension": "biological_support", "aspect": "old", "scope": "set", "target_ids": ["C1"]},
        {"report_ref": "ER:old-2", "dimension": "known_label_echo", "aspect": "old", "scope": "set", "target_ids": ["C2"]},
    ]
    rows = [
        ("biological_support", "rna", "set", ["C1"]),
        ("biological_support", "wxs", "set", ["C1"]),
        ("cross_modal_consistency", "representation", "pair", ["C1", "C2"]),
    ]
    new_reports = [{
        "report_ref": evidence_report_ref(signature, dimension, aspect, scope, tuple(targets)),
        "dimension": dimension, "aspect": aspect, "scope": scope, "target_ids": targets,
    } for dimension, aspect, scope, targets in rows]
    state["reports"] = [*old_reports, *new_reports]
    state["round_evidence"] = [{
        "partition_signature": signature, "dimension": dimension, "aspect": aspect,
        "scope": scope, "target_ids": targets, "tool_name": aspect,
    } for dimension, aspect, scope, targets in rows]
    state["history"] = [{
        "round": 3, "partition_signature": signature, "evidence_reports": old_reports,
        "router_plan": RouterPlan(evidence_requests=requests).model_dump(),
    }]

    context = latest_acquisition_context(state, signature)
    assert context["source_round"] == 3
    assert {row["report_ref"] for row in context["new_reports"]} == {
        row["report_ref"] for row in new_reports
    }


def test_optional_acquisition_stop_without_new_report_creates_no_closure():
    state = request_state([])
    signature = partition_signature(state["partition"]["sets"])
    state["history"] = [{
        "round": 2, "partition_signature": signature, "evidence_reports": [],
        "router_plan": RouterPlan(evidence_requests=[EvidenceRequest(
            dimension="biological_support", scope="set", target_ids=["C1"], question="Follow up?",
        )]).model_dump(),
    }]
    state["round_evidence"] = []
    acquisition = latest_acquisition_context(state, signature)
    assert acquisition["new_reports"] == []
    assert terminal_closure_refs(state["partition"]["sets"], acquisition["new_reports"]) == {
        "C1": set(), "C2": set(),
    }


def test_terminal_actions_must_cite_latest_pair_report_for_both_sets():
    state = request_state([])
    signature = partition_signature(state["partition"]["sets"])
    pair_ref = evidence_report_ref(signature, "cross_modal_consistency", "representation_concordance", "pair", ("C1", "C2"))
    base_ref = "ER:partition-screen"
    state["reports"] = [
        {"report_ref": base_ref, "dimension": "cross_modal_consistency", "aspect": "structural_diagnostics", "scope": "partition", "target_ids": []},
        {"report_ref": pair_ref, "dimension": "cross_modal_consistency", "aspect": "representation_concordance", "scope": "pair", "target_ids": ["C1", "C2"]},
    ]
    plan = RouterPlan(actions=[
        RouterAction(action="accept", target_ids=["C1"], evidence_report_refs=[base_ref], reason="Reason."),
        RouterAction(action="accept", target_ids=["C2"], evidence_report_refs=[base_ref, pair_ref], reason="Reason."),
    ])
    with pytest.raises(ValueError, match="does not close.*C1"):
        validate_router_plan(plan, state, set(), required_terminal_refs={
            "C1": {pair_ref}, "C2": {pair_ref},
        })


def test_terminal_actions_must_cite_new_partition_report_for_every_set():
    state = request_state([])
    state["reports"] = [{
        "report_ref": "ER:old-partition", "dimension": "cross_modal_consistency",
        "aspect": "structural_diagnostics", "scope": "partition", "target_ids": [],
    }, {
        "report_ref": "ER:new-partition", "dimension": "cross_modal_consistency",
        "aspect": "structural_diagnostics", "scope": "partition", "target_ids": [],
    }]
    plan = RouterPlan(actions=[
        RouterAction(action="accept", target_ids=[target], evidence_report_refs=["ER:old-partition"], reason="Reason.")
        for target in ("C1", "C2")
    ])
    with pytest.raises(ValueError, match="does not close"):
        validate_router_plan(plan, state, set(), required_terminal_refs={
            "C1": {"ER:new-partition"}, "C2": {"ER:new-partition"},
        })


def test_follow_up_requests_remain_valid_without_closing_latest_reports():
    state = request_state([])
    request = EvidenceRequest(
        dimension="biological_support", scope="set", target_ids=["C1"], question="Unresolved?",
    )
    validate_router_plan(
        RouterPlan(evidence_requests=[request]), state,
        {("biological_support", "set", ("C1",))},
        required_terminal_refs={"C1": {"ER:latest"}},
    )


def test_router_limits_cross_modal_pair_requests_to_structural_screen_targets():
    registry = make_registry(
        ("structural_diagnostics", "cross_modal_consistency", ("partition", "pair")),
        ("representation_concordance", "cross_modal_consistency", ("pair",)),
        ("rna", "biological_support", ("pair",)),
    )
    state = request_state([], sets=(
        ("C1", ["P1"]), ("C2", ["P2"]), ("C3", ["P3"]),
    ))
    signature = partition_signature(state["partition"]["sets"])
    state["tool_evidence"] = [{
        "partition_signature": signature, "tool_name": "structural_diagnostics",
        "scope": "partition", "target_ids": [],
        "metrics": {"partition": {"nearest_pair_targets": [["C1", "C2"], ["C2", "C3"]]}},
    }]
    captured = {}

    class Router:
        def invoke(self, payload):
            captured.update(payload)
            return {"actions": [{
                "action": "accept", "target_ids": [target], "evidence_report_refs": ["ER:screen"],
                "reason": "Positive evidence justifies retention.",
            } for target in ("C1", "C2", "C3")], "evidence_requests": []}

    state["reports"] = [{
        "report_ref": "ER:screen", "dimension": "cross_modal_consistency",
        "aspect": "structural_diagnostics", "scope": "partition", "target_ids": [],
    }]
    router_node(state, context(registry, None, router_model=Router()))
    available = captured["available_evidence_requests"]
    cross_modal_pairs = {
        tuple(item["target_ids"]) for item in available
        if item["dimension"] == "cross_modal_consistency" and item["scope"] == "pair"
    }
    assert cross_modal_pairs == {("C1", "C2"), ("C2", "C3")}
    assert {tuple(item["target_ids"]) for item in available if item["dimension"] == "biological_support"} == {
        ("C1", "C2"), ("C1", "C3"), ("C2", "C3"),
    }


def test_router_retries_closure_violation_without_changing_scientific_action(tmp_path):
    registry = make_registry(
        ("structural_diagnostics", "cross_modal_consistency", ("partition",)),
        ("rna", "biological_support", ("set",)),
    )
    state = request_state([])
    signature = partition_signature(state["partition"]["sets"])
    required_ref = evidence_report_ref(signature, "biological_support", "rna", "set", ("C1",))
    state["reports"] = [{
        "report_ref": "ER:partition", "dimension": "cross_modal_consistency",
        "aspect": "structural_diagnostics", "scope": "partition", "target_ids": [],
    }, {
        "report_ref": required_ref, "dimension": "biological_support", "aspect": "rna",
        "scope": "set", "target_ids": ["C1"],
    }]
    state["round_evidence"] = [{
        "partition_signature": signature, "dimension": "biological_support", "aspect": "rna",
        "scope": "set", "target_ids": ["C1"], "tool_name": "rna",
    }]
    state["tool_evidence"] = [{
        "partition_signature": signature, "tool_name": "structural_diagnostics",
        "scope": "partition", "target_ids": [], "metrics": {"partition": {}},
    }]
    state["history"] = [{
        "round": 1, "partition_signature": signature, "evidence_reports": [],
        "router_plan": RouterPlan(evidence_requests=[EvidenceRequest(
            dimension="biological_support", scope="set", target_ids=["C1"], question="RNA follow-up?",
        )]).model_dump(),
    }]

    class Router:
        config = {"router_plan_validation_retries": 1}

        def __init__(self):
            self.payloads = []

        def invoke(self, payload):
            self.payloads.append(payload)
            refs = ["ER:partition"] if len(self.payloads) == 1 else ["ER:partition", required_ref]
            return {"actions": [
                {"action": "accept", "target_ids": ["C1"], "evidence_report_refs": refs, "reason": "Retain."},
                {"action": "accept", "target_ids": ["C2"], "evidence_report_refs": ["ER:partition"], "reason": "Retain."},
            ], "evidence_requests": []}

    router = Router()
    trace_path = tmp_path / "trace.jsonl"
    state["control"]["max_rounds"] = 10
    output = router_node(state, context(registry, None, router_model=router, runtime_trace_path=str(trace_path)))
    assert len(router.payloads) == 2
    assert router.payloads[1]["validation_feedback"]["error"]
    instruction = router.payloads[1]["validation_feedback"]["instruction"].lower()
    assert "preserve the action" in instruction
    assert "scientific conclusions" in instruction
    assert "drop" not in instruction
    assert output["router_plan"]["actions"][0]["action"] == "accept"
    events = [json.loads(line)["event"] for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert "router_plan_validation_retry" in events


class BoundModel:
    model_name = "fake"

    def __init__(self, responses):
        self.responses = iter(responses)
        self.bind_calls = []

    def bind_tools(self, tools, **kwargs):
        self.bind_calls.append((tools, kwargs))
        return SimpleNamespace(invoke=lambda messages: next(self.responses))


def test_first_selection_requires_one_tool_and_retries_multiple_calls():
    model = BoundModel([
        SimpleNamespace(content="", tool_calls=[
            {"name": "rna", "args": {}, "id": "1"},
            {"name": "wxs", "args": {}, "id": "2"},
        ]),
        SimpleNamespace(content="", tool_calls=[{"name": "rna", "args": {}, "id": "3"}]),
    ])
    verifier = VerifierChatModel(model, None, "prompt", [
        SimpleNamespace(name="rna"), SimpleNamespace(name="wxs"),
    ], selection_retries=1)
    result = verifier.invoke({
        "mode": "select", "remaining_tools": ["rna", "wxs"], "require_tool": True,
        "round": 1,
    })
    assert result == {"selected_tool": "rna"}
    assert [call[1] for call in model.bind_calls] == [
        {"tool_choice": "required"}, {"tool_choice": "required"},
    ]


def test_later_selection_can_stop_without_a_tool_call():
    model = BoundModel([SimpleNamespace(content="No further evidence is needed.", tool_calls=[])])
    verifier = VerifierChatModel(model, None, "prompt", [SimpleNamespace(name="wxs")])
    result = verifier.invoke({
        "mode": "select", "remaining_tools": ["wxs"], "require_tool": False,
        "round": 2,
    })
    assert result["selected_tool"] is None
    assert result["stop_reason"] == "verifier_no_further_tool_call"
    assert model.bind_calls[0][1] == {}


def test_nonempty_selection_arguments_are_rejected_or_retried():
    model = BoundModel([
        SimpleNamespace(content="", tool_calls=[{"name": "rna", "args": {"scope": "set"}}]),
    ])
    verifier = VerifierChatModel(model, None, "prompt", [SimpleNamespace(name="rna")], selection_retries=0)
    with pytest.raises(ValueError, match="arguments must be empty"):
        verifier.invoke({"mode": "select", "remaining_tools": ["rna"], "require_tool": True})


def test_request_ref_is_stable_and_request_reports_are_exactly_scoped():
    request = EvidenceRequest(
        dimension="biological_support", scope="set", target_ids=["C1"], question="Question?",
    )
    ref = evidence_request_ref("signature", request)
    assert ref == evidence_request_ref("signature", request)
    assert ref.startswith("RQ:")
    assert len(reports_for_request([
        {"dimension": "biological_support", "scope": "set", "target_ids": ["C1"], "report_ref": "ER1"},
        {"dimension": "known_label_echo", "scope": "set", "target_ids": ["C1"], "report_ref": "ER2"},
    ], request)) == 1
