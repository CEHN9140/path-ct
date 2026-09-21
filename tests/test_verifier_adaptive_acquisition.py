from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agents.subtype_review.graph import (
    evidence_request_ref,
    initial_review_state,
    partition_signature,
    reports_for_request,
    router_node,
    validate_router_plan,
    verifier_node,
)
from agents.subtype_review.llm import VerifierChatModel
from agents.subtype_review.schemas import EvidenceReport, EvidenceRequest, RouterPlan
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
