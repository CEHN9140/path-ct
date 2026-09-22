from agents.subtype_review.graph import initial_review_state, partition_signature, validate_router_plan
from agents.subtype_review.schemas import RouterAction, RouterPlan


def report(ref, dimension, scope, target_ids, *, focus=None):
    row = {
        "report_ref": ref,
        "dimension": dimension,
        "aspect": "structural_diagnostics" if dimension == "cross_modal_consistency" else "rna_pathway_enrichment",
        "scope": scope,
        "target_ids": list(target_ids),
    }
    if focus:
        row["request_foci"] = [focus]
    return row


def state_with_reports(sets, reports, tool_evidence=()):
    state = initial_review_state(sets)
    state["reports"] = list(reports)
    state["tool_evidence"] = list(tool_evidence)
    return state


def test_controlled_action_fixtures_cover_accept_drop_split_merge():
    base_sets = [
        {"set_id": "C1", "member_ids": ["P1", "P2"]},
        {"set_id": "C2", "member_ids": ["P3", "P4"]},
    ]
    partition = report("ER:partition", "cross_modal_consistency", "partition", [])

    accept_state = state_with_reports(base_sets, [
        partition,
        report("ER:C1-membership", "cross_modal_consistency", "set", ["C1"], focus="membership_representation"),
        report("ER:C1-biology", "biological_support", "set", ["C1"]),
        report("ER:C2-biology", "biological_support", "set", ["C2"]),
    ])
    accept = RouterAction(action="accept", target_ids=["C1"], evidence_report_refs=[
        "ER:C1-membership", "ER:C1-biology",
    ], reason="identity and membership are supported")
    drop = RouterAction(action="drop", target_ids=["C2"], evidence_report_refs=[
        "ER:C2-biology",
    ], reason="independent retention is unsupported")
    validate_router_plan(RouterPlan(actions=[accept, drop]), accept_state, set())

    split_sets = [{"set_id": "C1", "member_ids": ["P1", "P2", "P3", "P4"]},
                  {"set_id": "C2", "member_ids": ["P5", "P6"]}]
    split_state = state_with_reports(
        split_sets,
        [partition, report("ER:C1-structure", "cross_modal_consistency", "set", ["C1"])],
        [{
            "tool_name": "structural_diagnostics", "scope": "set", "target_ids": ["C1"],
            "partition_signature": partition_signature(split_sets),
            "metrics": {"set": {"C1": {"solutions": {"2": {"child_sizes": [2, 2]}}}}},
        }],
    )
    split = RouterAction(action="split", target_ids=["C1"], n_children=2,
                         evidence_report_refs=["ER:C1-structure"], reason="feasible subdivision")
    validate_router_plan(RouterPlan(actions=[split]), split_state, set())

    merge_sets = split_sets + [{"set_id": "C3", "member_ids": ["P7", "P8"]}]
    merge_state = state_with_reports(
        merge_sets,
        [partition, report("ER:C1-C2-structure", "cross_modal_consistency", "pair", ["C1", "C2"])],
    )
    merge = RouterAction(action="merge", target_ids=["C1", "C2"],
                         evidence_report_refs=["ER:C1-C2-structure"], reason="revise boundary")
    validate_router_plan(RouterPlan(actions=[merge]), merge_state, set())
