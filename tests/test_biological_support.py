import pandas as pd

from tools.cnv_characterization import cnv_characterization
from tools.mutation_enrichment import enrichment_rows, mutation_enrichment
from tools.pathway_enrichment import pathway_enrichment, pathway_rows
from tools.subtype_review_common import ranked_decision_summary


def states():
    return [
        {"set_id": "C1", "member_ids": ["a1", "a2"]},
        {"set_id": "C2", "member_ids": ["b1", "b2"]},
    ]


def test_ranked_decision_summary_limits_and_orders_top5():
    rows = [
        {"feature": f"F{i}", "q_value": i / 10, "effect": 10 - i}
        for i in range(7)
    ]
    summary = ranked_decision_summary(rows, "effect")
    assert len(summary["top_by_q"]) == len(summary["top_by_effect"]) == 5
    assert [row["q_value"] for row in summary["top_by_q"]] == [0.0, 0.1, 0.2, 0.3, 0.4]
    assert [row["effect"] for row in summary["top_by_effect"]] == [10, 9, 8, 7, 6]


def test_rna_pathway_rows_use_setwise_fdr_and_direction(monkeypatch):
    p_values = iter([0.01, 0.02, 0.8, 0.9])

    class Result:
        def __init__(self, p_value):
            self.pvalue = p_value

    monkeypatch.setattr(
        "scipy.stats.mannwhitneyu",
        lambda *args, **kwargs: Result(next(p_values)),
    )
    scores = pd.DataFrame(
        {
            "P1": [4.0, 3.0, 1.0, 1.0],
            "P2": [3.0, 4.0, 1.0, 1.0],
        },
        index=["a1", "a2", "b1", "b2"],
    )
    rows = pathway_rows(scores, {"P1": 20, "P2": 20}, {
        "C1": {"a1", "a2"}, "C2": {"b1", "b2"}
    })

    c1 = [row for row in rows if row["candidate_set_id"] == "C1"]
    c2 = [row for row in rows if row["candidate_set_id"] == "C2"]
    assert [row["q_value"] for row in c1] == [0.02, 0.02]
    assert [row["q_value"] for row in c2] == [0.9, 0.9]
    assert {row["direction"] for row in c1} == {"up"}


def test_rna_decision_metrics_expose_only_core_fields(monkeypatch, tmp_path):
    scores = pd.DataFrame(
        {"P1": [4.0, 3.0, 1.0, 1.0]},
        index=["a1", "a2", "b1", "b2"],
    )
    monkeypatch.setattr("tools.pathway_enrichment.rna_feature_path", lambda _: "rna.csv")
    monkeypatch.setattr("tools.pathway_enrichment.feature_dataframe", lambda _: scores)
    monkeypatch.setattr(
        "tools.pathway_enrichment.read_gmt_gene_sets",
        lambda _: ({"P1": ["P1"] * 20}, {}),
    )
    monkeypatch.setattr("tools.pathway_enrichment.ssgsea_scores", lambda *args: scores)
    monkeypatch.setattr(
        "tools.pathway_enrichment.tool_parameters",
        lambda *_: {"min_pathway_overlap": 1, "pathway_gene_sets_path": "genes.gmt"},
    )
    result = pathway_enrichment(
        states()[0], {}, str(tmp_path), all_cluster_states=states()
    )
    decision = result["results"]["decision_metrics"]["per_set_rna_pathway_enrichment"]["C1"]
    row = decision["top_by_q"][0]
    assert set(row) == {
        "pathway", "standardized_mean_difference", "set_median_score",
        "rest_median_score", "set_available_n", "rest_available_n",
        "q_value", "direction"
    }
    assert "set_mean_score" in result["results"]["metrics"]["rna_pathway_enrichment"][0]
    assert set(decision) == {"summary", "top_by_q", "top_by_effect"}


def test_wxs_mutation_core_metrics_and_setwise_fdr():
    rows = enrichment_rows(
        ["mutation::TP53", "validation::TMB"],
        {
            "a1": {"mutation::TP53": 1, "validation::TMB": 10},
            "a2": {"mutation::TP53": 1, "validation::TMB": 11},
            "b1": {"mutation::TP53": 0, "validation::TMB": 1},
            "b2": {"mutation::TP53": 0, "validation::TMB": 2},
        },
        {"C1": {"a1", "a2"}, "C2": {"b1", "b2"}},
    )
    mutation_rows = [row for row in rows if row["wxs_block"] == "mutation"]
    assert {row["mutation_frequency"] for row in mutation_rows} == {0.0, 1.0}
    assert {row["rest_mutation_frequency"] for row in mutation_rows} == {0.0, 1.0}
    assert {row["delta_frequency"] for row in mutation_rows} == {-1.0, 1.0}
    assert {row["set_mutated_n"] for row in mutation_rows} == {0, 2}
    assert all(
        row["mutation_frequency"] == row["set_mutated_n"] / row["set_total_n"]
        and row["rest_mutation_frequency"]
        == row["rest_mutated_n"] / row["rest_total_n"]
        for row in mutation_rows
    )
    assert all(row["fisher_p_value"] is not None for row in mutation_rows)
    assert all(row["odds_ratio_ci95"] for row in mutation_rows)
    assert all(row["q_value"] is not None for row in rows)


def test_wxs_decision_metrics_keep_mutation_core_only(monkeypatch, tmp_path):
    feature_table = {
        "a1": {"mutation::TP53": 1, "validation::TMB": 10},
        "a2": {"mutation::TP53": 1, "validation::TMB": 11},
        "b1": {"mutation::TP53": 0, "validation::TMB": 1},
        "b2": {"mutation::TP53": 0, "validation::TMB": 2},
    }
    def read_table(path):
        return (
            (["mutation::TP53"], feature_table)
            if "discovery" in str(path)
            else (["validation::TMB"], feature_table)
        )

    monkeypatch.setattr("tools.mutation_enrichment.read_case_feature_table", read_table)
    result = mutation_enrichment(
        states()[0], {}, str(tmp_path), all_cluster_states=states()
    )
    decision = result["results"]["decision_metrics"]["per_set_wxs_feature_enrichment"]["C1"]
    assert set(decision) == {"summary", "top_by_q", "top_by_effect"}
    assert len(decision["top_by_q"]) == 1
    assert set(decision["top_by_q"][0]) == {
        "gene", "mutation_frequency", "rest_mutation_frequency", "delta_frequency",
        "odds_ratio", "odds_ratio_ci95", "fisher_p_value", "q_value",
        "set_mutated_n", "set_total_n", "rest_mutated_n", "rest_total_n"
    }


def test_wxs_fdr_separates_mutation_and_validation_blocks(monkeypatch):
    p_values = iter([0.01, 0.9, 0.02, 0.9])
    monkeypatch.setattr(
        "tools.mutation_enrichment.fisher_exact_result",
        lambda *args: (1.0, next(p_values)),
    )
    rows = enrichment_rows(
        ["mutation::TP53", "validation::TMB"],
        {
            "a1": {"mutation::TP53": 1, "validation::TMB": 1},
            "a2": {"mutation::TP53": 0, "validation::TMB": 0},
            "b1": {"mutation::TP53": 1, "validation::TMB": 1},
            "b2": {"mutation::TP53": 0, "validation::TMB": 0},
        },
        {"C1": {"a1", "a2"}, "C2": {"b1", "b2"}},
    )
    mutation_q = [row["q_value"] for row in rows if row["wxs_block"] == "mutation"]
    validation_q = [row["q_value"] for row in rows if row["wxs_block"] == "validation"]
    assert mutation_q == [0.01, 0.02]
    assert validation_q == [0.9, 0.9]


def test_cnv_decision_metrics_separate_events_from_continuous(monkeypatch, tmp_path):
    table = {
        "a1": {"chr1": 1.0}, "a2": {"chr1": 1.0},
        "b1": {"chr1": -1.0}, "b2": {"chr1": -1.0},
    }
    monkeypatch.setattr(
        "tools.cnv_characterization.read_case_feature_table",
        lambda _: (["chr1"], table),
    )
    result = cnv_characterization(
        states()[0], {}, str(tmp_path), all_cluster_states=states()
    )
    raw = result["results"]["metrics"]["cnv_characterization"]
    assert any(row["alteration_frequency"] is not None for row in raw)
    decision = result["results"]["decision_metrics"]["per_comparison_cnv"]["C1_vs_rest"]
    assert set(decision) == {"continuous", "gain_loss"}
    assert any(
        set(row) == {
            "feature", "alteration_frequency", "rest_alteration_frequency",
            "alteration_frequency_difference", "odds_ratio", "odds_ratio_ci95", "q_value",
            "left_altered_n", "left_total_n", "right_altered_n", "right_total_n"
        }
        for row in decision["gain_loss"]["top_by_q"]
    )
    event = decision["gain_loss"]["top_by_q"][0]
    assert event["alteration_frequency"] == event["left_altered_n"] / event["left_total_n"]
    assert event["rest_alteration_frequency"] == event["right_altered_n"] / event["right_total_n"]
    assert any(
        set(row) == {"feature", "cliffs_delta", "direction", "q_value"}
        for row in decision["continuous"]["top_by_q"]
    )
