import importlib.util
from pathlib import Path

import pandas as pd


path = Path(__file__).with_name("16_experiment_four_view_macro_micro_consistency.py")
spec = importlib.util.spec_from_file_location("macro_micro_consistency", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_consistency_results_identify_discovery_space_diagnostics(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "PERMUTATIONS", 9)
    output = tmp_path / "consistency"
    module.run(output_root=output)
    for filename in ("macro_micro_consistency_statistics.csv", "macro_micro_consistency_weighted.csv"):
        table = pd.read_csv(output / filename)
        assert set(table.analysis_role) == {"discovery_space_diagnostic"}
