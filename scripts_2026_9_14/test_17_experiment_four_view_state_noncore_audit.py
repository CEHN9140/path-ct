import importlib.util
from pathlib import Path


path = Path(__file__).with_name("17_experiment_four_view_state_noncore_audit.py")
spec = importlib.util.spec_from_file_location("state_noncore_audit", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_categorical_inputs_follow_frozen_patient_order():
    order = ["case_B", "case_A", "case_C"]
    source = {"case_A": {"stage": "I"}, "case_B": {"stage": "II"}}
    assert module.ordered_values(order, source, "stage") == [
        ("case_B", "II"), ("case_A", "I"), ("case_C", "")
    ]
