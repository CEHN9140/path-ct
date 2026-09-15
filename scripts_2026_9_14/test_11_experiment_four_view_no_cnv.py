import importlib.util
from pathlib import Path

path = Path(__file__).with_name("11_experiment_four_view_no_cnv.py")
spec = importlib.util.spec_from_file_location("four_view", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_four_view_disables_cnv_only():
    assert module.ACTIVE_MODALITIES == ("ct", "wsi", "rna", "wxs")
    assert "cnv" not in module.ACTIVE_MODALITIES


def test_manifest_declares_no_cnv_input(tmp_path):
    manifest = {
        "active_modalities": list(module.ACTIVE_MODALITIES),
        "disabled_modalities": ["cnv"],
    }
    assert "cnv" not in manifest["active_modalities"]
    assert manifest["disabled_modalities"] == ["cnv"]
