import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from tools import wsi_embeddings


def test_wsi_embedding_config_global_pool_is_true():
    import yaml

    config = yaml.safe_load(Path("configs/wsi_embeddings.yaml").read_text())
    assert config["global_pool"] is True


def run_fake_slide(monkeypatch, tmp_path, model_global_pool):
    calls = {}

    class FakeSlideEncoder:
        global_pool = model_global_pool

        def eval(self):
            return self

        def to(self, device):
            return self

        def __call__(self, tile_embeddings, coordinates, all_layer_embed=False):
            return [torch.zeros((1, 3))]

    slide_module = types.ModuleType("gigapath.slide_encoder")

    def create_model(*args, **kwargs):
        calls["kwargs"] = kwargs
        return FakeSlideEncoder()

    slide_module.create_model = create_model
    gigapath_module = types.ModuleType("gigapath")
    gigapath_module.slide_encoder = slide_module
    monkeypatch.setitem(sys.modules, "gigapath", gigapath_module)
    monkeypatch.setitem(sys.modules, "gigapath.slide_encoder", slide_module)

    class FakeLoader:
        def __init__(self, *args, **kwargs):
            pass

        def __iter__(self):
            yield np.zeros((1, 3, 224, 224), dtype=np.float32), [[0, 0]]

    monkeypatch.setattr(torch.utils.data, "DataLoader", FakeLoader)
    monkeypatch.setattr(wsi_embeddings, "load_selected_patch_records", lambda *args: [(Path("patch.png"), [0, 0])])
    models = {
        "torch": torch,
        "device": torch.device("cpu"),
        "use_autocast": False,
        "tile_encoder": lambda batch: torch.zeros((batch.shape[0], 3)),
        "slide_encoder": None,
        "tile_embedding_dim": None,
        "slide_encoder_checkpoint_path": "checkpoint.pth",
        "slide_encoder_arch": "gigapath_slide_enc12l768d",
        "flash_attention_available": True,
        "flash_attention_error": "",
        "global_pool": True,
    }
    wsi_embeddings.run_gigapath_slide(
        {"case_id": "A", "output_dir": str(tmp_path), "patch_dir": str(tmp_path), "tumor_coordinates_h5_path": "coords.h5", "batch_size": 1, "num_workers": 0, "pin_memory": False, "prefetch_factor": 1, "max_tiles_without_flash_attention": 1024},
        models,
    )
    return calls


def test_gigapath_create_model_receives_global_pool_true(monkeypatch, tmp_path):
    calls = run_fake_slide(monkeypatch, tmp_path, True)
    assert calls["kwargs"]["global_pool"] is True


def test_wsi_embedding_rejects_false_global_pool_model(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="global_pool=True"):
        run_fake_slide(monkeypatch, tmp_path, False)
