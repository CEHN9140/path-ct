from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = 1000000000


def resolve_device(device_value: object = "auto") -> str:
    import torch

    device = str(device_value or "auto").strip().lower()
    if device == "cpu":
        return "cpu"
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda:") and torch.cuda.is_available():
        try:
            device_index = int(device.split(":", 1)[1])
        except ValueError:
            return "cuda"
        return device if device_index < torch.cuda.device_count() else "cpu"
    return "cpu"


def artifacts_seg(
    path_slide: str,
    output_dir: str,
    config: dict[str, object] | None = None,
) -> None:
    import torch
    from openslide import open_slide

    from tools.pathology_qc.wsi_colors import colors_QC7 as colors
    from tools.pathology_qc.wsi_maps import make_overlay
    from tools.pathology_qc.wsi_process import mask_to_geojson, slide_process_single
    from tools.pathology_qc.wsi_slide_info import slide_info
    config = dict(config or {})
    config_dir = str(config.get("config_dir", "") or "")
    model_dir = Path(str(config["model_dir"])).expanduser()
    if not model_dir.is_absolute():
        model_dir = (Path(config_dir).expanduser().parent / model_dir).resolve() if config_dir else model_dir.resolve()
    device = resolve_device(config["device"])
    overlay_factor = int(config["overlay_factor"])
    mpp_model = float(config["mpp_model"])
    create_geojson = str(config["create_geojson"])
    create_geojson_enabled = create_geojson.upper() in {"Y", "YES", "TRUE", "1"}
    model_qc_dir = model_dir
    model_name_by_mpp = {
        str(key): str(value)
        for key, value in dict(config["model_name_by_mpp"]).items()
    }
    model_qc_name = model_name_by_mpp.get(str(mpp_model), "")
    if not model_qc_name:
        raise ValueError(
            "mpp_model must have a matching checkpoint in model_name_by_mpp"
        )

    patch_size = int(config["patch_size"])
    encoder_model = str(config["encoder_model"])
    encoder_weights = str(config["encoder_weights"])
    back_class = int(config["back_class"])
    slide_name = os.path.basename(path_slide).replace(".svs", "")

    geojson_root = os.path.join(output_dir, "geojson_qc")
    if create_geojson_enabled:
        os.makedirs(geojson_root, exist_ok=True)

    maps_dir = os.path.join(output_dir, "maps_qc")
    overlay_dir = os.path.join(output_dir, "overlays_qc")
    mask_dir = os.path.join(output_dir, "mask_qc")
    os.makedirs(maps_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    model_prim = torch.load(
        str(model_qc_dir / model_qc_name), map_location=device, weights_only=False
    )

    slide = open_slide(path_slide)
    p_s, patch_n_w_l0, patch_n_h_l0, mpp, w_l0, h_l0, _obj_power = slide_info(
        slide, patch_size, mpp_model
    )

    tis_det_map = Image.open(
        os.path.join(output_dir, "tis_det_mask", slide_name + "_MASK.png")
    ).convert("L")
    tis_det_map_mpp = np.array(
        tis_det_map.resize(
            (int(w_l0 * mpp / mpp_model), int(h_l0 * mpp / mpp_model)),
            Image.Resampling.LANCZOS,
        )
    )
    qc_map, full_mask = slide_process_single(
        model_prim,
        tis_det_map_mpp,
        slide,
        patch_n_w_l0,
        patch_n_h_l0,
        p_s,
        patch_size,
        colors,
        encoder_model,
        encoder_weights,
        device,
        back_class,
        mpp_model,
        mpp,
        w_l0,
        h_l0,
    )

    qc_map.save(os.path.join(maps_dir, slide_name + "_map_QC.png"))

    mask_path = os.path.join(mask_dir, slide_name + "_mask.png")
    cv2.imwrite(mask_path, full_mask)
    if create_geojson_enabled:
        factor = mpp_model / mpp
        mask_to_geojson(
            mask_path, os.path.join(geojson_root, slide_name + ".geojson"), factor
        )

    overlay = make_overlay(
        slide, qc_map, p_s, patch_n_w_l0, patch_n_h_l0, overlay_factor
    )
    Image.fromarray(overlay).save(
        os.path.join(overlay_dir, slide_name + "_overlay_QC.jpg")
    )
