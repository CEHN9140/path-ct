from __future__ import annotations

import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

Image.MAX_IMAGE_PIXELS = 1000000000


def process_patch_batch(batch: tuple[str, str, int, int, list[tuple[int, int]]]) -> list[list[int]]:
    from openslide import OpenSlide

    slide_path, patch_dir, level, patch_size, coordinates = batch
    saved_coordinates = []
    with OpenSlide(slide_path) as slide:
        for level0_x, level0_y in coordinates:
            patch_image = slide.read_region(
                (level0_x, level0_y),
                level,
                (patch_size, patch_size),
            ).convert("RGB")
            patch_image.info.clear()
            patch_image.save(Path(patch_dir) / f"{level0_y}_{level0_x}.png", icc_profile=None)
            saved_coordinates.append([level0_x, level0_y])
    return saved_coordinates


def find_qc_mask_path(
    output_root: str, case_id: str, wsi_record: Mapping[str, Any]
) -> Path | None:
    case_root = Path(output_root) / "wsi_qc" / case_id / "grandqc"
    slide_path = str(wsi_record.get("File Path", "") or "").strip()
    slide_id = str(wsi_record.get("Slide_ID", "") or "").strip()
    slide_name = os.path.basename(slide_path).replace(".svs", "")

    dir_names: list[str] = []
    file_names: list[str] = []
    for token in [slide_name, slide_id]:
        if token and token not in dir_names:
            dir_names.append(token)
        if token and token not in file_names:
            file_names.append(token)

    for dir_name in dir_names:
        for file_name in file_names:
            candidate = case_root / dir_name / "mask_qc" / f"{file_name}_mask.png"
            if candidate.exists():
                return candidate
    return None


def run_seg_wsi_patch(
    case_id: str,
    wsi_records: Sequence[Mapping[str, Any]],
    output_root: str = "/data/qijun/path-ct/output",
    config_dir: str = "",
) -> dict[str, Any]:
    import h5py
    import yaml
    from openslide import OpenSlide

    tool_config = yaml.safe_load((Path(config_dir).expanduser() / "wsi_patch.yaml").read_text(encoding="utf-8")) or {}
    patch_size = int(tool_config["patch_size"])
    level = int(tool_config["level"])
    qc_clean_tissue_value = int(tool_config["qc_clean_tissue_value"])
    num_workers = int(tool_config["num_workers"])
    batch_size = int(tool_config["batch_size"])
    if batch_size < 1:
        raise ValueError("wsi_patch.batch_size must be >= 1")

    result: dict[str, Any] = {
        "case_id": case_id,
        "patch_size": patch_size,
        "level": level,
        "coordinate_level": 0,
        "qc_clean_tissue_value": qc_clean_tissue_value,
        "slides": [],
        "errors": [],
    }
    if not wsi_records:
        result["errors"].append(
            f"No selected WSI records were provided for case '{case_id}'."
        )
        return result

    for wsi_record in wsi_records:
        slide_path = str(wsi_record.get("File Path", "") or "").strip()
        slide_name = os.path.basename(slide_path).replace(".svs", "") or str(
            wsi_record.get("Slide_ID", "") or case_id
        )
        try:
            if not slide_path:
                raise ValueError("WSI record is missing File Path.")

            qc_mask_path = find_qc_mask_path(output_root, case_id, wsi_record)
            if qc_mask_path is None:
                raise FileNotFoundError(
                    f"Unable to find qc mask for slide '{slide_name}' under case '{case_id}'."
                )

            patch_dir = Path(output_root) / "wsi_patch" / case_id
            coordinates_h5_path = Path(output_root) / "wsi_patch" / f"{case_id}.h5"
            if patch_dir.exists():
                shutil.rmtree(patch_dir)
            if coordinates_h5_path.exists():
                coordinates_h5_path.unlink()
            patch_dir.mkdir(parents=True, exist_ok=True)

            qc_mask = np.array(Image.open(qc_mask_path))
            if qc_mask.ndim == 3:
                qc_mask = qc_mask[..., 0]

            with OpenSlide(slide_path) as slide:
                if level < 0 or level >= len(slide.level_dimensions):
                    raise ValueError(
                        f"Invalid WSI patch level {level}; slide has {len(slide.level_dimensions)} levels."
                    )
                level_width, level_height = slide.level_dimensions[level]
                level_downsample = float(slide.level_downsamples[level])

            mask_height, mask_width = qc_mask.shape[:2]
            patch_coordinates: list[tuple[int, int]] = []
            for patch_y in range(0, level_height, patch_size):
                for patch_x in range(0, level_width, patch_size):
                    patch_x_end = min(patch_x + patch_size, level_width)
                    patch_y_end = min(patch_y + patch_size, level_height)
                    mask_x0 = int(np.floor(patch_x * mask_width / level_width))
                    mask_y0 = int(np.floor(patch_y * mask_height / level_height))
                    mask_x1 = int(np.ceil(patch_x_end * mask_width / level_width))
                    mask_y1 = int(np.ceil(patch_y_end * mask_height / level_height))
                    mask_x0 = min(max(mask_x0, 0), max(mask_width - 1, 0))
                    mask_y0 = min(max(mask_y0, 0), max(mask_height - 1, 0))
                    mask_x1 = min(max(mask_x1, mask_x0 + 1), mask_width)
                    mask_y1 = min(max(mask_y1, mask_y0 + 1), mask_height)

                    mask_patch = qc_mask[mask_y0:mask_y1, mask_x0:mask_x1]
                    if mask_patch.size and np.any(mask_patch == qc_clean_tissue_value):
                        patch_coordinates.append(
                            (
                                int(round(patch_x * level_downsample)),
                                int(round(patch_y * level_downsample)),
                            )
                        )

            batches = [
                (slide_path, str(patch_dir), level, patch_size, patch_coordinates[index:index + batch_size])
                for index in range(0, len(patch_coordinates), batch_size)
            ]
            saved_coordinates: list[list[int]] = []
            worker_count = os.cpu_count() if num_workers <= 0 else num_workers
            worker_count = max(1, min(int(worker_count or 1), len(batches) or 1))
            if worker_count == 1:
                batch_results = map(process_patch_batch, batches)
                try:
                    from tqdm.auto import tqdm

                    batch_results = tqdm(
                        batch_results,
                        total=len(batches),
                        desc=f"[wsi_patch] {case_id}",
                        unit="batch",
                    )
                except Exception:
                    pass
                for batch_coordinates in batch_results:
                    saved_coordinates.extend(batch_coordinates)
            else:
                with ProcessPoolExecutor(max_workers=worker_count) as executor:
                    batch_results = executor.map(process_patch_batch, batches)
                    try:
                        from tqdm.auto import tqdm

                        batch_results = tqdm(
                            batch_results,
                            total=len(batches),
                            desc=f"[wsi_patch] {case_id}",
                            unit="batch",
                        )
                    except Exception:
                        pass
                    for batch_coordinates in batch_results:
                        saved_coordinates.extend(batch_coordinates)

            with h5py.File(coordinates_h5_path, "w") as handle:
                handle.create_dataset(
                    "coordinates",
                    data=np.asarray(saved_coordinates, dtype=np.int64),
                )
                handle.attrs["slide_name"] = slide_name
                handle.attrs["slide_path"] = slide_path
                handle.attrs["patch_size"] = patch_size
                handle.attrs["level"] = level
                handle.attrs["coordinate_level"] = 0
                handle.attrs["level_downsample"] = level_downsample

            result["slides"].append(
                {
                    "slide_name": slide_name,
                    "slide_path": slide_path,
                    "patch_dir": str(patch_dir),
                    "coordinates_h5_path": str(coordinates_h5_path),
                    "patch_count": len(saved_coordinates),
                    "qc_mask_path": str(qc_mask_path),
                    "level": level,
                    "coordinate_level": 0,
                    "level_downsample": level_downsample,
                    "qc_clean_tissue_value": qc_clean_tissue_value,
                }
            )
        except Exception as exc:
            result["errors"].append(
                f"Failed to extract patches for '{slide_name}': {type(exc).__name__}: {exc}"
            )
    return result
