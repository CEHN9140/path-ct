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
patch_worker_slide = None


def mpp_patch_geometry(
    *,
    patch_size: int,
    target_mpp: float,
    source_mpp_x: float,
    source_mpp_y: float,
) -> dict[str, float | int]:
    if min(patch_size, target_mpp, source_mpp_x, source_mpp_y) <= 0:
        raise ValueError("Patch size and MPP must be positive.")
    return {
        "source_width": int(round(patch_size * target_mpp / source_mpp_x)),
        "source_height": int(round(patch_size * target_mpp / source_mpp_y)),
        "target_scale_x": source_mpp_x / target_mpp,
        "target_scale_y": source_mpp_y / target_mpp,
    }


def qc_mask_fractions(
    mask_patch: np.ndarray,
    *,
    full_mask_pixel_count: int,
    clean_tissue_value: int,
    artifact_values: Sequence[int],
    background_values: Sequence[int],
) -> dict[str, float]:
    if full_mask_pixel_count < mask_patch.size or full_mask_pixel_count < 1:
        raise ValueError("Full mask pixel count must cover the observed mask patch.")
    outside_pixel_count = full_mask_pixel_count - mask_patch.size
    return {
        "clean_tissue_fraction": float(
            np.count_nonzero(mask_patch == clean_tissue_value)
            / full_mask_pixel_count
        ),
        "artifact_fraction": float(
            np.count_nonzero(np.isin(mask_patch, artifact_values))
            / full_mask_pixel_count
        ),
        "background_fraction": float(
            (
                np.count_nonzero(np.isin(mask_patch, background_values))
                + outside_pixel_count
            )
            / full_mask_pixel_count
        ),
    }


def initialize_patch_worker(slide_path: str) -> None:
    from openslide import OpenSlide

    global patch_worker_slide
    patch_worker_slide = OpenSlide(slide_path)


def save_patch_batch(
    slide,
    batch: tuple[
        str,
        int,
        int,
        int,
        float,
        float,
        list[tuple[int, int]],
    ],
) -> tuple[list[list[int]], list[list[int]]]:
    (
        patch_dir,
        patch_size,
        source_width,
        source_height,
        target_scale_x,
        target_scale_y,
        source_coordinates,
    ) = batch
    saved_target_coordinates = []
    saved_source_coordinates = []
    for source_x, source_y in source_coordinates:
        patch_image = slide.read_region(
            (source_x, source_y),
            0,
            (source_width, source_height),
        ).convert("RGB")
        if patch_image.size != (patch_size, patch_size):
            patch_image = patch_image.resize(
                (patch_size, patch_size), Image.Resampling.BICUBIC
            )
        target_x = int(round(source_x * target_scale_x))
        target_y = int(round(source_y * target_scale_y))
        patch_image.info.clear()
        patch_image.save(
            Path(patch_dir) / f"{target_y}_{target_x}.png",
            icc_profile=None,
            compress_level=0,
        )
        saved_target_coordinates.append([target_x, target_y])
        saved_source_coordinates.append([source_x, source_y])
    return saved_target_coordinates, saved_source_coordinates


def process_patch_batch(
    batch: tuple[
        str,
        int,
        int,
        int,
        float,
        float,
        list[tuple[int, int]],
    ]
) -> tuple[list[list[int]], list[list[int]]]:
    if patch_worker_slide is None:
        raise RuntimeError("Patch worker slide was not initialized.")
    return save_patch_batch(patch_worker_slide, batch)


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

    tool_config = yaml.safe_load(
        (Path(config_dir).expanduser() / "wsi_patch.yaml").read_text(
            encoding="utf-8"
        )
    )
    patch_size = int(tool_config["patch_size"])
    target_mpp = float(tool_config["target_mpp"])
    min_clean_tissue_fraction = float(tool_config["min_clean_tissue_fraction"])
    max_artifact_fraction = float(tool_config["max_artifact_fraction"])
    qc_clean_tissue_value = int(tool_config["qc_clean_tissue_value"])
    qc_artifact_values = [int(value) for value in tool_config["qc_artifact_values"]]
    qc_background_values = [int(value) for value in tool_config["qc_background_values"]]
    num_workers = int(tool_config["num_workers"])
    batch_size = int(tool_config["batch_size"])
    if batch_size < 1:
        raise ValueError("wsi_patch.batch_size must be >= 1")
    if not 0.0 <= min_clean_tissue_fraction <= 1.0:
        raise ValueError("wsi_patch.min_clean_tissue_fraction must be between 0 and 1")
    if not 0.0 <= max_artifact_fraction <= 1.0:
        raise ValueError("wsi_patch.max_artifact_fraction must be between 0 and 1")

    result: dict[str, Any] = {
        "case_id": case_id,
        "patch_size": patch_size,
        "target_mpp": target_mpp,
        "coordinate_space": "target_mpp",
        "min_clean_tissue_fraction": min_clean_tissue_fraction,
        "max_artifact_fraction": max_artifact_fraction,
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
                source_mpp_x = float(slide.properties["openslide.mpp-x"])
                source_mpp_y = float(slide.properties["openslide.mpp-y"])
                level_width, level_height = slide.level_dimensions[0]
            geometry = mpp_patch_geometry(
                patch_size=patch_size,
                target_mpp=target_mpp,
                source_mpp_x=source_mpp_x,
                source_mpp_y=source_mpp_y,
            )
            source_width = int(geometry["source_width"])
            source_height = int(geometry["source_height"])
            target_scale_x = float(geometry["target_scale_x"])
            target_scale_y = float(geometry["target_scale_y"])

            mask_height, mask_width = qc_mask.shape[:2]
            patch_coordinates: list[tuple[int, int]] = []
            clean_tissue_fractions: list[float] = []
            artifact_fractions: list[float] = []
            background_fractions: list[float] = []
            for patch_y in range(0, level_height, source_height):
                for patch_x in range(0, level_width, source_width):
                    patch_x_end = min(patch_x + source_width, level_width)
                    patch_y_end = min(patch_y + source_height, level_height)
                    mask_x0 = int(np.floor(patch_x * mask_width / level_width))
                    mask_y0 = int(np.floor(patch_y * mask_height / level_height))
                    full_mask_x1 = int(
                        np.ceil(
                            (patch_x + source_width) * mask_width / level_width
                        )
                    )
                    full_mask_y1 = int(
                        np.ceil(
                            (patch_y + source_height) * mask_height / level_height
                        )
                    )
                    mask_x1 = int(np.ceil(patch_x_end * mask_width / level_width))
                    mask_y1 = int(np.ceil(patch_y_end * mask_height / level_height))
                    mask_x0 = min(max(mask_x0, 0), max(mask_width - 1, 0))
                    mask_y0 = min(max(mask_y0, 0), max(mask_height - 1, 0))
                    mask_x1 = min(max(mask_x1, mask_x0 + 1), mask_width)
                    mask_y1 = min(max(mask_y1, mask_y0 + 1), mask_height)

                    mask_patch = qc_mask[mask_y0:mask_y1, mask_x0:mask_x1]
                    if not mask_patch.size:
                        continue
                    fractions = qc_mask_fractions(
                        mask_patch,
                        full_mask_pixel_count=(full_mask_x1 - mask_x0)
                        * (full_mask_y1 - mask_y0),
                        clean_tissue_value=qc_clean_tissue_value,
                        artifact_values=qc_artifact_values,
                        background_values=qc_background_values,
                    )
                    clean_tissue_fraction = fractions["clean_tissue_fraction"]
                    artifact_fraction = fractions["artifact_fraction"]
                    if (
                        clean_tissue_fraction >= min_clean_tissue_fraction
                        and artifact_fraction <= max_artifact_fraction
                    ):
                        patch_coordinates.append((patch_x, patch_y))
                        clean_tissue_fractions.append(clean_tissue_fraction)
                        artifact_fractions.append(artifact_fraction)
                        background_fractions.append(fractions["background_fraction"])

            batches = [
                (
                    str(patch_dir),
                    patch_size,
                    source_width,
                    source_height,
                    target_scale_x,
                    target_scale_y,
                    patch_coordinates[index:index + batch_size],
                )
                for index in range(0, len(patch_coordinates), batch_size)
            ]
            saved_coordinates: list[list[int]] = []
            saved_source_coordinates: list[list[int]] = []
            worker_count = os.cpu_count() if num_workers <= 0 else num_workers
            worker_count = max(1, min(int(worker_count or 1), len(batches) or 1))
            if worker_count == 1:
                with OpenSlide(slide_path) as slide:
                    batch_results = (
                        save_patch_batch(slide, batch) for batch in batches
                    )
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
                    for batch_target_coordinates, batch_source_coordinates in batch_results:
                        saved_coordinates.extend(batch_target_coordinates)
                        saved_source_coordinates.extend(batch_source_coordinates)
            else:
                with ProcessPoolExecutor(
                    max_workers=worker_count,
                    initializer=initialize_patch_worker,
                    initargs=(slide_path,),
                ) as executor:
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
                    for batch_target_coordinates, batch_source_coordinates in batch_results:
                        saved_coordinates.extend(batch_target_coordinates)
                        saved_source_coordinates.extend(batch_source_coordinates)

            with h5py.File(coordinates_h5_path, "w") as handle:
                handle.create_dataset(
                    "coordinates",
                    data=np.asarray(saved_coordinates, dtype=np.int64).reshape(-1, 2),
                )
                handle.create_dataset(
                    "source_coordinates",
                    data=np.asarray(saved_source_coordinates, dtype=np.int64).reshape(-1, 2),
                )
                handle.create_dataset(
                    "clean_tissue_fractions",
                    data=np.asarray(clean_tissue_fractions, dtype=np.float32),
                )
                handle.create_dataset(
                    "artifact_fractions",
                    data=np.asarray(artifact_fractions, dtype=np.float32),
                )
                handle.create_dataset(
                    "background_fractions",
                    data=np.asarray(background_fractions, dtype=np.float32),
                )
                handle.attrs["slide_name"] = slide_name
                handle.attrs["slide_path"] = slide_path
                handle.attrs["patch_size"] = patch_size
                handle.attrs["level"] = 0
                handle.attrs["coordinate_space"] = "target_mpp"
                handle.attrs["target_mpp"] = target_mpp
                handle.attrs["source_mpp_x"] = source_mpp_x
                handle.attrs["source_mpp_y"] = source_mpp_y
                handle.attrs["source_patch_width"] = source_width
                handle.attrs["source_patch_height"] = source_height
                handle.attrs["min_clean_tissue_fraction"] = min_clean_tissue_fraction
                handle.attrs["max_artifact_fraction"] = max_artifact_fraction
                handle.attrs["qc_clean_tissue_value"] = qc_clean_tissue_value
                handle.attrs["qc_artifact_values"] = qc_artifact_values
                handle.attrs["qc_background_values"] = qc_background_values

            result["slides"].append(
                {
                    "slide_name": slide_name,
                    "slide_path": slide_path,
                    "patch_dir": str(patch_dir),
                    "coordinates_h5_path": str(coordinates_h5_path),
                    "patch_count": len(saved_coordinates),
                    "qc_mask_path": str(qc_mask_path),
                    "level": 0,
                    "coordinate_space": "target_mpp",
                    "target_mpp": target_mpp,
                    "source_mpp_x": source_mpp_x,
                    "source_mpp_y": source_mpp_y,
                    "source_patch_width": source_width,
                    "source_patch_height": source_height,
                    "min_clean_tissue_fraction": min_clean_tissue_fraction,
                    "max_artifact_fraction": max_artifact_fraction,
                    "qc_clean_tissue_value": qc_clean_tissue_value,
                }
            )
        except Exception as exc:
            result["errors"].append(
                f"Failed to extract patches for '{slide_name}': {type(exc).__name__}: {exc}"
            )
    return result
