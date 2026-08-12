from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

import h5py
import matplotlib.pyplot as plt
import numpy as np
from openslide import OpenSlide
from PIL import Image, ImageDraw, ImageFont

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.tool_utils import (
    make_tool_result,
    run_json_workers,
    safe_identifier,
    split_device_requests,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLASS_COLORS = [
    (228, 26, 28),
    (210, 210, 210),
    (140, 45, 4),
    (77, 175, 74),
    (255, 217, 47),
    (55, 126, 184),
    (255, 127, 0),
    (0, 0, 0),
    (152, 78, 163),
    (0, 191, 196),
]


def select_tumor_indices(
    probabilities: np.ndarray, *, tumor_index: int, threshold: float
) -> np.ndarray:
    return np.flatnonzero(probabilities[:, tumor_index] > threshold)


def save_tumor_coordinates(
    output_path: Path,
    coordinates: np.ndarray,
    source_coordinates: np.ndarray,
    selected_indices: np.ndarray,
    attributes: Mapping[str, Any],
) -> None:
    with h5py.File(output_path, "w") as handle:
        handle.create_dataset("coordinates", data=coordinates[selected_indices])
        handle.create_dataset(
            "source_coordinates", data=source_coordinates[selected_indices]
        )
        for key, value in attributes.items():
            handle.attrs[key] = value


def read_context(
    slide: OpenSlide,
    center_x: float,
    center_y: float,
    source_width: int,
    source_height: int,
    tile_size: int,
) -> np.ndarray:
    slide_width, slide_height = slide.dimensions
    x = min(
        max(int(round(center_x - source_width / 2)), 0),
        max(slide_width - source_width, 0),
    )
    y = min(
        max(int(round(center_y - source_height / 2)), 0),
        max(slide_height - source_height, 0),
    )
    image = slide.read_region((x, y), 0, (source_width, source_height)).convert(
        "RGB"
    )
    if image.size != (tile_size, tile_size):
        image = image.resize((tile_size, tile_size), Image.Resampling.BICUBIC)
    return np.asarray(image)


def load_pharaoh(shared: Mapping[str, Any]) -> dict[str, Any]:
    import onnxruntime as ort

    ort.preload_dlls()
    if not str(shared["device"]).lower().startswith("cuda"):
        raise ValueError("wsi_tumor_seg requires a CUDA device")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("ONNX Runtime CUDAExecutionProvider is unavailable")
    config = json.loads(
        Path(str(shared["inference_config_path"])).read_text(encoding="utf-8")
    )
    session = ort.InferenceSession(
        str(shared["model_path"]),
        providers=[
            (
                "CUDAExecutionProvider",
                {"device_id": int(shared["provider_device_id"])},
            )
        ],
    )
    if session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError("PHARAOH did not initialize on CUDA")
    classes = list(config["classes"])
    return {
        "session": session,
        "classes": classes,
        "tumor_index": classes.index("Clear Cell Renal Cell Carcinoma"),
        "tile_size": int(config["tile_size"]),
        "mpp": float(config["mpp"]),
    }


def save_maps(
    *,
    slide: OpenSlide,
    output_dir: Path,
    centers: np.ndarray,
    source_coordinates: np.ndarray,
    source_patch_width: int,
    source_patch_height: int,
    tumor_probabilities: np.ndarray,
    predicted_indices: np.ndarray,
    classes: list[str],
    overlay_max_size: int,
) -> None:
    thumbnail = slide.get_thumbnail((overlay_max_size, overlay_max_size)).convert(
        "RGB"
    )
    scale_x = thumbnail.width / slide.dimensions[0]
    scale_y = thumbnail.height / slide.dimensions[1]

    figure, axis = plt.subplots(figsize=(12, 10))
    axis.imshow(thumbnail)
    points = axis.scatter(
        centers[:, 0] * scale_x,
        centers[:, 1] * scale_y,
        c=tumor_probabilities,
        cmap="turbo",
        vmin=0,
        vmax=1,
        s=8,
        alpha=0.8,
    )
    axis.set_axis_off()
    axis.set_title("PHARAOH ccRCC probability")
    figure.colorbar(points, ax=axis, label="P(ccRCC)")
    figure.tight_layout()
    figure.savefig(output_dir / "tumor_probability_map.png", dpi=200)
    plt.close(figure)

    overlay = Image.new("RGBA", thumbnail.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for (source_x, source_y), class_index in zip(
        source_coordinates, predicted_indices
    ):
        color = CLASS_COLORS[int(class_index)]
        draw.rectangle(
            (
                int(round(source_x * scale_x)),
                int(round(source_y * scale_y)),
                int(round((source_x + source_patch_width) * scale_x)),
                int(round((source_y + source_patch_height) * scale_y)),
            ),
            fill=(*color, 115),
        )
    classified_thumbnail = Image.alpha_composite(thumbnail.convert("RGBA"), overlay)
    legend_width = 440
    result = Image.new(
        "RGBA",
        (thumbnail.width + legend_width, thumbnail.height),
        (255, 255, 255, 255),
    )
    result.paste(classified_thumbnail, (0, 0))
    result_draw = ImageDraw.Draw(result)
    font = ImageFont.truetype("DejaVuSans.ttf", 18)
    title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
    result_draw.text(
        (thumbnail.width + 24, 28),
        "PHARAOH predicted classes",
        fill=(0, 0, 0, 255),
        font=title_font,
    )
    for index, (name, color) in enumerate(zip(classes, CLASS_COLORS)):
        y = 82 + index * 48
        result_draw.rectangle(
            (thumbnail.width + 24, y, thumbnail.width + 52, y + 28),
            fill=(*color, 255),
            outline=(0, 0, 0, 255),
        )
        count = int(np.count_nonzero(predicted_indices == index))
        result_draw.text(
            (thumbnail.width + 66, y + 3),
            f"{name} ({count})",
            fill=(0, 0, 0, 255),
            font=font,
        )
    result.convert("RGB").save(output_dir / "overlay.png")


def run_case(run_config: Mapping[str, Any], shared: Mapping[str, Any], model) -> dict[str, Any]:
    case_id = str(run_config["case_id"])
    output_root = str(run_config["output_root"])
    output_dir = Path(output_root) / "wsi_tumor_seg" / safe_identifier(case_id)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    with h5py.File(str(run_config["coordinates_h5_path"]), "r") as handle:
        coordinates = np.asarray(handle["coordinates"], dtype=np.int64)
        source_coordinates = np.asarray(handle["source_coordinates"], dtype=np.int64)
        coordinate_attributes = dict(handle.attrs)
        source_mpp_x = float(handle.attrs["source_mpp_x"])
        source_mpp_y = float(handle.attrs["source_mpp_y"])
        source_patch_width = int(handle.attrs["source_patch_width"])
        source_patch_height = int(handle.attrs["source_patch_height"])
    if not len(coordinates) or len(coordinates) != len(source_coordinates):
        raise ValueError("WSI patch coordinates are empty or inconsistent")

    centers = source_coordinates.astype(np.float64)
    centers[:, 0] += source_patch_width / 2
    centers[:, 1] += source_patch_height / 2
    context_width = int(round(model["tile_size"] * model["mpp"] / source_mpp_x))
    context_height = int(round(model["tile_size"] * model["mpp"] / source_mpp_y))
    input_meta = model["session"].get_inputs()[0]
    channels_first = len(input_meta.shape) == 4 and input_meta.shape[1] == 3
    probabilities = []
    with OpenSlide(str(run_config["slide_path"])) as slide:
        from tqdm import tqdm

        starts = range(0, len(centers), int(shared["batch_size"]))
        for start in tqdm(
            starts,
            total=len(starts),
            desc=f"wsi_tumor_seg {case_id}",
            unit="batch",
            position=int(shared["worker_position"]),
            leave=True,
            dynamic_ncols=True,
        ):
            tiles = [
                read_context(
                    slide,
                    center_x,
                    center_y,
                    context_width,
                    context_height,
                    model["tile_size"],
                )[..., ::-1]
                for center_x, center_y in centers[
                    start : start + int(shared["batch_size"])
                ]
            ]
            batch = np.asarray(tiles, dtype=np.float32) / 255.0
            if channels_first:
                batch = batch.transpose(0, 3, 1, 2)
            probabilities.append(
                model["session"].run(None, {input_meta.name: batch})[0]
            )
        probabilities = np.concatenate(probabilities, axis=0)
        tumor_probabilities = probabilities[:, model["tumor_index"]]
        predicted_indices = probabilities.argmax(axis=1)
        selected_indices = select_tumor_indices(
            probabilities,
            tumor_index=model["tumor_index"],
            threshold=float(shared["tumor_probability_threshold"]),
        )
        selected_mask = np.zeros(len(probabilities), dtype=bool)
        selected_mask[selected_indices] = True

        tumor_coordinates_path = output_dir / "tumor_coordinates.h5"
        save_tumor_coordinates(
            tumor_coordinates_path,
            coordinates,
            source_coordinates,
            selected_indices,
            coordinate_attributes,
        )

        probability_path = output_dir / "patch_probabilities.csv"
        rows = []
        for index, ((target_x, target_y), prediction) in enumerate(
            zip(coordinates, probabilities)
        ):
            row = {
                "patch_index": index,
                "target_x": int(target_x),
                "target_y": int(target_y),
                "predicted_class": model["classes"][int(predicted_indices[index])],
                "tumor_probability": float(tumor_probabilities[index]),
                "selected_tumor": bool(selected_mask[index]),
            }
            row.update(
                {
                    f"prob_{name.lower().replace(' ', '_')}": float(value)
                    for name, value in zip(model["classes"], prediction)
                }
            )
            rows.append(row)
        with probability_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        save_maps(
            slide=slide,
            output_dir=output_dir,
            centers=centers,
            source_coordinates=source_coordinates,
            source_patch_width=source_patch_width,
            source_patch_height=source_patch_height,
            tumor_probabilities=tumor_probabilities,
            predicted_indices=predicted_indices,
            classes=model["classes"],
            overlay_max_size=int(shared["overlay_max_size"]),
        )

    class_counts = {
        name: int(np.count_nonzero(predicted_indices == index))
        for index, name in enumerate(model["classes"])
    }
    summary = {
        "case_id": case_id,
        "slide_path": str(run_config["slide_path"]),
        "patch_count": len(coordinates),
        "tumor_patch_count": len(selected_indices),
        "tumor_probability_threshold": float(shared["tumor_probability_threshold"]),
        "model_mpp": model["mpp"],
        "model_tile_size": model["tile_size"],
        "predicted_class_counts": class_counts,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "metrics": {
            "patch_count": len(coordinates),
            "tumor_patch_count": len(selected_indices),
            "tumor_probability_threshold": float(
                shared["tumor_probability_threshold"]
            ),
        },
        "artifacts": {
            "tumor_coordinates_h5_path": str(tumor_coordinates_path),
            "patch_probabilities_path": str(probability_path),
            "summary_path": str(summary_path),
            "tumor_probability_map_path": str(
                output_dir / "tumor_probability_map.png"
            ),
            "overlay_path": str(output_dir / "overlay.png"),
        },
        "payload": {
            **summary,
            "cache_signature": str(run_config.get("cache_signature") or ""),
            "semantic_cache_version": 1,
        },
    }


def run_worker(payload: dict[str, Any]) -> int:
    shared = dict(payload["shared"])
    shared["worker_position"] = int(payload["worker_position"])
    model = load_pharaoh(shared)
    for run_value in list(payload["runs"]):
        run_config = dict(run_value)
        case_id = str(run_config["case_id"])
        try:
            result = run_case(run_config, shared, model)
            make_tool_result(
                output_root=str(run_config["output_root"]),
                tool_name="wsi_tumor_seg",
                status="success",
                identifier=case_id,
                metrics=result["metrics"],
                artifacts=result["artifacts"],
                provenance={
                    "backend": "pharaoh-kidney-kirc",
                    "case_id": case_id,
                    "device": str(shared["device"]),
                },
                payload=result["payload"],
            )
        except Exception as exc:
            make_tool_result(
                output_root=str(run_config["output_root"]),
                tool_name="wsi_tumor_seg",
                status="failure",
                identifier=case_id,
                metrics={"returncode": 1},
                artifacts={},
                provenance={"backend": "pharaoh-kidney-kirc", "case_id": case_id},
                errors=[f"{type(exc).__name__}: {exc}"],
                payload={
                    "slide_path": str(run_config.get("slide_path") or ""),
                    "cache_signature": str(run_config.get("cache_signature") or ""),
                },
            )
    return 0


def run_wsi_tumor_seg_cohort(
    requests: list[Mapping[str, Any]], output_root: str, config_dir: str
) -> dict[str, dict[str, Any]]:
    import yaml

    if not requests:
        return {}
    config = yaml.safe_load(
        (Path(config_dir) / "wsi_tumor_seg.yaml").read_text(encoding="utf-8")
    )
    python_executable = Path(str(config["python_executable"])).expanduser()
    model_path = Path(str(config["model_path"])).expanduser()
    inference_config_path = Path(str(config["inference_config_path"])).expanduser()
    for path in (python_executable, model_path, inference_config_path):
        if not path.exists():
            raise FileNotFoundError(f"wsi_tumor_seg dependency is missing: {path}")
    shared = {
        "model_path": str(model_path),
        "inference_config_path": str(inference_config_path),
        "batch_size": int(config["batch_size"]),
        "tumor_probability_threshold": float(config["tumor_probability_threshold"]),
        "overlay_max_size": int(config["overlay_max_size"]),
    }
    runs = [{**dict(request), "output_root": output_root} for request in requests]
    devices = list(config["devices"])
    workers_per_gpu = int(config["workers_per_gpu"])
    assignments, visible_devices = split_device_requests(
        runs, devices, workers_per_device=workers_per_gpu
    )
    if visible_devices is None:
        raise ValueError("wsi_tumor_seg requires CUDA devices")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = visible_devices
    env["PYTHONPATH"] = os.pathsep.join(
        [str(PROJECT_ROOT), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    env["PYTHONUNBUFFERED"] = "1"
    payloads = []
    for assignment in assignments:
        payloads.append(
            {
                "shared": {
                    **shared,
                    "device": assignment["physical_device"],
                    "provider_device_id": int(
                        str(assignment["device"]).split(":", 1)[1]
                    ),
                },
                "runs": assignment["requests"],
                "worker_position": assignment["position"],
            }
        )
    returncodes = run_json_workers(
        [str(python_executable), str(Path(__file__).resolve()), "--internal-run"],
        payloads,
        env=env,
        cwd=str(PROJECT_ROOT),
    )
    if any(returncode != 0 for returncode in returncodes):
        raise RuntimeError(f"PHARAOH cohort workers failed: returncodes={returncodes}")
    results = {}
    for request in requests:
        case_id = str(request["case_id"])
        snapshot_path = (
            Path(output_root) / "wsi_tumor_seg" / f"{safe_identifier(case_id)}.json"
        )
        results[case_id] = dict(
            json.loads(snapshot_path.read_text(encoding="utf-8"))["tool_result"]
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--internal-run", action="store_true")
    if not parser.parse_args().internal_run:
        raise SystemExit("Use run_wsi_tumor_seg_cohort from Python")
    raise SystemExit(run_worker(json.loads(sys.stdin.read())))
