from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.tool_utils import (
    make_tool_result,
    quiet_tool_logs,
    run_json_workers,
    safe_identifier,
    split_device_requests,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_selected_patch_records(
    patch_dir: Path, coordinates_h5_path: Path
) -> list[tuple[Path, list[int]]]:
    import h5py
    import numpy as np

    with h5py.File(coordinates_h5_path, "r") as handle:
        coordinates = np.asarray(handle["coordinates"], dtype=np.int64)
    records = [
        (patch_dir / f"{target_y}_{target_x}.png", [int(target_x), int(target_y)])
        for target_x, target_y in coordinates
    ]
    missing = [str(path) for path, _ in records if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Selected WSI patch is missing: {missing[0]}")
    return records


def collate_patch_batch(batch: list[tuple[Path, list[int]]]) -> tuple[Any, list[list[int]]]:
    import numpy as np
    from PIL import Image, PngImagePlugin
    from torchvision import transforms

    PngImagePlugin.MAX_TEXT_CHUNK = max(PngImagePlugin.MAX_TEXT_CHUNK, 256 * 1024 * 1024)
    PngImagePlugin.MAX_TEXT_MEMORY = max(PngImagePlugin.MAX_TEXT_MEMORY, 1024 * 1024 * 1024)
    transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    arrays = []
    coordinates = []
    for image_path, coordinate in batch:
        with Image.open(image_path) as image:
            arrays.append(transform(image.convert("RGB")).numpy())
        coordinates.append(coordinate)
    return np.stack(arrays), coordinates


def load_gigapath_models(shared_config: Mapping[str, Any]) -> dict[str, Any]:
    import timm
    import torch

    torch.multiprocessing.set_sharing_strategy("file_system")
    requested_device = str(shared_config["device"]).lower()
    if requested_device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif requested_device == "cpu":
        device = torch.device("cpu")
    elif requested_device == "cuda":
        device = torch.device("cuda:0")
    elif requested_device.startswith("cuda:"):
        device = torch.device(requested_device)
    else:
        raise ValueError(f"Unsupported device for wsi_embeddings: {requested_device}")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device was requested but CUDA is unavailable: {device}")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index is unavailable: {device}, "
                f"available_count={torch.cuda.device_count()}"
            )

    with quiet_tool_logs():
        try:
            from gigapath.torchscale.component.flash_attention import flash_attn_func

            flash_attention_available = flash_attn_func is not None
            flash_attention_error = (
                "" if flash_attention_available else "flash_attn_func is None"
            )
        except Exception as exc:
            flash_attention_available = False
            flash_attention_error = f"{type(exc).__name__}: {exc}"

    config_path = Path(str(shared_config["tile_encoder_config_path"])).expanduser()
    checkpoint_path = Path(
        str(shared_config["tile_encoder_checkpoint_path"])
    ).expanduser()
    model_config = json.loads(config_path.read_text(encoding="utf-8"))
    model_args = dict(model_config.get("model_args") or {})
    model_args.setdefault("num_classes", int(model_config.get("num_classes") or 0))
    tile_encoder = timm.create_model(
        str(model_config.get("architecture") or "vit_giant_patch14_dinov2"),
        pretrained=False,
        **model_args,
    )
    state_dict = torch.load(str(checkpoint_path), map_location="cpu")
    missing_keys, unexpected_keys = tile_encoder.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Tile encoder checkpoint mismatch: "
            f"missing={len(missing_keys)}, unexpected={len(unexpected_keys)}"
        )
    tile_encoder.eval().to(device)
    return {
        "torch": torch,
        "device": device,
        "use_autocast": device.type == "cuda",
        "tile_encoder": tile_encoder,
        "slide_encoder": None,
        "tile_embedding_dim": None,
        "slide_encoder_checkpoint_path": str(
            Path(str(shared_config["slide_encoder_checkpoint_path"])).expanduser()
        ),
        "slide_encoder_arch": str(shared_config["slide_encoder_arch"]),
        "global_pool": bool(shared_config.get("global_pool", True)),
        "flash_attention_available": flash_attention_available,
        "flash_attention_error": flash_attention_error,
    }


def run_gigapath_slide(
    run_config: Mapping[str, Any], models: dict[str, Any]
) -> dict[str, Any]:
    import time
    from contextlib import nullcontext

    import numpy as np
    from torch.utils.data import DataLoader

    torch = models["torch"]
    device = models["device"]
    output_dir = Path(str(run_config["output_dir"])).expanduser()
    patch_dir = Path(str(run_config["patch_dir"])).expanduser()
    coordinates_h5_path = Path(
        str(run_config["tumor_coordinates_h5_path"])
    ).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    tile_embeddings_path = output_dir / "tile_embeddings.pt"
    slide_embedding_path = output_dir / "slide_embedding.pt"
    slide_embedding_npy_path = output_dir / "slide_embedding.npy"
    for legacy_name in ("tile_metadata.pt", "slide_layer_embeddings.pt", "metrics.json"):
        legacy_path = output_dir / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()

    started_at = time.time()
    dataset = load_selected_patch_records(patch_dir, coordinates_h5_path)
    patch_count = len(dataset)
    if not patch_count:
        raise ValueError("No tumor patch coordinates were selected")
    print(
        f"[wsi_embeddings] {run_config['case_id']}: {patch_count} patches, device={device}",
        flush=True,
    )
    max_tiles = max(1, int(run_config["max_tiles_without_flash_attention"]))
    if not models["flash_attention_available"] and patch_count > max_tiles:
        raise RuntimeError(
            "GigaPath slide encoder requires flash attention for this slide: "
            f"patch_count={patch_count}, max_tiles_without_flash={max_tiles}, "
            f"flash_attention_error={models['flash_attention_error']}"
        )

    batch_size = max(1, int(run_config["batch_size"]))
    loader_options = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": max(0, int(run_config["num_workers"])),
        "pin_memory": bool(run_config["pin_memory"]) and device.type == "cuda",
        "collate_fn": collate_patch_batch,
    }
    if loader_options["num_workers"] > 0:
        loader_options["persistent_workers"] = True
        loader_options["prefetch_factor"] = max(1, int(run_config["prefetch_factor"]))
    loader = DataLoader(dataset, **loader_options)
    try:
        from tqdm.auto import tqdm

        iterator = tqdm(loader, desc=f"[wsi_embeddings] {run_config['case_id']}", unit="batch")
    except Exception:
        iterator = loader

    embeddings = []
    coordinate_rows = []
    with torch.no_grad():
        for batch_array, batch_coordinates in iterator:
            coordinate_rows.extend(batch_coordinates)
            with (
                torch.cuda.amp.autocast(dtype=torch.float16)
                if models["use_autocast"]
                else nullcontext()
            ):
                batch_tensor = torch.from_numpy(batch_array).to(
                    device, non_blocking=loader_options["pin_memory"]
                )
                embeddings.append(models["tile_encoder"](batch_tensor).detach().cpu())
    tile_embeddings = torch.cat(embeddings, dim=0)
    coordinates = torch.tensor(coordinate_rows, dtype=torch.float32)
    torch.save(tile_embeddings, tile_embeddings_path)

    embedding_dim = int(tile_embeddings.shape[1])
    if models["slide_encoder"] is None:
        with quiet_tool_logs():
            import gigapath.slide_encoder as slide_encoder_module

            models["slide_encoder"] = slide_encoder_module.create_model(
                models["slide_encoder_checkpoint_path"],
                models["slide_encoder_arch"],
                embedding_dim,
                global_pool=bool(models["global_pool"]),
            )
        models["slide_encoder"].eval().to(device)
        models["tile_embedding_dim"] = embedding_dim
    elif embedding_dim != models["tile_embedding_dim"]:
        raise RuntimeError(
            f"Tile embedding dimension changed from {models['tile_embedding_dim']} to {embedding_dim}."
        )

    actual_global_pool = getattr(models["slide_encoder"], "global_pool", None)
    if actual_global_pool is not True:
        raise RuntimeError(
            "Prov-GigaPath slide encoder must use global_pool=True "
            f"for slide embedding extraction, got {actual_global_pool!r}"
        )

    with quiet_tool_logs(), torch.no_grad():
        with (
            torch.cuda.amp.autocast(dtype=torch.float16)
            if models["use_autocast"]
            else nullcontext()
        ):
            outputs = models["slide_encoder"](
                tile_embeddings.unsqueeze(0).to(device),
                coordinates.unsqueeze(0).to(device),
                all_layer_embed=False,
            )
    slide_embedding = [item.squeeze(0).detach().cpu() for item in outputs][-1]
    torch.save(slide_embedding, slide_embedding_path)
    np.save(slide_embedding_npy_path, slide_embedding.numpy())

    elapsed_seconds = round(time.time() - started_at, 3)
    metrics_payload = {
        "case_id": str(run_config["case_id"]),
        "slide_name": str(run_config.get("slide_name") or patch_dir.name),
        "slide_path": str(run_config.get("slide_path") or ""),
        "patch_dir": str(patch_dir),
        "patch_count": patch_count,
        "tile_embedding_dim": embedding_dim,
        "slide_embedding_dim": int(slide_embedding.shape[-1]),
        "flash_attention_available": models["flash_attention_available"],
        "slide_embedding_mode": "gigapath_slide_encoder_global_pool",
        "global_pool": True,
        "device": str(device),
        "batch_size": batch_size,
        "num_workers": loader_options["num_workers"],
        "pin_memory": loader_options["pin_memory"],
        "prefetch_factor": loader_options.get("prefetch_factor", 0),
        "elapsed_seconds": elapsed_seconds,
        "issues": [],
    }
    return {
        "metrics": {
            "returncode": 0,
            "patch_count": patch_count,
            "tile_embedding_dim": embedding_dim,
            "slide_embedding_dim": int(slide_embedding.shape[-1]),
            "global_pool": True,
            "elapsed_seconds": elapsed_seconds,
        },
        "artifacts": {
            "patch_dir": str(patch_dir),
            "tumor_coordinates_h5_path": str(coordinates_h5_path),
            "tile_embeddings_path": str(tile_embeddings_path),
            "slide_embedding_path": str(slide_embedding_path),
            "slide_embedding_npy_path": str(slide_embedding_npy_path),
        },
        "payload": metrics_payload,
    }


def run_embedding_worker(payload: dict[str, Any]) -> int:
    models = load_gigapath_models(dict(payload["shared"]))
    for run_value in list(payload["runs"]):
        run_config = dict(run_value)
        case_id = str(run_config["case_id"])
        try:
            slide_result = run_gigapath_slide(run_config, models)
            make_tool_result(
                output_root=str(run_config["output_root"]),
                tool_name="wsi_embeddings",
                status="success",
                identifier=case_id,
                metrics=slide_result["metrics"],
                artifacts=slide_result["artifacts"],
                provenance={"backend": "prov-gigapath", "case_id": case_id},
                payload={
                    **slide_result["payload"],
                    "cache_signature": str(run_config.get("cache_signature") or ""),
                    "semantic_cache_version": 3,
                },
            )
        except Exception as exc:
            make_tool_result(
                output_root=str(run_config["output_root"]),
                tool_name="wsi_embeddings",
                status="failure",
                identifier=case_id,
                metrics={"returncode": 1},
                artifacts={},
                provenance={"backend": "prov-gigapath", "case_id": case_id},
                errors=[f"{type(exc).__name__}: {exc}"],
                payload={
                    "slide_path": str(run_config.get("slide_path") or ""),
                    "cache_signature": str(run_config.get("cache_signature") or ""),
                },
            )
        finally:
            if models["device"].type == "cuda":
                models["torch"].cuda.empty_cache()
    return 0


def run_wsi_embeddings_cohort(
    requests: list[Mapping[str, Any]], output_root: str, config_dir: str = ""
) -> dict[str, dict[str, Any]]:
    import yaml

    if not requests:
        return {}
    config = yaml.safe_load(
        (Path(config_dir).expanduser() / "wsi_embeddings.yaml").read_text(
            encoding="utf-8"
        )
    )
    python_executable = Path(str(config["python_executable"])).expanduser()
    code_root = Path(str(config["prov_gigapath_code_root"])).expanduser()
    if not python_executable.exists():
        raise FileNotFoundError(f"GigaPath python environment was not found: {python_executable}")
    if not code_root.exists():
        raise FileNotFoundError(f"Prov-GigaPath code root does not exist: {code_root}")
    checkpoint_dir = code_root / "checkpoints"
    shared = {
        "tile_encoder_config_path": str(checkpoint_dir / "config.json"),
        "tile_encoder_checkpoint_path": str(checkpoint_dir / "pytorch_model.bin"),
        "slide_encoder_checkpoint_path": str(checkpoint_dir / "slide_encoder.pth"),
        "slide_encoder_arch": str(config["slide_encoder_arch"]),
        "global_pool": bool(config.get("global_pool", True)),
    }
    runs = []
    for request_value in requests:
        request = dict(request_value)
        case_id = str(request["case_id"])
        runs.append(
            {
                **request,
                "output_root": output_root,
                "output_dir": str(Path(output_root) / "wsi_embeddings" / safe_identifier(case_id)),
                "batch_size": int(config["batch_size"]),
                "max_tiles_without_flash_attention": int(config["max_tiles_without_flash_attention"]),
                "num_workers": int(config["num_workers"]),
                "pin_memory": bool(config["pin_memory"]),
                "prefetch_factor": int(config["prefetch_factor"]),
            }
        )

    env = dict(os.environ)
    python_paths = [str(PROJECT_ROOT), str(code_root)]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    env["PYTHONUNBUFFERED"] = "1"
    devices = list(config["devices"])
    assignments, visible_devices = split_device_requests(
        runs,
        devices,
        workers_per_device=int(config["workers_per_gpu"]),
    )
    if visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = visible_devices
    returncodes = run_json_workers(
        [str(python_executable), str(Path(__file__).resolve()), "--internal-run"],
        [
            {
                "shared": {**shared, "device": assignment["device"]},
                "runs": assignment["requests"],
            }
            for assignment in assignments
        ],
        env=env,
        cwd=str(PROJECT_ROOT),
    )
    if any(returncode != 0 for returncode in returncodes):
        raise RuntimeError(
            f"Prov-GigaPath cohort workers failed: returncodes={returncodes}"
        )

    results = {}
    for request in requests:
        case_id = str(request["case_id"])
        snapshot_path = Path(output_root) / "wsi_embeddings" / f"{safe_identifier(case_id)}.json"
        results[case_id] = dict(
            json.loads(snapshot_path.read_text(encoding="utf-8"))["tool_result"]
        )
    return results


def run_wsi_embeddings(
    *,
    case_id: str,
    wsi_record: Mapping[str, Any] | None,
    output_root: str,
    patch_dir: str = "",
    tumor_coordinates_h5_path: str = "",
    config_dir: str = "",
) -> dict[str, Any]:
    record = dict(wsi_record or {})
    slide_path = str(record.get("File Path", "") or "").strip()
    patch_path = Path(patch_dir).expanduser() if patch_dir else Path(output_root) / "wsi_patch" / case_id
    request = {
        "case_id": case_id,
        "slide_name": Path(slide_path).stem if slide_path else patch_path.name,
        "slide_path": slide_path,
        "patch_dir": str(patch_path),
        "tumor_coordinates_h5_path": str(tumor_coordinates_h5_path),
        "cache_signature": "",
    }
    return run_wsi_embeddings_cohort(
        [request], output_root, config_dir
    )[case_id]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Prov-GigaPath WSI embeddings.")
    parser.add_argument("--internal-run", action="store_true")
    if not parser.parse_args().internal_run:
        raise SystemExit("Use tools.wsi_embeddings.run_wsi_embeddings(...) from Python.")
    raise SystemExit(run_embedding_worker(json.loads(sys.stdin.read())))
