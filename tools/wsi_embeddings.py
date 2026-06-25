from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.tool_utils import quiet_tool_logs


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def collate_patch_batch(batch: list[tuple[Path, list[int]]]) -> tuple[Any, list[list[int]]]:
    import numpy as np
    from PIL import Image
    from PIL import PngImagePlugin
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
    batch_arrays = []
    batch_coordinates = []
    for image_path, coordinate in batch:
        with Image.open(image_path) as image:
            batch_arrays.append(transform(image.convert("RGB")).numpy())
        batch_coordinates.append(coordinate)
    return np.stack(batch_arrays), batch_coordinates


def run_wsi_embeddings(
    *,
    case_id: str,
    wsi_record: Mapping[str, Any] | None,
    output_root: str,
    patch_dir: str = "",
    config_dir: str = "",
) -> dict[str, Any]:
    import yaml

    from utils.tool_utils import make_tool_result

    tool_config = yaml.safe_load((Path(config_dir).expanduser() / "wsi_embeddings.yaml").read_text(encoding="utf-8")) or {}
    python_executable = Path(str(tool_config["python_executable"])).expanduser()
    prov_gigapath_code_root = Path(
        str(tool_config["prov_gigapath_code_root"])
    ).expanduser()
    checkpoint_dir = prov_gigapath_code_root / "checkpoints"
    slide_encoder_arch = str(tool_config["slide_encoder_arch"])
    batch_size = int(tool_config["batch_size"])
    device = str(tool_config["device"])

    record = dict(wsi_record or {})
    slide_path = str(record.get("File Path", "") or "").strip()
    slide_name = Path(slide_path).stem if slide_path else ""
    if not slide_name and patch_dir:
        slide_name = Path(patch_dir).name
    patch_dir_path = (
        Path(patch_dir).expanduser()
        if patch_dir
        else Path(output_root) / "wsi_patch" / case_id
    )

    case_output_dir = Path(output_root) / "wsi_embeddings" / case_id
    case_output_dir.mkdir(parents=True, exist_ok=True)

    tile_embeddings_path = case_output_dir / "tile_embeddings.pt"
    slide_embedding_path = case_output_dir / "slide_embedding.pt"
    slide_embedding_npy_path = case_output_dir / "slide_embedding.npy"
    artifacts = {}
    provenance = {"backend": "prov-gigapath", "case_id": case_id}

    if not python_executable.exists():
        return make_tool_result(
            output_root=output_root,
            tool_name="wsi_embeddings",
            status="failure",
            identifier=case_id,
            metrics={},
            artifacts=artifacts,
            provenance=provenance,
            errors=[f"GigaPath python environment was not found: {python_executable}"],
        )
    if not prov_gigapath_code_root.exists():
        return make_tool_result(
            output_root=output_root,
            tool_name="wsi_embeddings",
            status="failure",
            identifier=case_id,
            metrics={},
            artifacts=artifacts,
            provenance=provenance,
            errors=[
                f"Prov-GigaPath code root does not exist: {prov_gigapath_code_root}"
            ],
        )
    if not patch_dir_path.exists():
        return make_tool_result(
            output_root=output_root,
            tool_name="wsi_embeddings",
            status="failure",
            identifier=case_id,
            metrics={},
            artifacts=artifacts,
            provenance=provenance,
            errors=[f"WSI patch directory does not exist: {patch_dir_path}"],
        )

    run_config = {
        "case_id": case_id,
        "slide_name": slide_name,
        "slide_path": slide_path,
        "patch_dir": str(patch_dir_path),
        "output_dir": str(case_output_dir),
        "batch_size": batch_size,
        "device": device,
        "tile_encoder_config_path": str(checkpoint_dir / "config.json"),
        "tile_encoder_checkpoint_path": str(checkpoint_dir / "pytorch_model.bin"),
        "slide_encoder_checkpoint_path": str(checkpoint_dir / "slide_encoder.pth"),
        "slide_encoder_arch": slide_encoder_arch,
        "max_tiles_without_flash_attention": int(
            tool_config["max_tiles_without_flash_attention"]
        ),
        "num_workers": int(tool_config["num_workers"]),
        "pin_memory": bool(tool_config["pin_memory"]),
        "prefetch_factor": int(tool_config["prefetch_factor"]),
    }
    env = dict(os.environ)
    python_path_items = [str(PROJECT_ROOT), str(prov_gigapath_code_root)]
    if env.get("PYTHONPATH"):
        python_path_items.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path_items)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    with tempfile.TemporaryDirectory(
        prefix=f"wsi_embeddings_{case_id}_", dir="/tmp"
    ) as temp_dir:
        metrics_path = Path(temp_dir) / "metrics.json"
        run_config["metrics_path"] = str(metrics_path)
        completed = subprocess.run(
            [str(python_executable), str(Path(__file__).resolve()), "--internal-run"],
            env=env,
            cwd=str(PROJECT_ROOT),
            input=json.dumps(run_config, ensure_ascii=False),
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            return make_tool_result(
                output_root=output_root,
                tool_name="wsi_embeddings",
                status="failure",
                identifier=case_id,
                metrics={"returncode": int(completed.returncode)},
                artifacts=artifacts,
                provenance=provenance,
                errors=[
                    "Prov-GigaPath embedding extraction failed with "
                    f"return code {completed.returncode}."
                ],
            )
        if not metrics_path.exists():
            return make_tool_result(
                output_root=output_root,
                tool_name="wsi_embeddings",
                status="failure",
                identifier=case_id,
                metrics={"returncode": int(completed.returncode)},
                artifacts=artifacts,
                provenance=provenance,
                errors=["Prov-GigaPath finished without writing metrics.json."],
            )
        metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))

    issues = [str(item) for item in list(metrics_payload.get("issues") or [])]
    slide_embedding_mode = str(metrics_payload.get("slide_embedding_mode") or "")
    tool_artifacts = {
        "patch_dir": str(patch_dir_path),
        "tile_embeddings_path": str(tile_embeddings_path),
        "slide_embedding_path": str(slide_embedding_path),
        "slide_embedding_npy_path": str(slide_embedding_npy_path),
    }
    if issues or slide_embedding_mode != "gigapath_slide_encoder":
        failure_errors = list(issues)
        if not failure_errors and slide_embedding_mode != "gigapath_slide_encoder":
            failure_errors.append(
                f"Unexpected slide embedding mode: {slide_embedding_mode or 'unknown'}."
            )
        return make_tool_result(
            output_root=output_root,
            tool_name="wsi_embeddings",
            status="failure",
            identifier=case_id,
            metrics={
                "returncode": int(completed.returncode),
                "patch_count": int(metrics_payload.get("patch_count") or 0),
                "tile_embedding_dim": int(
                    metrics_payload.get("tile_embedding_dim") or 0
                ),
                "slide_embedding_dim": int(
                    metrics_payload.get("slide_embedding_dim") or 0
                ),
                "elapsed_seconds": float(
                    metrics_payload.get("elapsed_seconds") or 0.0
                ),
            },
            artifacts=tool_artifacts,
            provenance=provenance,
            errors=failure_errors,
            payload=metrics_payload,
        )

    return make_tool_result(
        output_root=output_root,
        tool_name="wsi_embeddings",
        status="success",
        identifier=case_id,
        metrics={
            "returncode": int(completed.returncode),
            "patch_count": int(metrics_payload.get("patch_count") or 0),
            "tile_embedding_dim": int(metrics_payload.get("tile_embedding_dim") or 0),
            "slide_embedding_dim": int(metrics_payload.get("slide_embedding_dim") or 0),
            "elapsed_seconds": float(metrics_payload.get("elapsed_seconds") or 0.0),
        },
        artifacts=tool_artifacts,
        provenance=provenance,
        errors=[],
        payload=metrics_payload,
    )


def run_embedding_worker(run_config: dict[str, Any]) -> int:
    import time
    from contextlib import nullcontext

    import numpy as np
    import timm
    import torch
    from torch.utils.data import DataLoader

    torch.multiprocessing.set_sharing_strategy("file_system")

    output_dir = Path(str(run_config["output_dir"])).expanduser()
    patch_dir = Path(str(run_config["patch_dir"])).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    tile_embeddings_path = output_dir / "tile_embeddings.pt"
    slide_embedding_path = output_dir / "slide_embedding.pt"
    slide_embedding_npy_path = output_dir / "slide_embedding.npy"
    metrics_path = Path(str(run_config["metrics_path"])).expanduser()
    tile_encoder_config_path = Path(
        str(run_config["tile_encoder_config_path"])
    ).expanduser()
    tile_encoder_checkpoint_path = Path(
        str(run_config["tile_encoder_checkpoint_path"])
    ).expanduser()
    slide_encoder_checkpoint_path = Path(
        str(run_config["slide_encoder_checkpoint_path"])
    ).expanduser()
    slide_encoder_arch = str(run_config["slide_encoder_arch"])
    for legacy_name in (
        "tile_metadata.pt",
        "slide_layer_embeddings.pt",
        "metrics.json",
    ):
        legacy_path = output_dir / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()

    started_at = time.time()
    requested_device = str(run_config["device"]).lower()
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
            raise RuntimeError(
                f"CUDA device was requested but CUDA is unavailable: {device}"
            )
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index is unavailable: {device}, "
                f"available_count={torch.cuda.device_count()}"
            )
    use_autocast = device.type == "cuda"
    image_paths = sorted(
        [path for path in patch_dir.glob("*.png")],
        key=lambda path: tuple(int(value) for value in path.stem.split("_")),
    )
    if not image_paths:
        raise FileNotFoundError(f"No patch png files were found under {patch_dir}")
    print(
        f"[wsi_embeddings] {run_config.get('case_id', '')}: {len(image_paths)} patches, device={device}",
        flush=True,
    )

    patch_count = len(image_paths)
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

    max_tiles_without_flash = max(1, int(run_config["max_tiles_without_flash_attention"]))
    if not flash_attention_available and patch_count > max_tiles_without_flash:
        raise RuntimeError(
            "GigaPath slide encoder requires flash attention for this slide: "
            f"patch_count={patch_count}, max_tiles_without_flash={max_tiles_without_flash}. "
            "Please run in a CUDA-visible gigapath environment with flash-attn installed. "
            f"device={device}, torch_cuda_available={torch.cuda.is_available()}, "
            f"flash_attention_error={flash_attention_error}"
        )

    batch_size = max(1, int(run_config["batch_size"]))
    model_config = json.loads(tile_encoder_config_path.read_text(encoding="utf-8"))
    model_args = dict(model_config.get("model_args") or {})
    if "num_classes" not in model_args:
        model_args["num_classes"] = int(model_config.get("num_classes") or 0)
    tile_encoder = timm.create_model(
        str(model_config.get("architecture") or "vit_giant_patch14_dinov2"),
        pretrained=False,
        **model_args,
    )
    state_dict = torch.load(str(tile_encoder_checkpoint_path), map_location="cpu")
    missing_keys, unexpected_keys = tile_encoder.load_state_dict(
        state_dict, strict=False
    )
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Tile encoder checkpoint mismatch: "
            f"missing={len(missing_keys)}, unexpected={len(unexpected_keys)}"
        )
    tile_encoder.eval().to(device)

    all_embeddings: list[torch.Tensor] = []
    coordinate_rows: list[list[int]] = []
    dataset = [
        (
            image_path,
            [int(image_path.stem.split("_")[1]), int(image_path.stem.split("_")[0])],
        )
        for image_path in image_paths
    ]

    data_loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": max(0, int(run_config["num_workers"])),
        "pin_memory": bool(run_config["pin_memory"]) and device.type == "cuda",
        "collate_fn": collate_patch_batch,
    }
    if data_loader_kwargs["num_workers"] > 0:
        data_loader_kwargs["persistent_workers"] = True
        data_loader_kwargs["prefetch_factor"] = max(1, int(run_config["prefetch_factor"]))
    data_loader = DataLoader(dataset, **data_loader_kwargs)
    try:
        from tqdm.auto import tqdm

        iterator = tqdm(
            data_loader,
            desc=f"[wsi_embeddings] {run_config.get('case_id', '')}",
            unit="batch",
        )
    except Exception:
        iterator = data_loader
    with torch.no_grad():
        for batch_array, batch_coordinates in iterator:
            coordinate_rows.extend(batch_coordinates)
            with (
                torch.cuda.amp.autocast(dtype=torch.float16)
                if use_autocast
                else nullcontext()
            ):
                batch_tensor = torch.from_numpy(batch_array).to(device, non_blocking=data_loader_kwargs["pin_memory"])
                all_embeddings.append(tile_encoder(batch_tensor).detach().cpu())
    tile_embeddings = torch.cat(all_embeddings, dim=0)
    coordinates = torch.tensor(coordinate_rows, dtype=torch.float32)
    torch.save(tile_embeddings, tile_embeddings_path)
    print(f"[wsi_embeddings] saved {tile_embeddings_path}", flush=True)

    with quiet_tool_logs():
        import gigapath.slide_encoder as slide_encoder_module

        slide_encoder = slide_encoder_module.create_model(
            str(slide_encoder_checkpoint_path),
            slide_encoder_arch,
            int(tile_embeddings.shape[1]),
        )
    slide_encoder.eval().to(device)
    with quiet_tool_logs():
        with torch.no_grad():
            with (
                torch.cuda.amp.autocast(dtype=torch.float16)
                if use_autocast
                else nullcontext()
            ):
                outputs = slide_encoder(
                    tile_embeddings.unsqueeze(0).to(device),
                    coordinates.unsqueeze(0).to(device),
                    all_layer_embed=False,
                )
    slide_embedding = [item.squeeze(0).detach().cpu() for item in outputs][-1]
    slide_embedding_mode = "gigapath_slide_encoder"

    torch.save(slide_embedding, slide_embedding_path)
    np.save(slide_embedding_npy_path, slide_embedding.detach().cpu().numpy())
    print(f"[wsi_embeddings] saved {slide_embedding_path}", flush=True)
    metrics_payload = {
        "case_id": str(run_config.get("case_id") or ""),
        "slide_name": str(run_config.get("slide_name") or patch_dir.name),
        "slide_path": str(run_config.get("slide_path") or ""),
        "patch_dir": str(patch_dir),
        "patch_count": patch_count,
        "tile_embedding_dim": int(tile_embeddings.shape[1]),
        "slide_embedding_dim": int(slide_embedding.shape[-1]),
        "flash_attention_available": flash_attention_available,
        "slide_embedding_mode": slide_embedding_mode,
        "device": str(device),
        "batch_size": batch_size,
        "num_workers": data_loader_kwargs["num_workers"],
        "pin_memory": data_loader_kwargs["pin_memory"],
        "prefetch_factor": data_loader_kwargs.get("prefetch_factor", 0),
        "elapsed_seconds": round(time.time() - started_at, 3),
        "issues": [],
    }
    metrics_path.write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Prov-GigaPath WSI embeddings.")
    parser.add_argument("--internal-run", action="store_true")
    if not parser.parse_args().internal_run:
        raise SystemExit(
            "Use tools.wsi_embeddings.run_wsi_embeddings(...) from Python."
        )
    raise SystemExit(run_embedding_worker(json.loads(sys.stdin.read())))
