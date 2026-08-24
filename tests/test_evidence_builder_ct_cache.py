from __future__ import annotations

import json
from pathlib import Path

from agents import quality_control


def write_config(config_dir: Path, model_dir: Path) -> None:
    (config_dir / "ct_tumor_seg.yaml").write_text(
        "\n".join(
            [
                "backend: nnunetv2_modelfolder",
                f"model_folder: {model_dir}",
                "output_label: 2",
            ]
        ),
        encoding="utf-8",
    )


def write_snapshot(output_root: Path, case_id: str, mask_path: Path, payload: dict) -> None:
    snapshot = {
        "tool_result": {
            "tool_name": "ct_tumor_seg",
            "status": "success",
            "metrics": {"returncode": 0},
            "artifacts": {"segmentation_path": str(mask_path)},
            "provenance": {
                "backend": "nnunetv2_modelfolder",
                "model_folder": str(output_root / "model"),
                "output_label": 2,
                "case_id": case_id,
            },
            "errors": [],
        },
        "payload": payload,
    }
    (output_root / "ct_tumor_seg" / f"{case_id}.json").write_text(
        json.dumps(snapshot), encoding="utf-8"
    )


def cache_context(tmp_path: Path, monkeypatch, payload: dict) -> dict:
    case_id = "TCGA-TEST-0001"
    output_root = tmp_path / "output"
    config_dir = tmp_path / "configs"
    model_dir = output_root / "model"
    ct_path = output_root / "ct_qc" / case_id / f"{case_id}.nii.gz"
    mask_path = output_root / "ct_tumor_seg" / case_id / f"{case_id}_mask.nii.gz"
    config_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    ct_path.parent.mkdir(parents=True)
    mask_path.parent.mkdir(parents=True)
    ct_path.touch()
    mask_path.touch()
    write_config(config_dir, model_dir)
    write_snapshot(output_root, case_id, mask_path, payload)
    monkeypatch.setattr(quality_control, "nifti_geometry_matches", lambda *_: True)
    state = {
        "case_id": case_id,
        "inventory": {
            "Case_ID": case_id,
            "CT": [
                {
                    "File Path": str(ct_path),
                    "Series UID": "SERIES-1",
                    "Study UID": "STUDY-1",
                }
            ],
        },
    }
    return quality_control.ct_tumor_seg_context(
        state, output_root=str(output_root), config_dir=str(config_dir)
    )


def test_legacy_ct_segmentation_cache_without_identity_is_not_reused(tmp_path, monkeypatch):
    context = cache_context(
        tmp_path,
        monkeypatch,
        {"case_id": "TCGA-TEST-0001", "reused_existing_mask": True},
    )
    assert context["reuse_existing_mask"] is False


def test_new_ct_segmentation_cache_rejects_changed_input(tmp_path, monkeypatch):
    context = cache_context(
        tmp_path,
        monkeypatch,
        {
            "case_id": "TCGA-TEST-0001",
            "ct_identity": {"series_uid": "OTHER", "study_uid": "STUDY-1"},
        },
    )
    assert context["reuse_existing_mask"] is False


def test_ct_segmentation_cache_requires_config_signature(tmp_path, monkeypatch):
    context = cache_context(
        tmp_path,
        monkeypatch,
        {
            "case_id": "TCGA-TEST-0001",
            "ct_identity": {"series_uid": "SERIES-1", "study_uid": "STUDY-1"},
        },
    )
    assert context["reuse_existing_mask"] is False


def test_nifti_geometry_matches_size_spacing_origin_and_direction(tmp_path):
    import SimpleITK as sitk

    ct_path = tmp_path / "ct.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    ct = sitk.Image([2, 3, 4], sitk.sitkInt16)
    mask = sitk.Image([2, 3, 4], sitk.sitkUInt8)
    for image in (ct, mask):
        image.SetSpacing((1.0, 1.2, 2.0))
        image.SetOrigin((3.0, 4.0, 5.0))
        image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    sitk.WriteImage(ct, str(ct_path))
    sitk.WriteImage(mask, str(mask_path))
    assert quality_control.nifti_geometry_matches(str(ct_path), str(mask_path)) is True
    mask.SetOrigin((3.0, 4.0, 5.01))
    sitk.WriteImage(mask, str(mask_path))
    assert quality_control.nifti_geometry_matches(str(ct_path), str(mask_path)) is False


def test_nifti_geometry_uses_effective_sform_when_qform_is_stale(tmp_path):
    import nibabel as nib
    import numpy as np

    ct_path = tmp_path / "ct.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    data = np.zeros((8, 9, 10), dtype=np.uint8)
    effective_affine = np.diag([0.757812, 0.757812, 7.5, 1.0])
    stale_qform = np.diag([0.703125, 0.703125, 7.5, 1.0])

    ct = nib.Nifti1Image(data, effective_affine)
    ct.set_qform(stale_qform, code=1)
    ct.set_sform(effective_affine, code=1)
    nib.save(ct, ct_path)

    mask = nib.Nifti1Image(data, effective_affine)
    mask.set_qform(effective_affine, code=1)
    mask.set_sform(effective_affine, code=1)
    nib.save(mask, mask_path)

    assert quality_control.nifti_geometry_matches(str(ct_path), str(mask_path)) is True
