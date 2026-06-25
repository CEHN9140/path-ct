import os
from pathlib import Path

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


def wsi_tis_detect(
    path_slide: str,
    output_dir: str,
    config: dict[str, object] | None = None,
) -> None:
    import cv2
    import segmentation_models_pytorch as smp
    import torch
    from openslide import OpenSlide

    from tools.pathology_qc.wsi_tis_detect_helper_fx import (
        get_preprocessing,
        make_class_map,
    )
    config = dict(config or {})
    config_dir = str(config.get("config_dir", "") or "")
    model_dir = Path(str(config["model_dir"])).expanduser()
    if not model_dir.is_absolute():
        model_dir = (Path(config_dir).expanduser().parent / model_dir).resolve() if config_dir else model_dir.resolve()
    Image.MAX_IMAGE_PIXELS = 1000000000
    device = resolve_device(config["device"])
    model_td_dir = model_dir
    model_td_name = str(config["model_name"])
    mpp_model_td = float(config["mpp_model"])
    patch_size = int(config["patch_size"])
    encoder_model = str(config["encoder_model"])
    encoder_weights = str(config["encoder_weights"])

    overlay_image_weight = float(config["overlay_image_weight"])
    overlay_mask_weight = float(config["overlay_mask_weight"])
    tissue_colors = list(config["tissue_colors"])

    mask_dir = os.path.join(output_dir, "tis_det_mask")
    overlay_dir = os.path.join(output_dir, "tis_det_overlay")
    thumb_dir = os.path.join(output_dir, "tis_det_thumbnail")
    mask_col_dir = os.path.join(output_dir, "tis_det_mask_col")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(thumb_dir, exist_ok=True)
    os.makedirs(mask_col_dir, exist_ok=True)

    slide_name = os.path.basename(path_slide).replace(".svs", "")

    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder_model, encoder_weights)
    model = smp.UnetPlusPlus(
        encoder_name=encoder_model,
        encoder_weights=encoder_weights,
        classes=2,
        activation=None,
    )
    model.load_state_dict(
        torch.load(str(model_td_dir / model_td_name), map_location="cpu")
    )
    model.to(device)
    model.eval()

    slide = OpenSlide(path_slide)

    w_l0, h_l0 = slide.level_dimensions[0]
    mpp = round(float(slide.properties["openslide.mpp-x"]), 4)
    reduction_factor = mpp_model_td / mpp

    image_or = slide.get_thumbnail((w_l0 // reduction_factor, h_l0 // reduction_factor))
    image_or.save(os.path.join(thumb_dir, slide_name + ".jpg"), quality=80)

    image = np.array(image_or)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
    _, image = cv2.imencode(".jpg", image, encode_param)
    image = cv2.imdecode(image, 1)
    image = Image.fromarray(image)

    width, height = image.size
    wi_n = width // patch_size
    he_n = height // patch_size
    overhang_wi = width - wi_n * patch_size
    overhang_he = height - he_n * patch_size

    for h in range(he_n + 1):
        for w in range(wi_n + 1):
            if w != wi_n and h != he_n:
                image_work = image.crop(
                    (
                        w * patch_size,
                        h * patch_size,
                        (w + 1) * patch_size,
                        (h + 1) * patch_size,
                    )
                )
            elif w == wi_n and h != he_n:
                image_work = image.crop(
                    (
                        width - patch_size,
                        h * patch_size,
                        width,
                        (h + 1) * patch_size,
                    )
                )
            elif w != wi_n and h == he_n:
                image_work = image.crop(
                    (
                        w * patch_size,
                        height - patch_size,
                        (w + 1) * patch_size,
                        height,
                    )
                )
            else:
                image_work = image.crop(
                    (width - patch_size, height - patch_size, width, height)
                )

            image_pre = get_preprocessing(image_work, preprocessing_fn)
            x_tensor = torch.from_numpy(image_pre).to(device).unsqueeze(0)
            predictions = model.predict(x_tensor)
            predictions = predictions.squeeze().cpu().numpy()
            mask = np.argmax(predictions, axis=0).astype("int8")
            class_mask = make_class_map(mask, tissue_colors)

            if w == 0:
                temp_image = mask
                temp_image_class_map = class_mask
            elif w == wi_n:
                mask = mask[:, patch_size - overhang_wi : patch_size]
                class_mask = class_mask[:, patch_size - overhang_wi : patch_size, :]
                temp_image = np.concatenate((temp_image, mask), axis=1)
                temp_image_class_map = np.concatenate(
                    (temp_image_class_map, class_mask), axis=1
                )
            else:
                temp_image = np.concatenate((temp_image, mask), axis=1)
                temp_image_class_map = np.concatenate(
                    (temp_image_class_map, class_mask), axis=1
                )

        if h == 0:
            end_image = temp_image
            end_image_class_map = temp_image_class_map
        elif h == he_n:
            temp_image = temp_image[patch_size - overhang_he : patch_size,]
            temp_image_class_map = temp_image_class_map[
                patch_size - overhang_he : patch_size, :, :
            ]
            end_image = np.concatenate((end_image, temp_image), axis=0)
            end_image_class_map = np.concatenate(
                (end_image_class_map, temp_image_class_map), axis=0
            )
        else:
            end_image = np.concatenate((end_image, temp_image), axis=0)
            end_image_class_map = np.concatenate(
                (end_image_class_map, temp_image_class_map), axis=0
            )

        Image.fromarray(end_image).save(
            os.path.join(mask_dir, slide_name + "_MASK.png")
        )
        Image.fromarray(end_image_class_map).save(
            os.path.join(mask_col_dir, slide_name + "_MASK_COL.png")
        )
        image_array = np.array(image)
        if end_image_class_map.shape[:2] != image_array.shape[:2]:
            end_image_class_map = cv2.resize(
                end_image_class_map,
                (image_array.shape[1], image_array.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        overlay = cv2.addWeighted(
            image_array,
            overlay_image_weight,
            end_image_class_map,
            overlay_mask_weight,
            0,
        )
        Image.fromarray(overlay).save(
            os.path.join(overlay_dir, slide_name + "_OVERLAY.jpg")
        )
