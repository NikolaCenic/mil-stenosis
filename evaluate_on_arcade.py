import wandb
from torch.utils.data import DataLoader
from mil_datasets import SegmentationDataset
from evaluate import load_config, load_model_and_weight_file
import torch.nn as nn
import torch
from tqdm import tqdm
import os
from PIL import Image
import math
import numpy as np
from torch.cuda.amp import autocast
from skimage.filters import (
    threshold_li,
    threshold_isodata,
    threshold_local,
    threshold_mean,
    threshold_minimum,
    threshold_multiotsu,
    threshold_niblack,
    threshold_otsu,
    threshold_sauvola,
    threshold_triangle,
    threshold_yen,
)
from ultralytics import YOLO
from datasets_utils import (
    normalize_per_image,
)
import json
from omegaconf import OmegaConf

BASELINE_MODEL_PATH = {
    "syntax": "/home/nikolac/stenosis-classification/stenosis_classification/data_scripts/arcade/angiography-segmentation/07-05_19-25-42_0.001_1.0_train/weights/best.pt",
    "stenosis": "/home/nikolac/stenosis-classification/stenosis_classification/data_scripts/arcade/angiography-stenosis/10-29_21-36-56_stenosis_0.001_1.0_train/weights/best.pt",
}


def normalize_batch(x):
    maxi = x.amax(dim=(-2, -1), keepdim=True)
    mini = x.amin(dim=(-2, -1), keepdim=True)
    return (x - mini) / (maxi - mini)


def compute_segmentation_metrics(preds, targets, eps=1e-7):
    preds = preds.int()
    targets = targets.int()

    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    tp = torch.sum((preds_flat == 1) & (targets_flat == 1)).float()
    fp = torch.sum((preds_flat == 1) & (targets_flat == 0)).float()
    fn = torch.sum((preds_flat == 0) & (targets_flat == 1)).float()
    tn = torch.sum((preds_flat == 0) & (targets_flat == 0)).float()

    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)

    return {
        "accuracy": accuracy.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
        "iou": iou.item(),
        "dice": dice.item(),
    }


class SegmentationModelWrap(nn.Module):
    def __init__(self, config, model_dir, task, aggregation_method):
        super().__init__()
        if model_dir == "baseline":
            ckpt = BASELINE_MODEL_PATH[task]
            self.model = YOLO(ckpt)
            print(f"YOLO Model loaded from {ckpt}")

            self.conf = 0.03
            self.baseline = True
        else:
            self.baseline = False
            self.model, weights_file = load_model_and_weight_file(config, model_dir)
            self.model.load_state_dict(torch.load(weights_file))
            self.head_aggregation, self.segments_aggregation = aggregation_method.split(
                "_"
            )
            self.eval()

    def aggregate_attention_scores(self, x):
        if self.head_aggregation == "mean":
            x = x[:, :, :, :, :].mean(1)
        elif self.head_aggregation == "max":
            x, _ = torch.max(x[:, :, :, :, :], dim=1)
        elif self.head_aggregation.isnumeric():
            x = x[:, int(self.head_aggregation), :, :, :]
        else:
            raise ValueError(f"{self.head_aggregation} is wrong head agregation!")
        if self.segments_aggregation == "mean":
            x = x[:, :, :, :].mean(1)
        elif self.segments_aggregation == "max":
            x, _ = torch.max(x[:, :, :, :], dim=1)

        elif self.segments_aggregation == "cls":
            x = x[:, 0, :, :]
        elif self.segments_aggregation == "artery-mean":
            x = x[:, 1:3, :, :].mean(1)
        elif self.segments_aggregation == "artery-max":
            x, _ = torch.max(x[:, 1:3, :, :], dim=1)
        elif self.segments_aggregation == "segment-mean":
            x = x[:, 3:, :, :].mean(1)
        elif self.segments_aggregation == "segment-max":
            x, _ = torch.max(x[:, 3:, :, :], dim=1)
        else:
            raise ValueError(
                f"{self.segments_aggregation} is wrong segment agregation!"
            )
        return x

    def forward(self, x):
        if self.baseline:
            return self.forward_baseline(x)
        x = x.unsqueeze(1).unsqueeze(1)
        _, attn_output_weights = self.model(
            x, [1] * len(x), torch.zeros((x.shape[0], 1, 2)), return_attn_weights=True
        )
        attn_output_weights = self.aggregate_attention_scores(attn_output_weights)
        patch_size = int(math.sqrt(attn_output_weights.shape[-1]))
        attn_output_weights = attn_output_weights.reshape(-1, patch_size, patch_size)

        attn_output_weights = normalize_batch(attn_output_weights)
        return attn_output_weights.squeeze()

    def forward_baseline(self, x):
        in_out_shape = (640, 640)
        x = torch.nn.functional.interpolate(x, size=in_out_shape, mode="bicubic")
        x = normalize_per_image(x)

        results = self.model(x, verbose=False, conf=self.conf)
        return torch.stack(
            [
                (
                    torch.max(r.masks.data.cpu(), axis=0).values
                    if r.masks and r.masks.data.numel() > 0
                    else torch.zeros(in_out_shape)
                )
                for r in results
            ]
        )


def get_test_loader(config, task, setting):
    return DataLoader(
        SegmentationDataset(
            task,
            setting=setting,
            backbone_type=config.model.backbone_type,
            resolution=config.resolution,
        ),
        batch_size=32,
        shuffle=False,
        drop_last=False,
        num_workers=8,
    )


def log_segmentation_images(x, y_pred_raw, y_pred, y, task):
    def interpolate_and_normalize(x, target_shape):
        x = x.float()
        x = nn.functional.interpolate(x[:, None, :, :], target_shape).squeeze()
        return normalize_batch(x).moveaxis(0, 1).flatten(1, 2)

    y_pred_raw = interpolate_and_normalize(y_pred_raw, target_shape=x.shape[-2:])
    y = interpolate_and_normalize(y, target_shape=x.shape[-2:])
    y_pred = interpolate_and_normalize(y_pred, target_shape=x.shape[-2:])

    x = normalize_batch(x[:, 0, :, :]).moveaxis(0, 1).flatten(1, 2)
    image_to_log = Image.fromarray(
        (255 * np.array(torch.concat([x, y_pred_raw, y_pred, y]))).astype(np.uint8)
    )
    wandb.log({f"Media/Predictions {task.capitalize()}": wandb.Image(image_to_log)})


from skimage import measure, morphology
from skimage.morphology import erosion, disk


def postprocess_batch(batch, min_size, thinning_coef, threshold_function):
    result = []
    for image in np.array(batch):
        threshold = threshold_function(image)
        image = image >= threshold
        image = morphology.remove_small_objects(image, min_size=min_size).astype(
            np.uint8
        )

        image = erosion(image, disk(thinning_coef))
        result.append(torch.tensor(image).int())

    return torch.stack(result)


def evaluate_on_arcade(
    model_dir: str,
    aggregation_method: str,
    setting: str,
    task: str,
    run: wandb.run = None,
    interpolation="bicubic",
    min_size=1000,
    thinning_coef=5,
    threshold_function=threshold_otsu,
):
    device = torch.device("cuda")

    new_run = run is None
    if new_run:
        wandb.init(
            project="stenosis-test",
            name=f"{task}_evaluation_{model_dir.split('/')[-1]}-{setting}-{aggregation_method}-int-{interpolation}_{str(threshold_function).split()[1]}",
            config={
                "interpolation": interpolation,
                "aggregation_method": aggregation_method,
                "min_size": min_size,
                "thinning_coef": thinning_coef,
                "threshold_function": str(threshold_function),
            },
        )

    baseline = model_dir == "baseline"
    if baseline:
        stored_file_path = os.path.join(
            os.path.dirname(BASELINE_MODEL_PATH[task]), "stored_evaluation_metrics.json"
        )
    else:
        stored_file_path = os.path.join(model_dir, "stored_evaluation_metrics.json")
    if os.path.exists(stored_file_path):
        with open(stored_file_path, "r") as f:
            stored_metrics = json.load(f)
    else:
        stored_metrics = {}
    try:
        if baseline:
            config = OmegaConf.create(
                {"model": {"backbone_type": "yolo"}, "resolution": 518}
            )
        else:
            config = load_config(model_dir)
        model = SegmentationModelWrap(
            config, model_dir, aggregation_method=aggregation_method, task=task
        ).to(device)

        test_loader = get_test_loader(config, task, setting)
        raw_predictions = []
        images = []
        targets = []
        for x, y in tqdm(
            test_loader,
            desc=f"Testing model {task.capitalize()}",
            total=len(test_loader),
        ):
            with torch.no_grad():
                x = x.to(device)
                with autocast():
                    y_pred_raw = model(x).cpu()

                images.append(x.cpu())
                raw_predictions.append(y_pred_raw)
                targets.append(y)

        images = torch.concat(images)
        targets = torch.concat(targets)
        raw_predictions = torch.concat(raw_predictions)

        if baseline:
            processed_preds = nn.functional.interpolate(
                raw_predictions.unsqueeze(1), size=targets.shape[-2:], mode="nearest"
            ).squeeze()

        else:
            raw_predictions = nn.functional.interpolate(
                raw_predictions.unsqueeze(1),
                size=targets.shape[-2:],
                mode=interpolation,
            ).squeeze()
            processed_preds = postprocess_batch(
                raw_predictions,
                min_size=min_size,
                thinning_coef=thinning_coef,
                threshold_function=threshold_function,
            )
        test_metrics = compute_segmentation_metrics(
            preds=processed_preds, targets=targets
        )
        cnt_to_log = 0
        step = 10
        for start in range(0, cnt_to_log, step):
            log_segmentation_images(
                x=images[start : start + step],
                y_pred_raw=raw_predictions[start : start + step],
                y_pred=processed_preds[start : start + step],
                y=targets[start : start + step],
                task=task,
            )
        seg_outdir = "segmentation_figure_images/figure"
        os.makedirs(seg_outdir, exist_ok=True)

        for i in tqdm(range(len(images))):
            for data, filename in [
                (raw_predictions, "raw_preds"),
                (processed_preds, "baseline_pred"),
            ]:
                im = np.array(255 * normalize_batch(data[i]).detach().cpu()).astype(
                    np.uint8
                )
                if len(im.shape) == 3:
                    im = np.moveaxis(im, 0, 2)
                im = Image.fromarray(im)
                if filename == "attn_score":
                    im = im.resize((518, 518), Image.LANCZOS)
                im.save(f"{seg_outdir}/{i+1}_{filename}.png")

        key = "Segmentation Dice Score" if task == "syntax" else "Stenosis Dice Score"
        if key in stored_metrics.keys():
            del stored_metrics[key]
        stored_metrics[f"{key} - {aggregation_method}"] = test_metrics["dice"]
        wandb.log({k.capitalize(): v for k, v in test_metrics.items()})

    finally:

        with open(stored_file_path, "w") as f:
            json.dump(stored_metrics, f, indent=4, sort_keys=True)
        wandb.finish()
        if new_run:
            wandb.finish()


def evalute_baseline(task):
    evaluate_on_arcade("baseline", setting="test", task=task, aggregation_method=None)


if __name__ == "__main__":
    task = "syntax"
    MODELS_SAVE_DIR = "stenosis-classification"
    best_model = "09_13_07:27:47-ytok-segment_mil-vit-patch--1e-05-518-proximal--FINAL_RUNS-0.4_0.4_0.2"
    ha = "max"
    sa = "max"
    threshold_function = threshold_triangle
    evaluate_on_arcade(
        os.path.join(MODELS_SAVE_DIR, best_model),
        task=task,
        setting="train",
        aggregation_method=f"{ha}_{sa}",
        threshold_function=threshold_function,
    )
    exit()
    models = [
        "09_13_15:00:19-yurm-segment_mil-vit-patch--1e-05-518-proximal--FINAL_RUNS-0.0_0.0_1.0",
        "09_12_23:45:28-anxb-segment_mil-vit-patch--1e-05-518-proximal--FINAL_RUNS-0.0_1.0_0.0",
        "09_12_23:45:28-emzz-segment_mil-vit-patch--1e-05-518-proximal--FINAL_RUNS-1.0_0.0_0.0",
        "09_14_09:54:02-wkqo-segment_mil-vit-patch--1e-05-518-proximal--PATCH_BEST_SETTING_NO_HIER-0.4_0.4_0.2",
        best_model,
    ]
    for m in models:
        evaluate_on_arcade(
            os.path.join(MODELS_SAVE_DIR, m),
            task=task,
            setting="test",
            aggregation_method=f"{ha}_{sa}",
            threshold_function=threshold_function,
        )

    for m, sa in [
        (
            "09_13_15:00:19-yurm-segment_mil-vit-patch--1e-05-518-proximal--FINAL_RUNS-0.0_0.0_1.0",
            "segment-max",
        ),
        (
            "09_12_23:45:28-anxb-segment_mil-vit-patch--1e-05-518-proximal--FINAL_RUNS-0.0_1.0_0.0",
            "artery-max",
        ),
        (
            "09_12_23:45:28-emzz-segment_mil-vit-patch--1e-05-518-proximal--FINAL_RUNS-1.0_0.0_0.0",
            "cls",
        ),
    ]:
        evaluate_on_arcade(
            os.path.join(MODELS_SAVE_DIR, m),
            task=task,
            setting="test",
            aggregation_method=f"{ha}_{sa}",
            threshold_function=threshold_function,
        )
