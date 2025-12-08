import os
import torch
import torch.nn.functional as F
from image_datasets import InternalViewLevel, CadicaViewLevel
from mil_datasets import MILDataset, CADICAMILDataset
from torch.cuda.amp import autocast
from segment_attention_utils import (
    overwrite_torch_mha,
    multi_head_attention_forward,
)
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import wandb
from utils import (
    get_segments_to_use,
    set_seed,
    compute_metrics,
    wandb_log_data,
    load_model,
    load_config,
    YOLOWrap,
)
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import warnings
import json

warnings.filterwarnings("ignore")
import warnings

warnings.filterwarnings("ignore")
from enum import Enum

logger = logging.getLogger(__name__)


class EvaluationMode(Enum):
    ViewLevel = "View Level"
    PatientLevel = "PatientLevel"


IMPORTANT_METRICS = ["auc", "f1", "precision", "recall"]
METRICS_TO_LOG = [
    "AUC",
    "AUC_Mean+-STD",
    "rca AUC",
    "rca AUC_Mean+-STD",
    "lca AUC",
    "lca AUC_Mean+-STD",
    "Macro Segments All AUC",
    "Macro Segments All AUC_Mean+-STD",
    "F1",
    "F1_Mean+-STD",
    "Precision",
    "Precision_Mean+-STD",
    "Recall",
    "Recall_Mean+-STD",
]


def bootstrap_metrics(
    test_logits, test_targets, best_threshold, approach, sampling_steps=1000, prefix=""
):
    samples_cnt = len(test_logits)
    sample_indices = np.array(range(samples_cnt))

    test_logits = torch.stack(test_logits)
    test_targets = torch.stack(test_targets)

    metrics_accumulation = {}

    bootstrap_indices = np.random.randint(
        low=0, high=len(sample_indices), size=(sampling_steps, samples_cnt)
    )

    for subset_indices in tqdm(bootstrap_indices, desc="Bootstrapping metrics"):
        subset_metrics, _ = compute_metrics(
            test_logits[subset_indices],
            test_targets[subset_indices],
            threshold=best_threshold,
            compute_artery_metrics=True,
            compute_segment_metrics=True,
            do_sigmoid=approach != "yolo_baseline",
        )
        for k, v in subset_metrics.items():
            if not any([m in k.lower() for m in IMPORTANT_METRICS]):
                continue

            if k not in metrics_accumulation:
                metrics_accumulation[k] = [v]
            else:
                metrics_accumulation[k].append(v)
    bootstraped_metrics = {}
    for metric, accumulated_values in metrics_accumulation.items():
        accumulated_values = np.array(accumulated_values)
        mean = np.mean(accumulated_values)
        std = np.std(accumulated_values)
        bootstraped_metrics.update(
            {
                f"{metric}_Mean+-STD": f"{np.round(mean, 3)}+-{np.round(std, 2)}",
            }
        )

    return bootstraped_metrics, metrics_accumulation


def load_model_and_weight_file(config, model_dir):
    model = load_model(config)
    model_filename = os.path.join(model_dir, "last.pt")
    assert os.path.isfile(model_filename)
    return model, model_filename


def get_test_loader(config, evaluation_mode, setting):
    if config.approach == "yolo_baseline":
        segments_to_use = get_segments_to_use("proximal")
        backbone_type = "yolo"
        resolution = config.get("resolution", 512)
        severity = 70
    else:
        segments_to_use = get_segments_to_use(config.segments)
        backbone_type = config.model.backbone_type
        resolution = config.resolution
        severity = config.severity

    if evaluation_mode == EvaluationMode.ViewLevel:

        if setting == "test_cadica":
            test_dataset = CadicaViewLevel(
                backbone_type=backbone_type,
                resolution=resolution,
                severity=70,
            )
        else:
            test_dataset = InternalViewLevel(
                backbone_type=backbone_type,
                resolution=resolution,
                segments=segments_to_use,
            )
    elif evaluation_mode == EvaluationMode.PatientLevel:
        if setting == "test_cadica":
            test_dataset = CADICAMILDataset(
                backbone_type=backbone_type,
                resolution=resolution,
                severity=severity,
                multiframe=config.multiframe,
            )
        else:
            test_dataset = MILDataset(
                backbone_type=backbone_type,
                setting=setting,
                resolution=resolution,
                severity=severity,
                segment_based=config.get("segment_based", False),
                artery_based=config.get("artery_based", False),
                multiframe=config.multiframe,
                segments=segments_to_use,
                augmentation=False,
            )

    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
    )
    logger.info(
        f"{evaluation_mode.value} Test Loader size: {len(test_loader)} (total samples {len(test_dataset)})"
    )
    return test_loader


def evaluate(
    model_dir: str,
    dataset: str,
    evaluation_mode: EvaluationMode,
    bootstrap: bool = True,
    segments: str = None,
):
    config = load_config(model_dir)
    if config is None:
        config = OmegaConf.create(
            {
                "approach": "yolo_baseline",
                "multiframe": {
                    "frames_per_view": 1,
                    "left": 0,
                    "right": 0,
                    "spacing_left": 0,
                    "spacing_right": 0,
                },
            }
        )

    set_seed(config.get("seed", 0))
    if segments is not None:
        config.segments = segments

    if evaluation_mode == EvaluationMode.ViewLevel:
        strict_load = config.multiframe.frames_per_view == 1
        config.multiframe.frames_per_view = 1
    else:
        strict_load = True
    device = (
        torch.device("cuda")
        if "cuda" in config.get("device", "cuda") and torch.cuda.is_available()
        else torch.device("cpu")
    )

    if config.approach == "yolo_baseline":
        model = YOLOWrap(model_dir)
    else:
        model, last_weights_file = load_model_and_weight_file(config, model_dir)
        model.load_state_dict(torch.load(last_weights_file), strict=strict_load)
        print(f"Model loaded from {last_weights_file}")
        model.eval()
    model.to(device)

    test_loader = get_test_loader(config, evaluation_mode, dataset)

    metrics_prefix = f"{dataset.upper()}_{evaluation_mode.value}"
    with torch.no_grad():
        test_logits, test_targets, test_samples_cnt = [], [], []
        for batch_ind, (
            images,
            targets,
            samples_cnt,
            ids,
            angulations,
            stenosis_percentage,
        ) in tqdm(
            enumerate(test_loader),
            total=len(test_loader),
            desc=f"Testing model",
        ):

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            angulations = angulations.to(device, non_blocking=True)

            with autocast():
                if config.approach == "yolo_baseline":
                    logits = model(images)
                    if evaluation_mode == EvaluationMode.PatientLevel:
                        logits = torch.tensor(
                            [
                                pred[: samples_cnt[i]].max()
                                for i, pred in enumerate(logits)
                            ]
                        ).to(device)

                elif config.approach in [
                    "max_mil",
                    "attn_mil",
                    "cls_mil",
                    "segment_mil",
                ]:
                    if evaluation_mode == EvaluationMode.ViewLevel:
                        logits = model(
                            images[:, None, None, :, :, :],
                            samples_cnt,
                            angulations,
                            return_attn_weights=False,
                        )
                        if len(logits.shape) == 2:
                            logits = logits[:, 0]
                    elif evaluation_mode == EvaluationMode.PatientLevel:
                        logits = model(images, samples_cnt, angulations)

                else:
                    raise ValueError()
            if dataset == "test_cadica" and len(logits.shape) > 1:
                logits = logits[:, 0]

            if len(targets.shape) == 1:
                targets = targets.unsqueeze(1)

            logits = logits.reshape(targets.shape)
            test_logits.extend(list(logits.cpu()))
            test_targets.extend(list(targets.cpu()))
            test_samples_cnt.extend(list(samples_cnt))

        test_metrics, best_threshold = compute_metrics(
            logits=test_logits,
            targets=test_targets,
            threshold=None if dataset == "val" else config.get("threshold", 0.5),
            compute_artery_metrics=True,
            compute_segment_metrics=True,
            do_sigmoid=config.approach != "yolo_baseline",
        )
        test_metrics["threshold"] = best_threshold

        if dataset == "val":
            config.threshold = best_threshold
            OmegaConf.save(config, os.path.join(model_dir, "config.yaml"))

    if bootstrap:
        bootstraped_metrics, _ = bootstrap_metrics(
            test_logits,
            test_targets,
            best_threshold=best_threshold,
            prefix=metrics_prefix,
            approach=config.approach,
        )
        test_metrics.update(bootstraped_metrics)

    torch.cuda.empty_cache()
    return test_metrics


def evaluate_baseline():
    baseline_dir = "/home/nikolac/stenosis-classification/stenosis_classification/bbox_baseline/frame-classification"
    model = "10-16_18-45-31-lr0.001-mos1.0_train"
    model_dir = os.path.join(baseline_dir, model)

    evaluate(
        model_dir,
        dataset="gold_standard",
        bootstrap=bootstrap,
        evaluation_mode=EvaluationMode.ViewLevel,
    )
    return
    evaluate_model(
        model_dir,
        dataset="gold_standard",
        bootstrap=bootstrap,
    )
    evaluate_model(
        model_dir,
        dataset="test_cadica",
        bootstrap=bootstrap,
        evaluation_mode=EvaluationMode.ViewLevel,
    )

    evaluate_model(
        model_dir,
        dataset="test_cadica",
        bootstrap=bootstrap,
    )

    evaluate_model(
        model_dir,
        dataset="gold_standard",
        bootstrap=bootstrap,
    )
    evaluate_model(model_dir, dataset="test", bootstrap=bootstrap)
    evaluate_model(model_dir, dataset="gold_standard", bootstrap=bootstrap)


if __name__ == "__main__":
    model_dir = "/home/nikolac/multi-view-stenosis-classification/stenosis-classification/12_04_18:16:56-airl"
    # evaluate_baseline()
    for dataset, level in [
        # ("val", EvaluationMode.PatientLevel),
        # ("test", EvaluationMode.PatientLevel),
        # ("test", EvaluationMode.ViewLevel),
        # ("test_cadica", EvaluationMode.PatientLevel),
        ("test_cadica", EvaluationMode.ViewLevel),
    ]:

        metrics = evaluate(
            model_dir, dataset=dataset, evaluation_mode=level, bootstrap=False
        )
        print(metrics)
