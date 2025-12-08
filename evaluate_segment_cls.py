from evaluate import load_config
from models import (
    SegmentBasedMultiInstanceClassifier,
)
from image_datasets import (
    SegmentExtractionDataset,
)
import torch
import pandas as pd
import os
import random
import json
from typing import List
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score


def compute_multiclass_metrics(preds, labels, average):
    if average == "macro":
        no_labels_segments_mask = [not all(annots == 0) for annots in labels.T]
        print(
            f"{sum(no_labels_segments_mask)}/{len(no_labels_segments_mask)} segments considered in macro metrics computation!"
        )
        preds = preds[:, no_labels_segments_mask]
        labels = labels[:, no_labels_segments_mask]
    preds_round = np.round(preds)

    metrics = {
        "AUC": roc_auc_score(y_true=labels, y_score=preds, average=average),
        "F1": f1_score(y_true=labels, y_pred=preds_round, average=average),
        "Precission": precision_score(
            y_true=labels, y_pred=preds_round, average=average
        ),
        "Recall": recall_score(y_true=labels, y_pred=preds_round, average=average),
    }
    return {k: np.round(v, 2) for k, v in metrics.items()}


def get_weigths_file(model_dir):
    weight_files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
    return os.path.join(model_dir, sorted(weight_files)[-1])


def create_segment_extraction_dataloader(config, merge_into_single_sample):

    dataset = SegmentExtractionDataset(
        resolution=config.resolution, merge_into_single_sample=merge_into_single_sample
    )
    print(f"Segment extraction dataset created! Total samples: {len(dataset)}")
    return DataLoader(
        dataset,
        batch_size=2 if merge_into_single_sample else 5,
        shuffle=False,
        drop_last=False,
        num_workers=1 if merge_into_single_sample else 8,
    )


def format_scores(attn_scores: List):
    df = None
    for view_attn_score in attn_scores:
        row = {
            "view_id": view_attn_score["view_id"],
            "labels": view_attn_score["labels"],
            **{i + 1: s for i, s in enumerate()},
        }


def get_segment_attention_scores(
    model_dir, zero_shot=False, remove_softmax: bool = True, raw: bool = True
):
    model_dir = os.path.join("stenosis-classification", model_dir)
    config = load_config(model_dir)
    seed = config.get("seed", 2000)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = "cuda"
    loader = create_segment_extraction_dataloader(
        config, merge_into_single_sample=not remove_softmax
    )
    model = SegmentBasedMultiInstanceClassifier(
        pretrained=True,
        embed_dim=config.model.embed_dim,
        resolution=config.resolution,
        projector_layers=config.model.projector_layers,
        classifier_layers=config.model.classifier_layers,
        attention_type=config.model.attention_type,
        attention_heads=config.model.attention_heads,
        shared_classifier=config.model.shared_classifier,
        hierarchical=config.model.hierarchical,
        transformer_layers=config.model.transformer_layers,
        monkey_patch_torch=remove_softmax,
    ).to(device)
    if not zero_shot:
        model.load_state_dict(torch.load(get_weigths_file(model_dir)))
    else:
        print("ZERO SHOT!")
    model.eval()
    all_preds = []
    all_labels = []
    metrics = {"model_dir": os.path.dirname(model_dir)}
    attn_scores = []
    with torch.no_grad():
        for images, labels, samples_cnt, ids in tqdm(
            loader, total=len(loader), desc="Predicting segment attention"
        ):
            images = images.to(device)
            logits, attn_output_weights = model(
                images, samples_cnt, return_attn_weights=True
            )  # batch_size, 17, 1 or 10
            attn_output_weights = attn_output_weights[
                :, 1:, :
            ]  # batch_size, 16 ,1 or 10

            if remove_softmax:
                if not raw:
                    attn_output_weights = (
                        attn_output_weights - attn_output_weights.mean(1, keepdim=True)
                    )
                    attn_output_weights = torch.nn.functional.sigmoid(
                        attn_output_weights
                    )

            else:
                attn_output_weights = attn_output_weights.swapaxes(1, 2)

            attn_output_weights = attn_output_weights.cpu().detach().squeeze().numpy()
            labels = labels.cpu().squeeze().numpy()

            all_preds.append(attn_output_weights)
            all_labels.append(labels)

            for i, id in enumerate(ids):
                attn_scores.append(
                    {
                        "view_id": id,
                        **{
                            (segment + 1): round(float(i), 4)
                            for segment, i in enumerate(attn_output_weights[i])
                        },
                        "labels": [int(s) for s in labels[i]],
                    }
                )
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    outdir = os.path.join("segments_predictions")
    os.makedirs(os.path.join(outdir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "attn_scores"), exist_ok=True)

    outfile = f'{"RAW_" if raw else ""}{"WITH_SOFTMAX_" if not remove_softmax else ""}{"ZERO_SHOT_" if zero_shot else ""}{os.path.basename(model_dir.rstrip("/"))}'
    if remove_softmax and not raw:
        metrics["metrics"] = {
            "micro": compute_multiclass_metrics(all_preds, all_labels, average="micro"),
            "macro": compute_multiclass_metrics(all_preds, all_labels, average="macro"),
        }
        metrics_outfile = os.path.join(outdir, "metrics", f"{outfile}.json")
        with open(metrics_outfile, "w") as f:
            json.dump(metrics, f, indent=4)
            print(f"Segment metrics written in {metrics_outfile}")

    scores_outfile = os.path.join(outdir, "attn_scores", f"{outfile}.csv")
    pd.DataFrame(attn_scores).to_csv(scores_outfile)


if __name__ == "__main__":
    model_dirs = [
        "12_20_01:40:15-single_frame-MIL-SB--balance_with_priority----transformer--ALL_frames-base_aug-0.0001-224-stenosis-classification_TUNE_LR_AND_WD_ON_LONG_RUN",
        "12_20_03:58:24-single_frame-MIL-SB--balance_with_priority--hier--transformer--ALL_frames-base_aug-0.0003-224-stenosis-classification_TUNE_LR_AND_WD_ON_LONG_RUN",
        "12_22_01:05:16-single_frame-MIL-SB--balance_with_priority---4-5-10-14-15-16-transformer--ALL_frames-base_aug-0.0003-224-stenosis-classification_TUNE_LR_AND_WD_ON_LONG_RUN_WITH_SELECTED_SEGMENTS",
        "12_22_07:46:14-single_frame-MIL-SB--balance_with_priority--hier-4-5-10-14-15-16-transformer--ALL_frames-base_aug-0.0003-224-stenosis-classification_TUNE_LR_AND_WD_ON_LONG_RUN_WITH_SELECTED_SEGMENTS",
    ]
    remove_softmax = True
    # get_segment_attention_scores(model_dirs[0], zero_shot=True, remove_softmax=remove_softmax)
    for model_dir in model_dirs:
        get_segment_attention_scores(model_dir, remove_softmax=remove_softmax, raw=True)
        get_segment_attention_scores(
            model_dir, remove_softmax=remove_softmax, raw=False
        )
