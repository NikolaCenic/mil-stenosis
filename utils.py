from sklearn import metrics
from typing import List
import numpy as np
from PIL import Image
import torch
import wandb
import cv2
from models import MaxMIL, AttnMIL, ClsMIL, SegmentMIL
from torch.utils.data import DataLoader
from image_datasets import SingleSampleDataset
from mil_datasets import MILDataset
import logging
import torch.nn.functional as F
from typing import Optional
import math
import torchvision.transforms as T
import random
import matplotlib.pyplot as plt
from constants import MAJOR_SEGMENTS, ALL_SEGMENTS, ARTERY_SEGMENTS
from omegaconf import DictConfig, OmegaConf
import os
import yaml
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class YOLOWrap(torch.nn.Module):
    def __init__(self, model_dir):
        super().__init__()
        self.model = YOLO(os.path.join(model_dir, "weights", "last.pt"))

    def get_conf(self, pred):
        confs = pred.boxes.conf
        if confs.numel() > 0:
            return confs.max()
        else:
            return torch.tensor(0).to(confs.device)

    def inference(self, x):
        return torch.stack([self.get_conf(p) for p in self.model(x, verbose=False)])

    def forward(self, x):
        if len(x.shape) == 6:
            bs, ms, frames_per_view, c, h, w = x.shape
            return self.inference(x.flatten(0, -4)).reshape(bs, ms)
        else:
            return self.inference(x)


def load_config(model_dir: str) -> DictConfig:
    config_file = os.path.join(model_dir, "config.yaml")
    if not os.path.isfile(config_file):
        return None

    with open(os.path.join(model_dir, "config.yaml"), "r") as f:
        return OmegaConf.create(yaml.safe_load(f))


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed}")


def get_segments_to_use(segments):
    assert segments in ["major", "all"]
    if segments == "major":
        return MAJOR_SEGMENTS
    elif segments == "all":
        return ALL_SEGMENTS
    else:
        raise ValueError(f"WRONG SEGMENTS SPECIFIED! {segments}")


def get_query_cnt(segments, segment_based, artery_based, cls_based=True):
    assert segments in ["proximal", "all"]
    assert artery_based in [True, False]
    queries = 1
    if artery_based:
        queries += len(ARTERY_SEGMENTS)
    if segment_based:
        if segments == "proximal":
            queries += len(MAJOR_SEGMENTS)
        elif segments == "all":
            queries += len(ALL_SEGMENTS)
        else:
            raise ValueError(
                f"WRONG SEGMENTS AND ARTERY SPECIFIED! Segments: {segments} Artery Based: {artery_based}"
            )
    return queries


def get_artery_mapping(segments):
    assert segments in ["proximal", "all"]
    segments_list = MAJOR_SEGMENTS if segments == "proximal" else ALL_SEGMENTS
    artery_mapping = {"RCA": [], "LCA": []}
    for i, s in enumerate(segments_list):
        for artery in artery_mapping.keys():
            if s in ARTERY_SEGMENTS[artery]:
                artery_mapping[artery].append(i)
                break
    return artery_mapping


def load_model(config):
    if config.approach == "max_mil":
        model = MaxMIL(
            backbone_type=config.model.backbone_type,
            pretrained=config.model.pretrained,
            encode_level="global",
            resolution=config.resolution,
            embed_dim=config.model.embed_dim,
            projector_layers=config.model.projector_layers,
            classifier_layers=config.model.classifier_layers,
        )
    elif config.approach == "attn_mil":
        model = AttnMIL(
            backbone_type=config.model.backbone_type,
            pretrained=config.model.pretrained,
            encode_level="global",
            resolution=config.resolution,
            embed_dim=config.model.embed_dim,
            projector_layers=config.model.projector_layers,
            classifier_layers=config.model.classifier_layers,
        )
    elif config.approach.endswith("cls_mil"):
        model = ClsMIL(
            backbone_type=config.model.backbone_type,
            pretrained=config.model.pretrained,
            encode_level=config.model.encode_level,
            embed_dim=config.model.embed_dim,
            resolution=config.resolution,
            projector_layers=config.model.projector_layers,
            classifier_layers=config.model.classifier_layers,
            attention_type=config.model.attention_type,
            attention_heads=config.model.attention_heads,
            frames_per_view=config.multiframe.frames_per_view,
            cnt_pos_embeddings=config.model.cnt_pos_embeddings,
            transformer_layers=config.model.transformer_layers,
            handle_multiframe=config.model.get("handle_multiframe", "concat"),
        )
    elif config.approach == "segment_mil":

        model = SegmentMIL(
            backbone_type=config.model.backbone_type,
            pretrained=config.model.pretrained,
            encode_level=config.model.encode_level,
            embed_dim=config.model.embed_dim,
            resolution=config.resolution,
            projector_layers=config.model.projector_layers,
            classifier_layers=config.model.classifier_layers,
            attention_type=config.model.attention_type,
            attention_heads=config.model.attention_heads,
            transformer_layers=config.model.transformer_layers,
            shared_classifier=config.model.shared_classifier,
            frames_per_view=config.multiframe.frames_per_view,
            cnt_pos_embeddings=config.model.cnt_pos_embeddings,
            hierarchical=config.model.hierarchical,
            segments_to_use=get_segments_to_use(config.segments),
            handle_multiframe=config.model.get("handle_multiframe", "concat"),
        )

    else:
        raise ValueError(f"Unknown approach: {config.approach}")

    if os.path.isfile(config.get("model", {}).get("full_model_weights", False)):
        weights = torch.load(config.model.full_model_weights)
        if config.model.model_part_to_load != "full":
            logger.info(f"Only {config.model.model_part_to_load} weights loaded!")
            weights = {
                k: v for k, v in weights.items() if config.model.model_part_to_load in k
            }
        model.load_state_dict(weights, strict=False)
        logger.info(
            f"MODEL INIT OVERWRITTEN WITH WEIGHTS FROM {config.model.full_model_weights}!"
        )

    return model


def get_precision_recall_curve(preds, targets):
    precision, recall, thresholds = metrics.precision_recall_curve(
        probas_pred=preds, y_true=targets
    )
    f1_scores = (
        2 * (precision * recall) / (precision + recall + 1e-8)
    )  # Add epsilon to avoid division by zero

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color="darkorange", lw=2, label="PR Curve")
    plt.plot(recall, f1_scores, color="green", lw=2, linestyle="--", label="F1 Score")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random")

    plt.ylabel("Precision/F1")
    plt.xlabel("Recall")
    plt.title("Precision-Recall Curve with F1 Score")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    return wandb.Image(plt)


def get_roc_curve(preds, targets):
    fpr, tpr, thrs = metrics.roc_curve(y_score=preds, y_true=targets)
    # Plot ROC curve
    plt.figure(figsize=(6, 5))
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label=f"AUC = {metrics.auc(fpr, tpr):.2f}",
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    roc_curve = wandb.Image(plt)
    return roc_curve


def create_confusion_heatmap(
    logits: torch.tensor, targets: torch.tensor, threshold: float
):

    preds_rounded = torch.where(F.sigmoid(logits) >= threshold, 1, 0).float()
    counts = []
    for s in range(0, targets.shape[1]):
        s_counts = torch.mean(preds_rounded[targets[:, s] == 1], 0)
        counts.append(s_counts)

    counts = torch.stack(counts)[None, None, :, :]
    counts = F.interpolate(counts, size=(17 * 17, 17 * 17), mode="nearest").squeeze()
    counts = (255 * np.array(counts)).astype(np.uint8)
    return wandb.Image(Image.fromarray(counts))


def compute_metrics_aux(
    logits: torch.tensor,
    targets: torch.tensor,
    threshold: float,
    do_sigmoid: bool = True,
):

    if do_sigmoid:
        preds = np.array(F.sigmoid(logits))
    else:
        preds = np.array(logits)
    targets = np.array(targets).astype(np.int32)

    # search for best threshold
    if threshold is None:
        metrics_per_threshold = {}
        thresholds = np.concatenate(
            [np.linspace(0.0, 0.1, 9), np.linspace(0.1, 0.9, 9)]
        )
        for thr in thresholds:
            preds_rounded = np.where(preds >= thr, 1, 0).astype(np.int32)
            metrics_per_threshold[thr] = {
                "f1_score": metrics.f1_score(y_pred=preds_rounded, y_true=targets),
                "recall": metrics.recall_score(y_pred=preds_rounded, y_true=targets),
                "precision": metrics.precision_score(
                    y_pred=preds_rounded, y_true=targets
                ),
            }

        abs_dif = 0.1

        while threshold is None:
            metrics_per_threshold_to_consider = {
                thr: thr_dict
                for thr, thr_dict in metrics_per_threshold.items()
                if abs(thr_dict["precision"] - thr_dict["recall"]) <= abs_dif
            }
            if metrics_per_threshold_to_consider != {}:
                threshold = float(
                    max(
                        metrics_per_threshold,
                        key=lambda k: metrics_per_threshold[k]["f1_score"],
                    )
                )
            else:
                abs_dif += 0.1
    best_threshold = threshold
    # compute metrics
    preds_rounded = np.where(preds >= best_threshold, 1, 0).astype(np.int32)
    all_metrics = {
        "AUC": np.round(metrics.roc_auc_score(y_score=preds, y_true=targets), 3),
        "F1": np.round(metrics.f1_score(y_pred=preds_rounded, y_true=targets), 3),
        "Precision": np.round(
            metrics.precision_score(y_pred=preds_rounded, y_true=targets), 3
        ),
        "Recall": np.round(
            metrics.recall_score(y_pred=preds_rounded, y_true=targets), 3
        ),
    }

    tn, fp, fn, tp = metrics.confusion_matrix(
        y_pred=preds_rounded, y_true=targets
    ).ravel()
    confusion_matrix = {
        "Total Possitives": sum(targets),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
    }

    return all_metrics, confusion_matrix, best_threshold


def create_loaders(config):
    if config.multi_instance:
        train_dataset = MILDataset(
            backbone_type=config.model.backbone_type,
            setting="train",
            resolution=config.resolution,
            severity=config.severity,
            multiframe=config.multiframe,
            segment_based=config.segment_based,
            artery_based=config.artery_based,
            segments=get_segments_to_use(config.segments),
            augmentation=config.augmentation,
        )
        val_dataset = MILDataset(
            backbone_type=config.model.backbone_type,
            setting="val",
            resolution=config.resolution,
            severity=config.severity,
            multiframe=config.multiframe,
            segment_based=config.segment_based,
            artery_based=config.artery_based,
            segments=get_segments_to_use(config.segments),
            augmentation=False,
        )
    else:
        train_dataset = SingleSampleDataset(
            backbone_type=config.model.backbone_type,
            setting="train",
            resolution=config.resolution,
            severity=config.severity,
            segments=get_segments_to_use(config.segments),
            augmentation=config.augmentation,
        )
        val_dataset = SingleSampleDataset(
            backbone_type=config.model.backbone_type,
            setting="val",
            resolution=config.resolution,
            severity=config.severity,
            segments=get_segments_to_use(config.segments),
            augmentation=False,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
    )
    logger.info(
        f"Train Loader size: {len(train_loader)} (total samples {len(train_dataset)})"
    )
    logger.info(
        f"Valid Loader size: {len(val_loader)} (total samples {len(val_dataset)})"
    )
    return train_loader, val_loader


def compute_metrics(
    logits: List[float],
    targets: List[float],
    threshold: float = None,
    compute_artery_metrics: bool = False,
    compute_segment_metrics: bool = False,
    do_sigmoid: bool = True,
):
    segment_based = False
    artery_based = False

    if targets[0].numel() > 1:
        segments = ALL_SEGMENTS if targets[0].numel() == 16 else MAJOR_SEGMENTS
        segment_based = True
        artery_based = True

    if not isinstance(logits, torch.Tensor):
        logits = torch.stack(logits)
        targets = torch.stack(targets)

    if not segment_based and not artery_based:
        logits = logits.unsqueeze(1)
        targets = targets.unsqueeze(1)

    # compute general cls metrics
    cls_metrics, _, best_threshold = compute_metrics_aux(
        logits[:, 0],
        targets[:, 0],
        threshold=threshold,
        do_sigmoid=do_sigmoid,
    )

    metrics_to_log = cls_metrics

    if artery_based and compute_artery_metrics:
        rca_metrics, _, _ = compute_metrics_aux(
            logits[:, 1],
            targets[:, 1],
            threshold=best_threshold,
            do_sigmoid=do_sigmoid,
        )
        lca_metrics, _, _ = compute_metrics_aux(
            logits[:, 2],
            targets[:, 2],
            threshold=best_threshold,
            do_sigmoid=do_sigmoid,
        )
        metrics_to_log.update({f"rca/{k}": v for k, v in rca_metrics.items()})
        metrics_to_log.update({f"lca/{k}": v for k, v in lca_metrics.items()})

    if segment_based and compute_segment_metrics:

        all_segment_metrics, conf_mat_all_segments, _ = compute_metrics_aux(
            logits[:, 3:].flatten(0, 1),
            targets[:, 3:].flatten(0, 1),
            threshold=best_threshold,
            do_sigmoid=do_sigmoid,
        )

        metrics_to_log.update(
            {f"Micro Segments All/{k}": v for k, v in all_segment_metrics.items()}
        )

        for s in range(1, len(segments) + 1):
            # compute segment metrics
            segment_logits = logits[:, s]
            segment_targets = targets[:, s]
            if sum(segment_targets) == 0:
                continue
            segment_metrics, _, _ = compute_metrics_aux(
                segment_logits,
                segment_targets,
                threshold=best_threshold,
                do_sigmoid=do_sigmoid,
            )

            metrics_to_log.update(
                {f"Segment {segments[s-1]}/{k}": v for k, v in segment_metrics.items()}
            )

        metrics_to_log["Macro Segments All/AUC"] = np.round(
            np.array(
                [
                    metrics_to_log[f"Segment {segments[s-1]}/AUC"]
                    for s in range(1, len(segments) + 1)
                    if metrics_to_log.get(f"Segment {segments[s-1]}/AUC", False)
                ]
            ).mean(),
            3,
        )

    return metrics_to_log, best_threshold


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def normalize_per_image(x):
    maxi = x.amax(dim=(-2, -1), keepdim=True)
    mini = x.amin(dim=(-2, -1), keepdim=True)
    return (x - mini) / (maxi - mini)


def wandb_log_data(
    images,
    preds,
    targets,
    samples_cnt: List[int],
    attention_weights: Optional[torch.tensor],
    prefix="Train",
):
    def prepare_image_for_logging(image, ind):
        h, w = 256, 256
        if len(image.shape) == 5:

            # image is ms, frames, 3, h, w
            image = image[: samples_cnt[ind]]  # ms, frames, 3, H, W
            image = image.flatten(0, 1)  # ms*frames, 3, H, W
            image = image[:, 0, :, :]  # ms*frames, H, W
            image = T.Resize((h, w), interpolation=T.InterpolationMode.NEAREST_EXACT)(
                image
            )  # ms*frames, h, w

            image = image.moveaxis(0, -2).flatten(1, 2)  # h, ms*frames*w
        else:
            # image is 3,h,w
            image = T.Resize((h, w), interpolation=T.InterpolationMode.NEAREST_EXACT)(
                image
            )
            image = image[0, :, :]  # h,w

        image = normalize(np.array(image))

        if attention_weights is not None:

            image_attn_weights = (
                attention_weights[ind, :, :, : samples_cnt[ind], :, :].detach().cpu()
            )  # transformer_heads,  queries, ms, frames_per_view, patch_size^2
            transformer_heads, queries, _, frames_per_view = image_attn_weights.shape[
                :4
            ]
            patch_size = int(math.sqrt(image_attn_weights.shape[-1]))
            image_attn_weights = image_attn_weights.reshape(
                transformer_heads * queries * samples_cnt[ind] * frames_per_view,
                1,
                patch_size,
                patch_size,
            )  # transformer_heads*queries* ms*frames_per_view, frames, ps, ps

            image_attn_weights = T.Resize(
                (h, w), interpolation=T.InterpolationMode.NEAREST_EXACT
            )(
                image_attn_weights
            )  # transformer_heads*queries* ms* frames_per_view, h, w

            image_attn_weights = normalize_per_image(image_attn_weights)
            image_attn_weights = image_attn_weights.reshape(
                transformer_heads, queries, samples_cnt[ind], frames_per_view, h, w
            )
            # flatten out
            image_attn_weights = image_attn_weights.flatten(
                2, 3
            )  # transformer_heads, queries, ms* frames_per_view, h, w
            image_attn_weights = image_attn_weights.moveaxis(2, 3).flatten(
                3, 4
            )  # transformer_heads, queries, h,  ms* frames_per_view*w
            image_attn_weights = image_attn_weights.flatten(
                1, 2
            )  # transformer_heads, queries*h, ms* frames_per_view*w
            # add separator

            separator = torch.ones(transformer_heads, 5, image_attn_weights.shape[-1])
            image_attn_weights = torch.cat((image_attn_weights, separator), axis=1)

            image_attn_weights = image_attn_weights.flatten(0, 1)  # queries*h, w*ms

            image = np.concatenate((image, image_attn_weights), axis=0)
        return Image.fromarray((255 * image).astype(np.uint8))

    # apply sigmoid
    preds = np.array(F.sigmoid(preds))
    # round
    preds = np.round(preds).astype(np.int32)
    targets = np.round(np.array(targets)).astype(np.int32)
    wandb.log(
        {
            f"Media/{prefix}": [
                wandb.Image(
                    prepare_image_for_logging(image, ind),
                    caption=f"Predicted: {preds[ind]}, Actual: {targets[ind]}",
                )
                for ind, image in enumerate(images[:8])
            ]
        }
    )


class EarlyStopper:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_metric = 0

    def early_stop(self, validation_metric):
        if validation_metric >= self.max_metric:
            self.max_metric = validation_metric
            self.counter = 0
        elif validation_metric < (self.max_metric - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def plot_cosine_similarity(pos_embeddings):
    x_normalized = F.normalize(pos_embeddings, p=2, dim=1)  # shape: (N, K)

    # Compute cosine similarity as dot product between normalized vectors
    cos_sim = np.array(x_normalized @ x_normalized.T)  # shape: (N, N)
    # Resize to (n x n)
    if cos_sim.shape[0] < 256:
        cos_sim = cv2.resize(cos_sim, (256, 256), interpolation=cv2.INTER_NEAREST)

    colored = plt.get_cmap("viridis")(cos_sim)

    # Convert to uint8 RGB image
    return Image.fromarray((colored[:, :, :3] * 255).astype(np.uint8))


def plot_actual_values(pos_embeddings):
    pos_embeddings = np.array(normalize(pos_embeddings))
    n, d = pos_embeddings.shape
    pos_embeddings = cv2.resize(pos_embeddings, (d, d), interpolation=cv2.INTER_NEAREST)

    colored = plt.get_cmap("viridis")(pos_embeddings)
    return Image.fromarray((colored[:, :, :3] * 255).astype(np.uint8))


def plot_l2_distance(pos_embeddings):
    x_norm_sq = (pos_embeddings**2).sum(dim=1, keepdim=True)  # (N, 1)

    # Compute squared distances using broadcasting:
    # dist_sq[i,j] = ||x_i||^2 + ||x_j||^2 - 2 * x_i.x_j
    dist_sq = x_norm_sq + x_norm_sq.T - 2 * (pos_embeddings @ pos_embeddings.T)

    # Clamp to zero (to avoid negative due to numerical error)
    dist_sq = torch.clamp(dist_sq, min=0.0)

    # Take sqrt to get L2 distances
    dist = np.array(torch.sqrt(dist_sq))
    if dist.shape[0] < 256:
        dist = cv2.resize(dist, (256, 256), interpolation=cv2.INTER_NEAREST)
    dist = normalize(dist)

    colored = plt.get_cmap("viridis")(dist)  # shape: (H, W, 4), RGBA format

    # Convert to uint8 RGB image
    return Image.fromarray((colored[:, :, :3] * 255).astype(np.uint8))


def plot_time_embeddings(time_embeddings: torch.tensor, prefix: str):
    wandb.log(
        {f"{prefix}/Cosine": wandb.Image(plot_cosine_similarity(time_embeddings))}
    )
    wandb.log({f"{prefix}/Values": wandb.Image(plot_actual_values(time_embeddings))})


def plot_pos_embeddings(pos_embeddings: torch.tensor, prefix: str):
    if pos_embeddings is None:
        return
    # Normalize each row to unit vector (L2 norm)
    pos_embeddings = pos_embeddings.detach().cpu()
    plot_cuts = 3

    for fixed_x in np.linspace(
        0, pos_embeddings.shape[0] - 1, plot_cuts, dtype=np.int32
    ):
        wandb.log(
            {
                f"{prefix} Cosine/X={fixed_x+1}": wandb.Image(
                    plot_cosine_similarity(pos_embeddings[fixed_x, :])
                )
            }
        )
        wandb.log(
            {
                f"{prefix} Values/X={fixed_x+1}": wandb.Image(
                    plot_actual_values(pos_embeddings[fixed_x, :])
                )
            }
        )
    for fixed_y in np.linspace(
        0, pos_embeddings.shape[1] - 1, plot_cuts, dtype=np.int32
    ):
        wandb.log(
            {
                f"{prefix} Cosine/Y={fixed_y+1}": wandb.Image(
                    plot_cosine_similarity(pos_embeddings[:, fixed_y])
                )
            }
        )
        wandb.log(
            {
                f"{prefix} Values/Y={fixed_y+1}": wandb.Image(
                    plot_actual_values(pos_embeddings[:, fixed_y])
                )
            }
        )


if __name__ == "__main__":
    get_artery_mapping("proximal")
    get_artery_mapping("all")
