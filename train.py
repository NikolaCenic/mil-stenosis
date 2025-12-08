import torch.optim as optim
from evaluate import (
    evaluate,
    EvaluationMode,
)
from loss import WeightedBCE
from utils import (
    compute_metrics,
    load_model,
    create_loaders,
    EarlyStopper,
    set_seed,
)

from ignite.handlers.checkpoint import ModelCheckpoint
import os
import wandb
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import autocast, GradScaler
import torch
from tqdm import tqdm
import logging
import shutil
import random
import string

import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


def create_optimizer(model: torch.nn.Module, config: DictConfig):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizers = {
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "rms_prop": optim.RMSprop,
        "SGD": optim.SGD,
    }
    optimizer = optimizers[config.optim.type.lower()](
        params, lr=config.optim.lr, weight_decay=config.optim.weight_decay
    )
    logger.info(
        f"Optimizer type: {config.optim.type},  LR: {config.optim.lr} WD: {config.optim.weight_decay}"
    )
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    return optimizer, lr_scheduler


def adapt_config(config):
    config.multiframe.frames_per_view = (
        config.multiframe.left + config.multiframe.right + 1
    )
    if config.approach in ["max_mil", "attn_mil"]:
        config.loss.cls_lambda = 1
        config.loss.artery_lambda = 0
        config.loss.segment_lambda = 0
        config.segment_based = False
        config.artery_based = False
    elif config.approach == "segment_mil":
        config.segment_based = True
        config.artery_based = True

    return config


@hydra.main(config_path="configs/", config_name="config.yaml", version_base="1.3")
def train(config: DictConfig):
    config = adapt_config(config)
    run_id = "".join(random.choices(string.ascii_lowercase, k=4))
    timestamp = datetime.now().strftime("%m_%d_%H:%M:%S")
    run_name = f"{timestamp}-{run_id}"
    model_dir = os.path.join("stenosis-classification", run_name)
    os.makedirs(model_dir, exist_ok=True)

    set_seed(config.seed)

    device = (
        torch.device(config.device)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    train_loader, val_loader = create_loaders(config)

    model = load_model(config)
    model.to(device)

    optimizer, lr_scheduler = create_optimizer(model, config)

    logger.info(
        f"Model loaded from: {config.model.pretrained}, {model.count_params()/1_000_000}m trainable params"
    )

    frequency_weights, _ = train_loader.dataset.get_class_weights()
    criterion = WeightedBCE(
        device=device, frequency_weights=frequency_weights, loss_config=config.loss
    )

    scaler = GradScaler()
    checkpointer = ModelCheckpoint(
        model_dir,
        filename_prefix="best",
        score_function=lambda x: x["AUC"],
        score_name="auc",
        n_saved=1,
    )

    early_stoper = EarlyStopper(patience=25, min_delta=0.05)

    if config.model.freeze:
        model.freeze_encoder(config.model.freeze)

    try:
        for epoch in range(config.epochs):
            model.train()
            optimizer.zero_grad()
            train_epoch_loss = 0.0
            train_logits, train_targets, train_samples_cnt = [], [], []
            if config.model.unfreeze and epoch == config.model.unfreeze:
                model.unfreeze_encoder()
            for images, targets, samples_cnt, _, angulations, _ in tqdm(
                train_loader,
                total=len(train_loader),
                desc=f"Epoch {epoch+1} Train Loop",
            ):

                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                angulations = angulations.to(device, non_blocking=True)
                with autocast():
                    logits = model(images, samples_cnt, angulations)
                logits = logits.reshape(targets.shape)

                loss, _ = criterion(logits, targets)

                scaler.scale(loss / config.optim.grad_accumulation_steps).backward()
                train_loss = loss.item()
                train_epoch_loss += train_loss

                train_logits.extend(list(logits.cpu().detach()))
                train_targets.extend(list(targets.cpu().detach()))
                train_samples_cnt.extend(list(samples_cnt))

            lr_scheduler.step()

            model.eval()
            with torch.no_grad():
                val_epoch_loss = 0.0
                val_logits, val_targets, val_samples_cnt = [], [], []
                for (
                    images,
                    targets,
                    samples_cnt,
                    _,
                    angulations,
                    _,
                ) in tqdm(
                    val_loader,
                    total=len(val_loader),
                    desc=f"Epoch {epoch+1} Val Loop",
                ):
                    images = images.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    angulations = angulations.to(device, non_blocking=True)
                    with autocast():
                        logits = model(images, samples_cnt, angulations)
                        logits = logits.reshape(targets.shape)
                    val_loss, _ = criterion(logits, targets)
                    val_epoch_loss += val_loss.item()
                    val_logits.extend(list(logits.cpu()))
                    val_targets.extend(list(targets.cpu()))
                    val_samples_cnt.extend(list(samples_cnt))

            train_metrics, _ = compute_metrics(
                logits=train_logits,
                targets=train_targets,
                threshold=0.5,
                compute_segment_metrics=False,
            )
            val_metrics, _ = compute_metrics(
                logits=val_logits,
                targets=val_targets,
                threshold=0.5,
                compute_segment_metrics=False,
            )

            logger.info(
                f"Epoch {epoch+1}, Train Loss: {train_epoch_loss / len(train_loader)} AUC: {train_metrics['AUC']}"
            )
            logger.info(
                f"          Valid Loss: {val_epoch_loss / len(val_loader)} AUC: {val_metrics['AUC']}"
            )

            checkpointer(val_metrics, {"model": model})

            if config.early_stop and early_stoper.early_stop(val_metrics["AUC"]):
                print("-----------------     Early Stopping!      -----------------")
                break
    finally:
        if len(os.listdir(model_dir)) == 0:
            shutil.rmtree(model_dir)
            wandb.finish()
        else:
            OmegaConf.save(config, os.path.join(model_dir, "config.yaml"))
            torch.save(model.state_dict(), os.path.join(model_dir, "last.pt"))
            evaluate(
                model_dir,
                dataset="val",
                evaluation_mode=EvaluationMode.PatientLevel,
                bootstrap=False,
                segments=config.segments,
            )


if __name__ == "__main__":
    train()
