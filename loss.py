import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf


class WeightedBCE(nn.Module):
    def __init__(
        self,
        device,
        frequency_weights: torch.tensor,
        loss_config: OmegaConf,
    ):
        super(WeightedBCE, self).__init__()
        self.loss_config = loss_config
        self.set_lambdas()

        if self.loss_config.use_frequency_weights:
            frequency_weights /= frequency_weights.max()
            if self.cls_lambda > 0:
                frequency_weights[0] = 1
            if self.artery_lambda > 0:
                frequency_weights[1:3] /= frequency_weights[1:3].max()
            if self.segment_lambda > 0:
                frequency_weights[3:] /= frequency_weights[3:].max()
        else:
            frequency_weights = torch.ones(frequency_weights.shape)
        self.frequency_weights = frequency_weights.to(device)
        print(
            f"Loss lambdas: cls {self.cls_lambda}, artery: {self.artery_lambda}, segment: {self.segment_lambda}"
        )

    def set_lambdas(self):

        self.cls_lambda = self.loss_config.cls_lambda
        self.artery_lambda = self.loss_config.artery_lambda
        self.segment_lambda = self.loss_config.segment_lambda
        assert (
            abs(sum([self.cls_lambda, self.artery_lambda, self.segment_lambda]) - 1)
            < 1e-4
        )

    def forward(self, logits, targets):

        loss = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction="none", pos_weight=self.frequency_weights
        )
        if len(loss.shape) == 1:
            loss = loss[:, None]

        total_loss, loss_dict = 0, {}
        if self.cls_lambda > 0:
            cls_loss = loss[:, 0].mean()
            loss_dict.update(
                {"cls_loss": cls_loss, "scaled_cls_loss": self.cls_lambda * cls_loss}
            )
            total_loss += loss_dict["scaled_cls_loss"]
        if self.artery_lambda > 0:
            artery_loss = loss[:, 1:3].mean()
            loss_dict.update(
                {
                    "artery_loss": artery_loss,
                    "scaled_artery_loss": self.artery_lambda * artery_loss,
                }
            )
            total_loss += loss_dict["scaled_artery_loss"]
        if self.segment_lambda > 0:
            segment_loss = loss[:, 3:].mean()
            loss_dict.update(
                {
                    "segment_loss": segment_loss,
                    "scaled_segment_loss": self.segment_lambda * segment_loss,
                }
            )
            total_loss += loss_dict["scaled_segment_loss"]
        loss_dict["total_loss"] = total_loss

        return total_loss, loss_dict
