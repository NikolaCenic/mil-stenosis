import torch
from typing import List
import torch.nn as nn
import torchvision.models as models
import numpy as np
import os
import timm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from constants import ARTERY_SEGMENTS
from segment_attention_utils import (
    CustomTransformerDecoder,
    CustomTransformerDecoderLayer,
)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


class GridIndexer:
    def __init__(self, W, H):
        self.x_min = -70
        self.x_max = 110
        self.y_min = -50
        self.y_max = 50
        self.W = W
        self.H = H

        self.cell_width = (self.x_max - self.x_min) / W
        self.cell_height = (self.y_max - self.y_min) / H

    def get_index(self, angulations):
        assert len(angulations.shape) == 2
        x = angulations[:, 0]
        y = angulations[:, 1]
        col = (x - self.x_min) // self.cell_width
        row = (y - self.y_min) // self.cell_height

        row = torch.clip(row, 0, self.H - 1).int()
        col = torch.clip(col, 0, self.W - 1).int()
        return col, row


def sinusoidal_embeddings_1d(
    seq_len,
    d_model,
):
    assert d_model % 2 == 0, "Embedding dimension must be even for sin/cos pairs."

    positions = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)

    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )

    embeddings = torch.zeros((seq_len, d_model))
    embeddings[:, 0::2] = torch.sin(positions * div_term)
    embeddings[:, 1::2] = torch.cos(positions * div_term)

    return embeddings


def sinusoidal_embeddings_2d(height, width, d_model):
    if d_model % 4 != 0:
        raise ValueError(
            "Cannot use sin/cos positional encoding with "
            "odd dimension (got dim={:d})".format(d_model)
        )
    pe = torch.zeros(d_model, height, width)
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0.0, width).unsqueeze(1)
    pos_h = torch.arange(0.0, height).unsqueeze(1)
    pe[0:d_model:2, :, :] = (
        torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    )
    pe[1:d_model:2, :, :] = (
        torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    )
    pe[d_model::2, :, :] = (
        torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    )
    pe[d_model + 1 :: 2, :, :] = (
        torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    )
    return pe.moveaxis(0, -1)


class EncoderWrapper(nn.Module):
    def __init__(self, backbone, encode_level, backbone_type):
        super().__init__()
        self.backbone = backbone
        self.encode_level = encode_level
        self.backbone_type = backbone_type
        if backbone_type == "resnet" and encode_level == "global":
            self.adaptive_pool = torch.nn.AdaptiveAvgPool2d(output_size=(1))

    def forward(self, x):
        if self.backbone_type == "vit":
            x = self.backbone(x, is_training=True)
            if self.encode_level == "patch":
                return x["x_norm_patchtokens"]
            else:
                return x["x_norm_clstoken"]
        elif self.backbone_type == "vit_timm":
            x = self.backbone.forward_features(x)
            if self.encode_level == "patch":
                return x[:, 1:, :]
            else:
                return x[:, 0, :]
        else:
            x = self.backbone(x)
            if self.encode_level == "patch":
                return x.flatten(2, 3).moveaxis(2, 1)
            else:
                return self.adaptive_pool(x).view(x.size(0), x.size(1))


def load_dinov2_model(pretrained: bool | str):
    encoder = torch.hub.load(
        repo_or_dir="facebookresearch/dinov2", model="dinov2_vits14"
    )

    if isinstance(pretrained, str):
        assert os.path.isfile(pretrained)
        weights = torch.load(pretrained)
        encoder.load_state_dict(weights, strict=True)
        print(f"Pretrained VIT weights loaded from {pretrained} ")
    return encoder


class SingleInstanceClassifier(nn.Module):
    def __init__(
        self,
        backbone_type: str,
        pretrained: bool,
        encode_level: str,
        resolution: int,
        embed_dim: int,
        projector_layers: int,
        classifier_layers: int,
    ):
        super().__init__()

        self.resolution = resolution
        self.encode_level = encode_level
        self.embed_dim = embed_dim
        self.projector_layers = projector_layers
        self.classifier_layers = classifier_layers
        self.backbone_type = backbone_type
        self.create_backbone(pretrained=pretrained)
        self.classifier = self.create_classifier(
            cls_depth=self.classifier_layers,
            input=self.embed_dim,
            output=1,
        )

    def count_params(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])

    def create_backbone(self, pretrained: bool):
        assert self.backbone_type in ["resnet", "vit", "vit_timm"]
        assert self.encode_level in ["patch", "global"]
        if self.backbone_type == "vit":
            print("VIT BACKBONE USED!")
            backbone = load_dinov2_model(pretrained=pretrained)
            encoder_embed_dim = backbone.embed_dim
        elif self.backbone_type == "vit_timm":
            print(f"VIT TIMM PRETRAINED BACKBONE USED")
            backbone = timm.create_model(
                "vit_small_patch16_224", pretrained=pretrained, num_classes=0
            )
            encoder_embed_dim = backbone.embed_dim
        else:
            print("RESNET BACKBONE USED!")
            resnet50 = models.resnet50(
                progress=True,
                weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None,
            )
            backbone = torch.nn.Sequential(  # out is bs, 2048, h, w
                *(list(resnet50.children())[:-2]),
            )
            encoder_embed_dim = 2048

        self.encoder = EncoderWrapper(backbone, self.encode_level, self.backbone_type)

        if self.embed_dim == encoder_embed_dim:
            self.projector = nn.Identity()
        else:
            self.projector = self.create_classifier(
                cls_depth=self.projector_layers,
                input=encoder_embed_dim,
                output=self.embed_dim,
            )
            print("Projector weight init!")
            self.projector.apply(init_weights)

    def freeze_encoder(self, blocks_to_freeze):
        if blocks_to_freeze == "full":
            print(f"FULL ENCODER FROZEN!")
            for p in self.encoder.parameters():
                p.requires_grad = False
        else:
            print(f"{blocks_to_freeze} ENCODER BLOCKS FROZEN!")
            for i, block in enumerate(self.encoder.backbone.blocks):
                if i < blocks_to_freeze:
                    for param in block.parameters():
                        param.requires_grad = False

    def unfreeze_encoder(self):
        print(f"ENCODER UNFROZEN!")
        for param in self.encoder.parameters():
            param.requires_grad = True

    def create_classifier(
        self, cls_depth: int, input: int, output: int, intermediate: int = None
    ):

        if cls_depth == 0:
            layers = [nn.Linear(input, output)]
        else:
            if intermediate is None:
                intermediate = input // 2
            layers = [nn.Linear(input, intermediate), nn.ReLU()]

            for i in range(cls_depth - 2):
                layers.extend(
                    [
                        nn.Linear(intermediate, intermediate),
                        nn.ReLU(),
                    ]
                )

            layers.extend([nn.Linear(intermediate, output)])
        classifier = nn.Sequential(*layers)
        classifier.apply(init_weights)
        return classifier

    def forward_encoder(self, x):
        return self.projector(self.encoder(x))

    def forward(self, x: torch.tensor, sample_cnt=None, patient_level=False):
        if patient_level:  # x is bc, ms, frames, 3, h, w
            assert not self.training
            # used in patient level evlaution only
            bs, ms, frames, _, _, _ = x.shape
            x = x.flatten(0, 2)  # bs*ms*frames
            ys = self.classifier(self.forward_encoder(x)).reshape(bs, ms * frames)
            return ys.max(dim=1)
        else:
            return self.classifier(self.forward_encoder(x)).squeeze()


class MaxMIL(SingleInstanceClassifier):

    def merge_predictions(self, single_sample_pred: torch.tensor, samples_cnt: int):
        if self.training:
            return torch.logsumexp(single_sample_pred[:samples_cnt], dim=0)
        else:
            return torch.max(single_sample_pred[:samples_cnt])

    def forward(
        self,
        x: torch.tensor,
        samples_cnt: List[int],
        angulations: torch.tensor = None,
        return_attn_weights: bool = False,
    ):
        if len(x.shape) == 4:
            x = x.unsqueeze(1)
        bs, samples_per_patient, frames_per_view, c, h, w = x.shape

        x = x.flatten(0, 2)

        y = self.classifier(self.forward_encoder(x)).reshape(
            bs, samples_per_patient * frames_per_view
        )
        y = torch.stack(
            [
                self.merge_predictions(
                    single_sample_pred, samples_cnt[i] * frames_per_view
                )
                for i, single_sample_pred in enumerate(y)
            ]
        )
        return y


class ProbAttentionMIL(nn.Module):
    # https://arxiv.org/pdf/1802.04712 formula 7 and 8
    def __init__(self, hidden_dim):
        super().__init__()

        self.att_V = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.att_w = nn.Linear(hidden_dim, 1, bias=False)

        self.att_V.apply(init_weights)
        self.att_w.apply(init_weights)

    def forward(self, embeddings):
        if len(embeddings.shape) == 1:
            embeddings = embeddings.unsqueeze(1)
        # Compute attention scores
        H = torch.tanh(self.att_V(embeddings))  # [ms, attn_dim]
        A = self.att_w(H)  # [N]

        # Softmax on bag
        A = torch.softmax(A, dim=0)  # [N]
        # Weighted sum of original probabilities
        bag_embeddings = torch.sum(A * embeddings, dim=0)  # [B]
        return bag_embeddings


class AttnMIL(MaxMIL):
    def __init__(
        self,
        backbone_type,
        pretrained,
        encode_level,
        resolution,
        embed_dim,
        projector_layers,
        classifier_layers,
    ):
        super().__init__(
            backbone_type,
            pretrained,
            encode_level,
            resolution,
            embed_dim,
            projector_layers,
            classifier_layers,
        )
        self.prob_attn_mil = ProbAttentionMIL(embed_dim)

    def merge_predictions(self, embeddings: torch.tensor):
        return self.prob_attn_mil(embeddings)

    def forward(
        self,
        x: torch.tensor,
        samples_cnt: List[int],
        angulations: torch.tensor = None,
        return_attn_weights: bool = False,
    ):
        if len(x.shape) == 4:
            x = x.unsqueeze(1)
        bs, samples_per_patient, frames_per_view, c, h, w = x.shape

        x = x.flatten(0, 2)

        embeddings = self.forward_encoder(x).reshape(
            bs, samples_per_patient * frames_per_view, self.embed_dim
        )

        bag_embeddings = torch.stack(
            [
                self.merge_predictions(emb[: samples_cnt[i] * frames_per_view])
                for i, emb in enumerate(embeddings)
            ]
        )
        y = self.classifier(bag_embeddings)
        return y


class ClsMIL(SingleInstanceClassifier):
    def __init__(
        self,
        backbone_type: str,
        pretrained: bool,
        encode_level: str,
        embed_dim: int,
        resolution: int,
        projector_layers: int,
        classifier_layers: int,
        attention_type: str,
        attention_heads: int,
        transformer_layers: int,
        cnt_pos_embeddings: int,
        frames_per_view: int,
        num_queries: int = 1,
        handle_multiframe: str = "concat",
    ):
        super().__init__(
            backbone_type=backbone_type,
            pretrained=pretrained,
            encode_level=encode_level,
            embed_dim=embed_dim,
            resolution=resolution,
            projector_layers=projector_layers,
            classifier_layers=classifier_layers,
        )
        self.handle_multiframe = handle_multiframe
        self.num_queries = num_queries
        self.attention_type = attention_type
        self.frames_per_view = frames_per_view

        # initialize
        if attention_type == "mha":
            self.attention = nn.MultiheadAttention(
                self.embed_dim, attention_heads, batch_first=True
            )
            self.transformer_layers = 1
        elif attention_type == "transformer":

            decoder_layer = CustomTransformerDecoderLayer(
                d_model=self.embed_dim,
                nhead=attention_heads,
                batch_first=True,
                norm_first=True,
            )

            self.attention = CustomTransformerDecoder(
                decoder_layer, num_layers=transformer_layers
            )

            self.transformer_layers = transformer_layers
        else:
            raise ValueError(f"Attention type {attention_type} not supported!")
        self.create_pos_embeddings(cnt_pos_embeddings)
        self.create_queries()

    def create_pos_embeddings(self, cnt_pos_embeddings):
        def is_perfect_square(n):
            if n < 0:
                return False
            root = int(math.sqrt(n))
            return root * root == n

        self.embeddings_proj = nn.Identity()
        if cnt_pos_embeddings > 0:
            assert cnt_pos_embeddings in [16, 64, 128, 256, 512, 1024]
            if is_perfect_square(cnt_pos_embeddings):
                h = int(math.sqrt(cnt_pos_embeddings))
                w = h
            else:
                h = int(math.sqrt(cnt_pos_embeddings / 2))
                w = h * 2
            assert h * w == cnt_pos_embeddings
            self.pos_embeddings = nn.Parameter(
                sinusoidal_embeddings_2d(h, w, self.embed_dim), requires_grad=False
            )
            self.grid_indexer = GridIndexer(W=w, H=h)
            print(f"Fixed Pos Embeddings of shape: {self.pos_embeddings.shape}")
        else:
            self.pos_embeddings = None

        if self.frames_per_view > 1:
            assert self.handle_multiframe in [
                "max",
                "min",
                "learned_embedding",
                "fixed_embedding",
            ]
            if self.handle_multiframe == "learned_embedding":
                self.time_embeddings = nn.Parameter(
                    torch.empty(self.frames_per_view, self.embed_dim),
                    requires_grad=True,
                )
                nn.init.normal_(self.time_embeddings, mean=0.0, std=0.02)
                print(f"Learned Time Embeddings of shape: {self.time_embeddings.shape}")
            elif self.handle_multiframe == "fixed_embedding":
                self.time_embeddings = nn.Parameter(
                    sinusoidal_embeddings_1d(self.frames_per_view, self.embed_dim),
                    requires_grad=False,
                )
                self.embeddings_proj = self.create_classifier(
                    cls_depth=0, input=self.embed_dim, output=self.embed_dim
                )
                print(
                    f"Fixed Time Embeddings of shape: {self.time_embeddings.shape} and Embedding Projection"
                )

    def add_pos_embeddings(self, x, angulations):
        bs, ms, _ = angulations.shape
        embeddings = torch.zeros(x.shape).cuda()
        if self.pos_embeddings is not None:
            grid_x, grid_y = self.grid_indexer.get_index(angulations.flatten(0, 1))
            pe = self.pos_embeddings[grid_x, grid_y].reshape(
                bs, ms, 1, 1, self.embed_dim
            )
            embeddings += pe.repeat(
                1, 1, self.frames_per_view, self.tokens_per_frame, 1
            )
        if self.frames_per_view > 1:

            if self.handle_multiframe == "mean":
                x = x.mean(dim=2, keepdim=True)
            elif self.handle_multiframe == "max":
                x = x.max(dim=2, keepdim=True)[0]
            elif self.handle_multiframe in ["learned_embedding", "fixed_embedding"]:
                te = self.time_embeddings.reshape(
                    1, 1, self.frames_per_view, 1, self.embed_dim
                )

                embeddings += te.repeat(bs, ms, 1, self.tokens_per_frame, 1)

        if not torch.all(embeddings == 0):
            x += self.embeddings_proj(embeddings)

        return x

    def create_queries(self):
        self.query = torch.empty(self.num_queries, self.embed_dim)
        torch.nn.init.normal_(self.query)
        self.query /= np.sqrt(self.embed_dim)
        self.query = nn.Parameter(self.query)

    def forward_attention(self, query, embedding, angulations, samples_cnt):

        embedding = self.add_pos_embeddings(
            embedding, angulations
        )  # (bs, ms, fpv, t, embed_dim)
        src_key_padding_mask = self.get_key_padding_mask(
            embedding.shape, samples_cnt
        ).to(
            embedding.device
        )  # (bs, ms, fpv, t)

        embedding = embedding.flatten(1, 3)  # (bs, ms * fpv * t, embed_dim)
        src_key_padding_mask = src_key_padding_mask.flatten(1, 3)  # (bs, ms * fpv * t)

        if self.attention_type == "mha":
            y, attn_output_weights = self.attention(
                query=query,
                key=embedding,
                value=embedding,
                key_padding_mask=src_key_padding_mask,
            )
        elif self.attention_type == "transformer":
            y, attn_output_weights = self.attention(
                tgt=query,
                memory=embedding,
                memory_key_padding_mask=src_key_padding_mask,
            )

        attn_output_weights = attn_output_weights.reshape(
            self.bs,
            self.transformer_layers,
            self.num_queries,
            self.max_samples_per_patient,
            -1,
            self.tokens_per_frame,
        )
        if self.frames_per_view > 1 and self.handle_multiframe in ["max", "mean"]:
            attn_output_weights = attn_output_weights.repeat(
                1, 1, 1, 1, self.frames_per_view, 1
            )

        return y, attn_output_weights

    def get_key_padding_mask(
        self,
        embedding_shape,
        samples_cnt: List[int],
    ):

        attn_mask = torch.zeros(embedding_shape[:-1])
        for i in range(len(attn_mask)):
            attn_mask[i, samples_cnt[i] :] = 1
        return attn_mask.bool()

    def forward_full_backbone(
        self, x: torch.tensor, samples_cnt: List[int], angulations: torch.tensor
    ):
        # bs, samples_per_patient, frame_per_view, 3, h, w
        self.bs, self.max_samples_per_patient, _, c, h, w = x.shape

        x = x.flatten(0, 2)
        embedding = self.forward_encoder(x).reshape(
            self.bs,
            self.max_samples_per_patient,
            self.frames_per_view,
            -1,
            self.embed_dim,
        )

        self.tokens_per_frame = embedding.shape[3]

        y, attn_output_weights = self.forward_attention(
            self.query.repeat(self.bs, 1, 1), embedding, angulations, samples_cnt
        )
        return y, attn_output_weights

    def forward(
        self,
        x: torch.tensor,
        samples_cnt: List[int],
        angulations: torch.tensor,
        return_attn_weights: bool = False,
    ):
        y, attn_output_weights = self.forward_full_backbone(x, samples_cnt, angulations)
        y = self.classifier(y).squeeze()

        if return_attn_weights:
            return y, attn_output_weights
        else:
            return y


class SegmentMIL(ClsMIL):
    def __init__(
        self,
        backbone_type: str,
        pretrained: bool,
        encode_level: str,
        embed_dim: int,
        resolution: int,
        projector_layers: int,
        classifier_layers: int,
        attention_type: str,
        attention_heads: int,
        shared_classifier: bool,
        hierarchical: bool,
        transformer_layers: int,
        segments_to_use: List[int],
        frames_per_view: int,
        cnt_pos_embeddings: int,
        handle_multiframe: str = "concat",
    ):
        super().__init__(
            backbone_type=backbone_type,
            pretrained=pretrained,
            encode_level=encode_level,
            embed_dim=embed_dim,
            resolution=resolution,
            projector_layers=projector_layers,
            classifier_layers=classifier_layers,
            attention_type=attention_type,
            attention_heads=attention_heads,
            cnt_pos_embeddings=cnt_pos_embeddings,
            frames_per_view=frames_per_view,
            transformer_layers=transformer_layers,
            num_queries=3 + len(segments_to_use),
            handle_multiframe=handle_multiframe,
        )
        self.segments_to_use = segments_to_use
        self.view_evaluation = False
        self.hierarchical = hierarchical
        self.shared_classifier = shared_classifier
        self.create_segment_classifiers()

    def create_segment_classifiers(self):
        if self.shared_classifier:
            print("Shared classifier is used!")
        else:
            self.classifiers = nn.ModuleList(
                [
                    self.classifier,
                    *[
                        self.create_classifier(
                            cls_depth=self.classifier_layers,
                            input=self.embed_dim,
                            output=1,
                        )
                        for _ in range(self.num_queries - 1)
                    ],
                ]
            )

            print(
                f"{len(self.classifiers)-1} artery/segment specific classifiers created (+1 for CLS)"
            )

    def forward_classifiers(self, x: torch.tensor):
        def hierarchical_prediction(logits):
            if len(logits.shape) == 1:
                logits = logits.unsqueeze(0)

            cls_logit = logits[:, 0]
            rca_logits = logits[:, 1]
            lca_logits = logits[:, 2]

            rca_indices = [
                3 + ind
                for ind, s in enumerate(self.segments_to_use)
                if s in ARTERY_SEGMENTS["RCA"]
            ]
            lca_indices = [
                3 + ind
                for ind, s in enumerate(self.segments_to_use)
                if s in ARTERY_SEGMENTS["LCA"]
            ]

            logits[:, rca_indices] += rca_logits[:, None].repeat(1, len(rca_indices))
            logits[:, lca_indices] += lca_logits[:, None].repeat(1, len(lca_indices))
            logits[:, 1:] += cls_logit[:, None].repeat(1, self.num_queries - 1)
            return logits

        if self.shared_classifier:
            logits = self.classifier(x).squeeze()
        else:
            classifier_preds = [
                classifier(x[:, ind, :])  # bs, segment, enc
                for ind, classifier in enumerate(self.classifiers)
            ]
            logits = torch.stack(classifier_preds, dim=1).squeeze()

        if self.hierarchical:
            logits = hierarchical_prediction(logits)
        return logits

    def forward(
        self,
        x: torch.tensor,
        samples_cnt: List[int],
        angulations: torch.tensor,
        return_attn_weights: bool = False,
    ):
        y, attn_output_weights = self.forward_full_backbone(x, samples_cnt, angulations)
        y = self.forward_classifiers(y)
        if return_attn_weights:
            return y, attn_output_weights
        else:
            return y

    def special_video_level_evaluation(
        self, x: torch.tensor, samples_cnt: List[int], temperature: float
    ):
        y, attn_weights = self.forward(x, samples_cnt, return_attn_weights=True)
        cls_weights = attn_weights[:, 0, :]  # bs, max_samples
        cls_preds = torch.nn.functional.sigmoid(y[:, 0])
        predictions = []
        for batch_idx, single_sample_weights in enumerate(cls_weights):
            single_sample_weights = torch.nn.functional.softmax(
                single_sample_weights[: samples_cnt[batch_idx]] / temperature
            )
            predictions.append(single_sample_weights * cls_preds[batch_idx])
        return torch.cat(predictions)
