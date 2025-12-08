import torchvision.transforms.v2 as T
from torchvision import tv_tensors
from ast import literal_eval
import pandas as pd
from PIL import Image
import torch
import json
import torchio as tio


def read_data(filename: str):
    data = pd.read_csv(filename).fillna(value="0")
    with open("data_splits/METADATA.json", "r") as f:
        metadata = json.load(f)
    for col in [
        "annotation",
        "shape",
        "pseudo_labels",
        "view_stenotic_frames",
        "key_frames",
        "severity",
    ]:
        if col in data.columns:
            data[col] = data[col].apply(literal_eval)
    return data, metadata


def base_augmentation(target_shape: tuple):
    print("Base augmentation!")
    return T.Compose(
        [
            T.RandomResizedCrop(size=target_shape, scale=(0.5, 1.5)),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.0, hue=0.0),
            T.GaussianBlur(kernel_size=5, sigma=(1, 2)),
        ]
    )


def normalize_0_1(x):
    if 0 <= x.min() and x.max() <= 1:
        return x
    return (x - x.min()) / (x.max() - x.min())


def normalize_per_image(x):
    maxi = x.amax(dim=(-2, -1), keepdim=True)
    mini = x.amin(dim=(-2, -1), keepdim=True)
    return (x - mini) / (maxi - mini)


class Transform:
    def __init__(
        self,
        backbone_type: str,
        setting: str,
        resolution: int,
        augmentation: str = True,
        frames_per_view: int = 1,
    ):
        assert backbone_type in ["vit", "resnet", "vit_timm", "inception", "yolo"]

        self.resolution = resolution
        if backbone_type == "vit":
            final_shape = (518, 518)
        else:
            final_shape = (resolution, resolution)

        if setting == "train" and augmentation:
            if augmentation == "base":
                augmentation_transform = base_augmentation(final_shape)
            else:
                raise ValueError("Wrong augmentation!")
        else:
            augmentation_transform = T.Lambda(lambda x: x)

        self.transformer = T.Compose(
            [
                T.Lambda(lambda x: torch.tensor(x).unsqueeze(-3)),
                T.Lambda(normalize_0_1),
                self.resize(backbone_type),
                T.Lambda(lambda x: x.repeat_interleave(3, dim=-3)),
                T.Lambda(
                    lambda x: (
                        tv_tensors.Video(x)
                        if frames_per_view > 1
                        else tv_tensors.Image(x)
                    )
                ),
                augmentation_transform,
                self.normalizer(backbone_type),
            ]
        )

    def resize(self, backbone_type):
        if backbone_type == "vit":
            resizes = []
            if self.resolution != 518:
                resizes.append(T.Resize((self.resolution, self.resolution)))
            resizes.append(T.Resize((518, 518)))
            return T.Compose(resizes)
        else:
            return T.Resize((self.resolution, self.resolution))

    def normalizer(self, backbone_type):

        if backbone_type in ["inception", "vit", "resnet"]:
            return T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        elif backbone_type == "yolo":
            return T.Lambda(normalize_0_1)
        else:
            raise ValueError("Wrong backbone type!")

    def __call__(self, x):
        return self.transformer(x)


if __name__ == "__main__":
    torch.manual_seed(0)
    a = torch.rand(500, 500)
    for b in ["resnet", "vit"]:
        t = Transform(
            backbone_type=b, setting="train", resolution=224, augmentation="base"
        )
        y = t(a)
        print(b, y.shape, a.shape, y.min(), y.max())
