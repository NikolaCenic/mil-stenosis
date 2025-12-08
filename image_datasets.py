from torch.utils.data import Dataset
import warnings
from datasets_utils import (
    read_data,
    Transform,
)
import numpy as np
from ast import literal_eval
import pandas as pd
import torch
import random
import json
from typing import List
from PIL import Image


class BaseDataset(Dataset):
    def __init__(
        self,
        backbone_type: str,
        setting: str,
        resolution: int,
        severity: int,
        frames_per_view: int,
        segments: str,
        only_key_frames: bool = False,
        augmentation: str = "base",
        data_file=None,
    ):
        self.segments = segments
        self.only_key_frames = only_key_frames
        self.severity = severity
        self.setting = setting
        self.resolution = resolution

        self.data_file = self.get_data_file() if data_file is None else data_file
        self.image_transformer = Transform(
            backbone_type=backbone_type,
            setting=setting,
            resolution=resolution,
            frames_per_view=frames_per_view,
            augmentation=augmentation,
        )
        self.load_samples()

        self.get_label = lambda x: (
            x["label"][0] if isinstance(x["label"], list) else x["label"]
        )

        self.no_stenosis_cnt = len([i for i in self.samples if not self.get_label(i)])
        self.stenosis_cnt = len([i for i in self.samples if self.get_label(i)])

        print(
            f"{setting.capitalize()} Total: {self.__len__()}    0: {self.no_stenosis_cnt}  1: {self.stenosis_cnt}"
        )

    def get_data_file(self):
        return f"data_splits/{self.setting}.csv"

    def balance_dataset(self):
        def select_samples(samples, sample_cnt):
            random.Random(2024).shuffle(samples)
            return samples[:sample_cnt]

        no_stenosis = [i for i in self.samples if not self.get_label(i)]
        stenosis = [i for i in self.samples if self.get_label(i)]
        samples_per_class = min(len(no_stenosis), len(stenosis))
        self.samples = select_samples(
            no_stenosis, sample_cnt=samples_per_class
        ) + select_samples(stenosis, sample_cnt=samples_per_class)

    def __len__(self):
        return len(self.samples)

    def load_samples(self):
        raise NotImplementedError()

    def __getitem__(self, ind):
        raise NotImplementedError()


class SingleSampleDataset(BaseDataset):
    def __init__(
        self,
        backbone_type: str,
        setting: str,
        resolution: int,
        severity: int,
        segments: List[int],
        augmentation: str = "base",
    ):
        super().__init__(
            backbone_type=backbone_type,
            setting=setting,
            resolution=resolution,
            only_key_frames=setting != "train",
            frames_per_view=1,
            severity=severity,
            multiframe={},
            augmentation=augmentation,
            segments=segments,
        )

    def get_frame_label(self, frame_annotation):
        if self.setting == "test":
            return int(
                any([y["STENOSIS_DEGREE"] >= self.severity for y in frame_annotation])
            )
        else:
            return frame_annotation

    def load_samples(self):
        if self.setting == "test":
            warnings.warn(
                "Using SingleSampleDataset in test mode. Only use it for view level testing!"
            )
        data, _ = read_data(self.data_file)
        self.samples = []
        for _, row in data.iterrows():
            stenotic_segments = {}
            for segment_annotation in row["annotation"]:
                major_segment = segment_annotation["ID_SEGMENT"] in self.segments
                stenosis_in_segment = (
                    segment_annotation["STENOSIS_DEGREE"] >= self.severity
                )
                if not major_segment and stenosis_in_segment:
                    continue
                if major_segment:
                    stenotic_segments[segment_annotation["ID_SEGMENT"]] = (
                        stenosis_in_segment
                    )
            doctor_label = int(any(stenotic_segments.values()))
            view_annotations = (
                row["view_stenotic_frames"]
                if self.setting == "test"
                else row["pseudo_labels"]
            )
            for frame, frame_annotation in view_annotations.items():
                frame_label = self.get_frame_label(frame_annotation)
                if doctor_label == 0 or doctor_label == frame_label:
                    self.samples.append(
                        {
                            "siud": row["siud"],
                            "mmap_path": row["view_path"],
                            "label": doctor_label,
                            "frame": frame,
                            "shape": row["shape"],
                        }
                    )

    def __getitem__(self, ind):
        sample = self.samples[ind]

        frame = np.memmap(
            sample["mmap_path"], dtype="float32", mode="r", shape=tuple(sample["shape"])
        )[sample["frame"]]

        frame_transformed = self.image_transformer(frame)

        label = torch.tensor(sample["label"], dtype=torch.float32)
        return frame_transformed, label, 1, sample["siud"]


class InternalViewLevel:
    def __init__(
        self,
        backbone_type: str,
        resolution: int,
        segments: List[int],
        severity: int = 70,
    ):
        self.segments = segments
        self.get_label = lambda x: x["severity"] >= severity

        self.load_samples()
        self.transform = Transform(
            backbone_type=backbone_type,
            setting="test",
            resolution=resolution,
            frames_per_view=1,
        )

        self.no_stenosis_cnt = len([i for i in self.samples if not self.get_label(i)])
        self.stenosis_cnt = len([i for i in self.samples if self.get_label(i)])

        print(
            f"View Level Gold standard test-set Total: {self.__len__()}    0: {self.no_stenosis_cnt}  1: {self.stenosis_cnt}"
        )

    def load_samples(self):
        with open("data_splits/METADATA.json", "r") as f:
            metadata = json.load(f)

        def assign_angulation(row):
            dicom_meta = metadata[row["siud"]][row["view"]]["dicom_meta"]
            return float(dicom_meta["Positioner Primary Angle"]), float(
                dicom_meta["Positioner Secondary Angle"]
            )

        data = pd.read_csv("data_splits/view_level_test.csv")
        data["severity"] = data["severity"].apply(literal_eval)

        self.samples = []
        for (p, v), pv_data in data.groupby(["siud", "view"]):

            max_surviving_pixels = pv_data["surviving_pixels"].max()
            if max_surviving_pixels == 0:
                continue
            key_frame_data = pv_data[
                pv_data["surviving_pixels"] == max_surviving_pixels
            ]
            severity = key_frame_data["severity"].max()

            assert len(key_frame_data["frame_index"].unique()) == 1

            key_frame_data = key_frame_data.iloc[0]
            self.samples.append(
                {
                    "view": key_frame_data["view"],
                    "segment": key_frame_data["segment"],
                    "frame_path": key_frame_data["frame_path"],
                    "angulation": assign_angulation(key_frame_data),
                    "severity": severity[0],
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, ind):
        sample = self.samples[ind]
        image = self.transform(np.array(Image.open(sample["frame_path"])))
        id = sample["view"]
        samples_cnt = 1
        angulation = torch.tensor(sample["angulation"]).float().reshape(1, 2)
        severity = sample["severity"]
        label = torch.tensor(self.get_label(sample)).float()
        return (
            image,
            label,
            samples_cnt,
            id,
            angulation,
            severity,
        )


class CadicaViewLevel(InternalViewLevel):
    def __init__(self, backbone_type: str, resolution: int, severity: int = 70):
        self.severity = severity
        self.load_samples()
        self.transform = Transform(
            backbone_type=backbone_type,
            setting="test",
            resolution=resolution,
            frames_per_view=1,
        )

        self.get_label = lambda x: x["severity"] >= severity

        self.no_stenosis_cnt = len([i for i in self.samples if not self.get_label(i)])
        self.stenosis_cnt = len([i for i in self.samples if self.get_label(i)])

        print(
            f"View Level Cadica test-set Total: {self.__len__()}    0: {self.no_stenosis_cnt}  1: {self.stenosis_cnt}"
        )

    def load_samples(self):
        data = pd.read_csv("data_splits/cadica_test.csv")

        data = data[data["annotated"]]
        data["severities"] = data["severities"].apply(literal_eval)
        self.samples = []
        for _, pv_data in data.groupby(["patient", "view"]):
            pv_data = pv_data.iloc[0].to_dict()
            pv_data["angulation"] = [0, 0]
            pv_data["severity"] = max(pv_data["severities"], default=0)
            del pv_data["severities"]

            self.samples.append(pv_data)
