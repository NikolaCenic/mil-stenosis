from datasets_utils import read_data
from image_datasets import BaseDataset, Transform
import pandas as pd
import torch
import numpy as np
import os
import json
from sklearn.cluster import KMeans
from PIL import Image
from typing import List
from ast import literal_eval
from torch.utils.data import Dataset
import cv2
from constants import ARTERY_SEGMENTS
from omegaconf import OmegaConf


class MILDataset(BaseDataset):
    def __init__(
        self,
        backbone_type: str,
        setting: str,
        resolution: int,
        severity: int,
        segment_based: bool,
        artery_based: bool,
        segments: List[int],
        multiframe: OmegaConf,
        augmentation: str = "base",
        data_file=None,
    ):
        self.segment_based = segment_based
        self.artery_based = artery_based
        self.first_frame = lambda x: max(x - 10, 0)
        self.multiframe = multiframe
        assert (
            self.multiframe.frames_per_view
            == 1 + self.multiframe.left + self.multiframe.right
        )
        super().__init__(
            backbone_type=backbone_type,
            setting=setting,
            resolution=resolution,
            only_key_frames=False,
            severity=severity,
            frames_per_view=multiframe.frames_per_view,
            segments=segments,
            augmentation=augmentation,
            data_file=data_file,
        )
        samples_per_patient = [len(i["views_and_key_frames"]) for i in self.samples]
        self.max_samples = max(samples_per_patient)
        self.min_samples = min(samples_per_patient)
        print(f"Views per patient: {self.min_samples} - {self.max_samples}")

    def get_class_weights(self):
        print("Getting label weights..")
        bin_count = torch.stack([torch.tensor(s["label"]) for s in self.samples]).sum(0)

        pos_counts = bin_count
        neg_counts = len(self.samples) - bin_count
        if self.artery_based or self.segment_based:
            assert all(pos_counts + neg_counts == len(self.samples))
        else:
            assert pos_counts + neg_counts == len(self.samples)
        total = pos_counts + neg_counts + 1
        weight_pos = total / (pos_counts + 1)
        weight_neg = total / (neg_counts + 1)
        return weight_pos, weight_neg

    def _get_segment_annotations_for_patient_(self, df):
        assert (
            len(
                df["annotation"]
                .apply(lambda x: tuple([v["STENOSIS_DEGREE"] for v in x]))
                .unique()
            )
            == 1
        )
        return {
            sa["ID_SEGMENT"]: sa["STENOSIS_DEGREE"] for sa in df["annotation"].iloc[0]
        }

    def _get_patient_label_(self, df: pd.DataFrame):

        stenosis_in_segments = self._get_segment_annotations_for_patient_(df)
        assert len(stenosis_in_segments) == 16 and sorted(
            list(stenosis_in_segments.keys())
        ) == list(range(1, 17))

        if self.setting == "gold_standard":
            cls_label = max(stenosis_in_segments.values()) >= self.severity
        else:
            major_segment_max_stenosis = max(
                [v for k, v in stenosis_in_segments.items() if k in self.segments]
            )

            minor_segment_max_stenosis = max(
                [v for k, v in stenosis_in_segments.items() if k not in self.segments],
                default=0,
            )
            if major_segment_max_stenosis < self.severity < minor_segment_max_stenosis:
                return None, None

            cls_label = major_segment_max_stenosis >= self.severity

        artery_labels = self.get_artery_labels(stenosis_in_segments)
        segments_labels = [
            stenosis_in_segments[i] >= self.severity for i in self.segments
        ]
        stenosis_percentage = [stenosis_in_segments[i] for i in self.segments]
        if self.artery_based and self.segment_based:
            return [cls_label, *artery_labels, *segments_labels], stenosis_percentage
        elif self.artery_based:
            return [cls_label, *artery_labels], stenosis_percentage
        elif self.segment_based:
            return [cls_label, *segments_labels], stenosis_percentage

        else:
            return cls_label, stenosis_percentage

    def get_artery_labels(self, stenosis_in_segments):
        arterty_labels = []
        for _, artery_segments in ARTERY_SEGMENTS.items():
            arterty_labels.append(
                any(
                    [
                        stenosis_in_segments[s] >= self.severity
                        for s in self.segments
                        if s in artery_segments
                    ]
                )
            )
        return arterty_labels

    def load_samples(self):
        def get_angulation(x):

            dicom_meta = metadata[x["siud"]][x["view"]].get("dicom_meta", {})
            primary_angle = literal_eval(
                dicom_meta.get("Positioner Primary Angle", "None")
            )
            secondary_angle = literal_eval(
                dicom_meta.get("Positioner Secondary Angle", "None")
            )
            if primary_angle is not None and secondary_angle is not None:
                return primary_angle, secondary_angle
            else:
                return False

        data, metadata = read_data(self.data_file)
        data["angulation"] = data.apply(lambda x: get_angulation(x), axis=1)

        total_views = len(data)
        data = data[data["angulation"] != False]
        print(f"{len(data)}/{total_views} have angulation information!")
        self.samples = []
        for siud, siud_data in data.groupby("siud"):
            siud_data = siud_data.loc[siud_data["angulation"].sort_values().index]

            assert len(siud_data) == len(siud_data["view"].unique())
            label, stenosis_percentage = self._get_patient_label_(siud_data)
            if label is None:
                continue

            self.samples.append(
                {
                    "siud": siud,
                    "label": label,
                    "stenosis_percentage": stenosis_percentage,
                    "views_and_key_frames": [
                        {
                            "mmap_path": row["view_path"],
                            "key_frames": row["key_frames"],
                            "shape": row["shape"],
                            "angulation": row["angulation"],
                        }
                        for _, row in siud_data.iterrows()
                    ],
                }
            )

    def select_frames_to_use(self, key_frame, max_frame, min_frame):
        if self.multiframe.frames_per_view == 1:
            return [key_frame]
        else:
            left_frames = np.linspace(
                max(min_frame, key_frame - self.multiframe.spacing_left),
                key_frame,
                self.multiframe.left + 1,
            )[:-1]
            left_frames = np.round(left_frames).astype(np.int32)

            right_frames = np.linspace(
                key_frame,
                min(max_frame, key_frame + self.multiframe.spacing_right),
                self.multiframe.right + 1,
            )[1:]
            right_frames = np.round(right_frames).astype(np.int32)
            selected_frames = [*left_frames, key_frame, *right_frames]
            return selected_frames

    def load_view_data(self, x):
        frames_to_use = self.select_frames_to_use(
            x["key_frames"][0], max_frame=x["shape"][0] - 1, min_frame=0
        )
        mmap = np.memmap(
            x["mmap_path"], dtype="float32", mode="r", shape=tuple(x["shape"])
        )[frames_to_use]
        return self.image_transformer(mmap)

    def create_angulation_bins(self, n_bins):
        def sort_centroids(centroids):
            idx = np.lexsort((centroids[:, 1], centroids[:, 0]))
            return centroids[idx]

        if n_bins < 1:
            return None
        angulations = np.array(
            [v["angulation"] for s in self.samples for v in s["views_and_key_frames"]]
        )
        kmeans = KMeans(n_clusters=n_bins, random_state=42)
        kmeans.fit_predict(angulations)
        sorted_cluster_centroids = sort_centroids(kmeans.cluster_centers_)

        return torch.tensor(sorted_cluster_centroids).float()

    def __getitem__(self, ind):

        sample = self.samples[ind]
        views_and_key_frames = sample["views_and_key_frames"]
        images = torch.stack([self.load_view_data(v) for v in views_and_key_frames])
        angulations = torch.stack(
            [
                torch.as_tensor(v["angulation"], dtype=torch.float32)
                for v in views_and_key_frames
            ]
        )
        number_of_views = len(views_and_key_frames)
        if number_of_views < self.max_samples:
            padding_samples = self.max_samples - number_of_views

            images = torch.concat(
                [
                    images,
                    torch.zeros((padding_samples, *images[0].shape)),
                ]
            )
            angulations = torch.concat(
                [angulations, torch.zeros((padding_samples, *angulations[0].shape))]
            )

        label = torch.tensor(sample["label"], dtype=torch.float32)
        stenosis_percentage = torch.tensor(
            sample["stenosis_percentage"], dtype=torch.float32
        )
        return (
            images,
            label,
            number_of_views,
            sample["siud"],
            angulations,
            stenosis_percentage,
        )


class CADICAMILDataset(MILDataset):
    def __init__(
        self,
        backbone_type,
        resolution,
        severity,
        multiframe,
        no_stenosis_severity=None,
    ):
        self.no_stenosis_severity = (
            no_stenosis_severity if no_stenosis_severity is not None else severity
        )
        super().__init__(
            backbone_type=backbone_type,
            setting="test",
            resolution=resolution,
            severity=severity,
            segment_based=False,
            artery_based=False,
            segments=None,
            multiframe=multiframe,
            augmentation=False,
        )

    def get_data_file(self):
        return f"data_splits/cadica_test.csv"

    def load_view_data(self, x):
        frames_to_use = self.select_frames_to_use(
            x["key_frame"], max_frame=x["max_frame"], min_frame=x["min_frame"]
        )
        view_frames = x["frames"]

        frames = []
        for f in frames_to_use:
            frame_path = view_frames[view_frames["frame_index"] == f][
                "frame_path"
            ].iloc[0]
            frames.append(np.array(Image.open(frame_path)))

        return self.image_transformer(np.stack(frames))

    def load_samples(self):
        set_data = pd.read_csv(self.data_file)
        set_data["severities"] = set_data["severities"].apply(literal_eval)
        self.samples = []
        for patient, patient_data in set_data.groupby("patient"):
            views_and_key_frames = []
            for view, view_data in patient_data.groupby("view"):
                key_row = view_data[
                    view_data["surviving_pixels"] == view_data["surviving_pixels"].max()
                ].iloc[0]
                views_and_key_frames.append(
                    {
                        "view": view,
                        "frames": view_data[["frame_path", "frame_index"]],
                        "severity": view_data["severities"]
                        .apply(lambda x: max(x, default=0))
                        .max(),
                        "key_frame": key_row["frame_index"],
                        "min_frame": view_data["frame_index"].min(),
                        "max_frame": view_data["frame_index"].max(),
                    }
                )

            no_stenosis_views = [
                x
                for x in views_and_key_frames
                if x["severity"] < self.no_stenosis_severity
            ]
            if no_stenosis_views == views_and_key_frames:
                self.samples.append(
                    {
                        "siud": patient,
                        "label": 0,
                        "views_and_key_frames": no_stenosis_views,
                    }
                )
            elif max([x["severity"] for x in views_and_key_frames]) >= self.severity:
                self.samples.append(
                    {
                        "siud": patient,
                        "label": 1,
                        "views_and_key_frames": views_and_key_frames,
                    }
                )

    def __getitem__(self, ind):
        sample = self.samples[ind]
        views_and_key_frames = sample["views_and_key_frames"]
        images = torch.stack([self.load_view_data(v) for v in views_and_key_frames])

        number_of_views = len(views_and_key_frames)
        if number_of_views < self.max_samples:
            padding_samples = self.max_samples - number_of_views
            images = torch.concat(
                [images, torch.zeros((padding_samples, *images[0].shape))]
            )

        label = torch.tensor(sample["label"], dtype=torch.float32)
        return (
            images,
            label,
            number_of_views,
            sample["siud"],
            torch.zeros((self.max_samples, 2)),
            torch.zeros(1),
        )


class SegmentationDataset(Dataset):
    def __init__(self, task, setting, backbone_type, resolution):
        assert task in ["stenosis", "syntax"]
        self.data_dir = f"/opt/data/arcade/{task}/{setting}"
        self.setting = setting
        self.image_transformer = Transform(
            backbone_type=backbone_type,
            setting=setting,
            resolution=resolution,
            frames_per_view=1,
            augmentation=False,
        )
        self.load_samples()

    def load_samples(self):
        with open(
            os.path.join(self.data_dir, "annotations", f"{self.setting}.json"), "r"
        ) as f:
            annotations = json.load(f)
        samples = {
            i["id"]: {
                "path": os.path.join(self.data_dir, "images", i["file_name"]),
                "segmentations": [],
                "shape": (i["height"], i["width"]),
            }
            for i in annotations["images"]
        }

        for annotation in annotations["annotations"]:
            samples[annotation["image_id"]]["segmentations"].extend(
                annotation["segmentation"]
            )
        samples = {k: samples[k] for k in sorted(samples.keys())}
        self.samples = list(samples.values())

        print(f"{self.__len__()} sample loaded from {self.data_dir}!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, ind):
        sample = self.samples[ind]
        x = self.image_transformer(torch.tensor(np.array(Image.open(sample["path"]))))
        mask = np.zeros(sample["shape"])
        segmentations = [
            np.array(seg, dtype=np.int32).reshape(-1, 1, 2)
            for seg in sample["segmentations"]
        ]
        cv2.fillPoly(mask, segmentations, color=1)
        mask = cv2.resize(mask, x.shape[1:], interpolation=cv2.INTER_NEAREST)
        return x, torch.tensor(mask).float()
