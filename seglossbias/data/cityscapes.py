"""
File: cityscapes.py
Author: Bingyuan Liu
Date: Feb 12, 2021
Brief: dataset wrapper for cityscapes datase
"""

import os
import os.path as osp
import logging
from PIL import Image
import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F

from utils.file_io import load_list, load_leison_classes
from utils.visualizer import Visualizer
import dataset.paired_transforms_tv04 as p_tr
from utils.colormap import colormap


logger = logging.getLogger(__name__)

_EPS = 1e-10


class CityscapesDataset(Dataset):

    PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
               [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
               [0, 80, 100], [0, 0, 230], [119, 11, 32]]
    IDS = [
        7, 8, 11, 12, 13, 17,
        19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 31, 32,
        33
    ]

    def __init__(self, data_root : str,
                 split : str = "train",
                 transforms : Optional[p_tr.Compose] = None,
                 return_id : bool = False) -> None:
        assert split in [
            "train", "val", "test"
        ], "Split '{}' not supported".format(split)
        self.data_root = data_root
        self.split = split
        self.transforms = transforms
        self.img_dir = osp.join(data_root, "leftImg8bit", split)
        self.anno_dir = osp.join(data_root, "gtFine", split)
        self.return_id = return_id
        self.samples = self.load_semantic_files()

        self.classes = [
            'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
            'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
            'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
            'bicycle',
        ]
        self.num_classes = len(self.classes)

    def load_semantic_files(self):
        img_suffix = "leftImg8bit.png"
        anno_suffix = "gtFine_labelTrainIds.png"

        samples = []
        cities = sorted(os.listdir(self.img_dir))
        logger.info("Cityscapes : {} cities found in {}".format(
            len(cities), self.split))
        for city in cities:
            city_img_dir = osp.join(self.img_dir, city)
            city_anno_dir = osp.join(self.anno_dir, city)
            for basename in sorted(os.listdir(city_img_dir)):
                img_file = osp.join(city_img_dir, basename)
                assert basename.endswith(img_suffix), basename
                basename = basename[: -len(img_suffix)]
                anno_file = osp.join(city_anno_dir, basename + anno_suffix)
                assert osp.isfile(anno_file), anno_file
                samples.append((img_file, anno_file, basename))
        assert len(samples), "No samples found"
        logger.info("Cityscapes : {} samples found in {}".format(
            len(samples), self.split))

        return samples

    def __getitem__(self, index : int):
        img_file, anno_file, sample_id = self.samples[index]
        # img = Image.open(img_file).convert("RGB")
        # mask = Image.open(anno_file)
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(anno_file, cv2.IMREAD_GRAYSCALE)

        if self.transforms is not None:
            result = self.transforms(image=img, mask=mask)
            img = result["image"]
            mask = result["mask"].long()

        if self.return_id:
            return (img, mask, sample_id)
        else:
            return (img, mask)

    def __len__(self) -> int:
        return len(self.samples)


if __name__ == "__main__":
    from utils.visualizer import image_mask_show
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    data_root = "./data/cityscapes"

    data_transform = A.Compose([
        A.OneOrOther(
            A.Resize(768, 768, interpolation=cv2.INTER_CUBIC),
            A.Sequential([
                A.RandomScale([0.5, 2.0], interpolation=cv2.INTER_CUBIC, p=1.),
                A.RandomCrop(768, 768),
            ], p=1),
            p=0.3
        ),
        A.HorizontalFlip(),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.HueSaturationValue(),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
        A.Normalize(),
        ToTensorV2()
    ])

    dataset = CityscapesDataset(
        data_root, split="train", transforms=None, return_id=True)

    class_area = np.zeros(dataset.num_classes + 1)
    total_area = 0
    class_freq = [None] * (dataset.num_classes + 1)

    for i in range(len(dataset)):
        img, mask, sample_id = dataset[i]
        area = mask.shape[0] * mask.shape[1]
        total_area += area
        labels, counts = np.unique(mask, return_counts=True)
        for k in range(labels.shape[0]):
            l = labels[k]
            # if l == 255:
            #     class_area[-1] += counts[k]
            #     class_freq[-1].append(counts[k] / area)
            if l != 255:
                class_area[l] += counts[k]
                if class_freq[l] is not None:
                    class_freq[l].append(counts[k] / area)
                else:
                    class_freq[l] = [counts[k] / area]

        # result = data_transform(image=img, mask=mask) 
        # new_img = result["image"]
        # new_mask = result["mask"]
        # print(new_img.shape, new_mask.shape)
        # image_mask_show(img, ask, palette=CityscapesDataset.PALETTE)
    # print("total area : ")
    # print(total_area, np.sum(cass_area))
    # print("class ratio : "
    # class_ratio = class_area / total_area
    # print(class_ratio.sum())
    # for i in range(dataset.num_classes):
    #     print(dataset.classes[i], class_ratio[i])
    #     freq = np.array(class_freq[i])
    #     print("{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(
    #         dataset.classes[i], np.max(freq), np.min(freq),
    #         np.mean(freq), np.median(freq))
    #     )

    # print(len(dataset), len(dataset) * 1024 * 2048)

    # import pickle
    # with open("cityscapes_classes_ratrios.p", "wb") as fp:
    #     pickle.dump(class_freq, fp)

    print("done")
