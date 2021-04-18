"""
File: cityscapes.py
Author: Bingyuan Liu
Date: Feb 12, 2021
Brief: dataset wrapper for cityscapes datase
"""

import os
import os.path as osp
import logging
import cv2
from torch.utils.data.dataset import Dataset

logger = logging.getLogger(__name__)


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

    classes = [
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
        'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
        'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
        'bicycle',
    ]

    def __init__(self, data_root : str,
                 split : str = "train",
                 transforms=None,
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
