import torch
import os
import os.path as osp
import cv2
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List, Optional, Any
from torch.utils.data.dataset import Dataset

from .data_transform import DATA_TRANSFORM


class PolypDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        set_name: str = "Kvasir",
        data_transformer: Optional[A.Compose] = None,
        return_id: bool = False,
    ) -> None:
        assert split in [
            "train", "val", "test"
        ], "Split '{}' is not supported".format(split)

        self.data_root = data_root
        self.split = split
        self.set_name = set_name
        self.data_transformer = data_transformer
        self.return_id = return_id
        
        self.load_list()
    
    def load_list(self):
        if self.split in ["train", "val"]:
            subdir = osp.join(self.data_root, "TrainDataset")
            self.img_dir = osp.join(subdir, "images")
            self.mask_dir = osp.join(subdir, "masks")
            self.images, self.masks = [], []
            # read lines from the file
            with open(osp.join(subdir, "{}.txt".format(self.split)), "r") as f:
                for line in f:
                    name = line.strip()
                    self.images.append(osp.join(self.img_dir, name))
                    self.masks.append(osp.join(self.mask_dir, name))
        else:
            subdir = osp.join(self.data_root, "TestDataset", self.set_name)
        
            self.img_dir = osp.join(subdir, "images")
            self.mask_dir = osp.join(subdir, "masks")

            self.images = [
                osp.join(self.img_dir, f) for f in os.listdir(self.img_dir) if f.endswith(".jpg") or f.endswith(".png")
            ]
            self.masks = [
                osp.join(self.mask_dir, f) for f in os.listdir(self.mask_dir) if f.endswith(".jpg") or f.endswith(".png")
            ]
            self.images.sort()
            self.masks.sort()

    def __getitem__(self, index) -> List[Any]:
        img = cv2.imread(self.images[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[index], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)
        # assert np.unique(mask).size == 2, "Mask must be binary"
        
        if self.data_transformer is not None:
            result = self.data_transformer(image=img, mask=mask)
            img = result["image"]
            mask = result["mask"]
        
        ret = [img, mask]

        if self.return_id:
            sample_name = osp.basename(self.images[index])
            sample_name = osp.splitext(sample_name)[0]
            ret.append(sample_name)

        return ret

    def __len__(self) -> int:
        return len(self.images)

    def __repr__(self) -> str:
        return (
            "PolypDataset(data_root={}, split={})\tSamples : {}".format(
                self.data_root, self.split, self.__len__())
        )


@DATA_TRANSFORM.register("polyp")
def data_transformation(*args, is_train: bool = True):
    if is_train:
        transformer = A.Compose([
            A.LongestMaxSize(max_size=640),
            A.PadIfNeeded(
                min_height=512, min_width=512,
                border_mode=cv2.BORDER_CONSTANT,
                value=(0, 0, 0), mask_value=255
            ),
            A.RandomCrop(height=512, width=512),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Normalize(),
            ToTensorV2()
        ])
    else:
        transformer = A.Compose([
            A.LongestMaxSize(max_size=512),
            A.PadIfNeeded(
                min_height=512, min_width=512,
                border_mode=cv2.BORDER_CONSTANT,
                value=(0, 0, 0), mask_value=0
            ),
            # A.Normalize(),
            # ToTensorV2()
        ])

    return transformer