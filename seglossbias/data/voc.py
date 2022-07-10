import os.path as osp
import numpy as np
import cv2
from PIL import Image
from typing import Callable, Optional
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

CLASSES = ('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
           'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor')

PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
           [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
           [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
           [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
           [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]


class VOCSegmentation(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        data_transform: Optional[Callable] = None,
        return_id: bool = False
    ):
        assert split in {"train", "val", "trainval", "test"}
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.data_transform = data_transform
        self.return_id = return_id

        self.classes = CLASSES
        self.num_classes = 21

        self.load_list()

    def load_list(self):
        self.img_dir = osp.join(self.data_root, "JPEGImages")
        self.mask_dir = osp.join(self.data_root, "SegmentationClass")
        self.split_dir = osp.join(self.data_root, "ImageSets/Segmentation")

        split_file = osp.join(self.split_dir, "{}.txt".format(self.split))

        with open(split_file, "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [osp.join(self.img_dir, x + ".jpg") for x in file_names]
        self.masks = [osp.join(self.mask_dir, x + ".png") for x in file_names]
        assert len(self.images) == len(self.masks)

    def __getitem__(self, index):
        img = cv2.imread(self.images[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = np.array(Image.open(self.masks[index]))

        if self.data_transform is not None:
            result = self.data_transform(
                image=img, mask=mask
            )
            img = result["image"]
            mask = result["mask"].long()

        if self.return_id:
            return (
                img, mask, self.images[index].split("/")[-1].split(".")[0]
            )
        else:
            return img, mask

    def __len__(self) -> int:
        return len(self.images)

    def __repr__(self) -> str:
        return (
            "VOCSegmentation (data_root={},split={})\tSamples : {}".format(
                self.data_root, self.split, self.__len__()
            )
        )
