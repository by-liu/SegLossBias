import os.path as osp
import cv2
from PIL import Image
import albumentations as A
import numpy as np
import torch
from typing import List, Tuple, Optional
from torch.utils.data.dataset import Dataset

from seglossbias.utils import load_list
# from utils.colormap import colormap

_EPS = 1e-10
_NUM_CLASSES = 8
# _LESION_COLORS = colormap(rgb=True, maximum=1)[:_NUM_CLASSES, :].tolist()


class RetinalLesionsDataset(Dataset):
    """
    Wrapper for retinal lesion dataset.
    """
    def __init__(self, data_root : str,
                 sample_list_path : str,
                 classes_path : str,
                 transforms : Optional[A.Compose] = None,
                 label_values : List[int] = [127, 255],
                 binary : bool = False,
                 region_number : bool = False,
                 return_id : bool = False) -> None:
        self.data_root = data_root
        self.image_dir = osp.join(self.data_root, "images_896x896")
        self.seg_dir = osp.join(self.data_root, "lesion_segs_896x896")
        self.samples : List[str] = load_list(sample_list_path)
        self.classes, self.classes_abbrev = self.load_classes(classes_path)
        self.transforms = transforms
        self.label_values = label_values
        self.binary = binary
        self.region_number = region_number
        self.return_id = return_id

    def load_classes(self, path : str):
        """load retinal lesion classes info from text file"""
        assert osp.exists(path), "{} does not exist".format(path)
        classes = []
        classes_abbrev = []
        with open(path, "r") as f:
            for line in f:
                class_name, class_abbrev_name = line.strip().split(",")
                classes.append(class_name)
                classes_abbrev.append(class_abbrev_name)
        return classes, classes_abbrev

    def get_target(self, img_shape : np.array, label_dir : str, label_values : List[int]):
        target = np.zeros((img_shape[0], img_shape[1], len(self.classes)), dtype=np.uint8)
        for i, class_name in enumerate(self.classes):
            expected_path = osp.join(label_dir, "{}.png".format(class_name))
            if osp.exists(expected_path):
                img = cv2.imread(expected_path, cv2.IMREAD_GRAYSCALE)
                mask = np.zeros_like(img)
                for val in self.label_values:
                    mask[np.where(img == val)] = 1
                target[:, :, i] = mask
        if self.binary:
            target = np.einsum("ijk->ij", target)
            target = (target > 0).astype(np.uint8)
        return target

    def get_region_number(self, target : torch.Tensor) -> torch.Tensor:
        region_number = torch.zeros(target.size(0), 1)
        for c in range(target.size(0)):
            thres = (target[c].numpy() * 255).astype(np.uint8)
            countours, _ = cv2.findContours(thres, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            region_number[c] = len(countours)
        return region_number

    def __getitem__(self, index : int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_name = self.samples[index]
        img = cv2.imread(osp.join(self.image_dir, "{}.jpg".format(sample_name)))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        target = self.get_target(
            img.shape, osp.join(self.seg_dir, sample_name), self.label_values
        )

        if self.transforms is not None:
            result = self.transforms(image=img, mask=target)
            img = result["image"]
            target = result["mask"].long()
            # img, target = self.transforms(Image.fromarray(img), Image.fromarray(target))
            # target = torch.unsqueeze(target, 0).type(torch.float32)
        ret = [img, target]

        if self.region_number:
            ret.append(self.get_region_number(target))

        if self.return_id:
            ret.append(sample_name)

        return ret

    def __len__(self) -> int:
        return len(self.samples)
