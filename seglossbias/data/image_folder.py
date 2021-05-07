"""
File: image_folder.py
Author: Bingyuan Liu
Date: May 4, 2021
Brief: dataset wrapper for image folder
"""

import os
import os.path as osp
import logging
import cv2
from PIL import Image
from torch.utils.data.dataset import Dataset
from typing import Optional, Tuple, Callable, Any

logger = logging.getLogger(__name__)


class ImageFolder(Dataset):
    """A test data dataset wrapper where the images are arranged in this way: 
        data_root/xxx.ext
        data_root/xxy.ext
        ......
    """

    def __init__(
        self,
        data_root: str,
        exts: Tuple[str] = ("jpg", "jpeg", "png"),
        return_id: bool = True,
        transforms: Optional[Callable] = None,
    ) -> None:
        self.data_root = data_root
        self.exts = exts
        self.return_id = return_id
        self.transforms = transforms
        # get all the test samples
        self.samples = self.get_images_files()

    def get_images_files(self):
        samples = sorted(os.listdir(self.data_root))
        samples = list(filter(lambda x: x.lower().endswith(self.exts), samples))
        return samples

    def __getitem__(self, index: int) -> Tuple[Any, ...]:
        sample_name = self.samples[index]
        img_file = osp.join(self.data_root, sample_name)
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            img = self.transforms(Image.fromarray(img))

        if self.return_id:
            return img, sample_name
        else:
            return img

    def __len__(self) -> int:
        return len(self.samples)
