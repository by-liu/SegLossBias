import unittest
import os.path as osp

from seglossbias.config import get_cfg
from seglossbias.data import RetinalLesionsDataset
from seglossbias.data.data_transform import retinal_lesion


class TestRetinalLesionsDataset(unittest.TestCase):
    def test(self):
        cfg = get_cfg()
        cfg.DATA.NAME = "retinal-lesions"
        cfg.DATA.RESIZE = [512, 512]

        data_root = "./data/retinal-lesions-v20191227"
        sample_list_path = osp.join(data_root, "samples.txt")
        classes_path = osp.join(data_root, "classes.txt")
        transforms = retinal_lesion(cfg)
        dataset = RetinalLesionsDataset(
            data_root,
            sample_list_path,
            classes_path,
            transforms=transforms,
            binary=True
        )

        for i in range(3):
            img, target = dataset[i]

        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
