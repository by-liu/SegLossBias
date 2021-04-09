"""
File: dataset.py
Author: Binguan Liu
Date: Dec 15, 2020
Brief: Inerface of dataset
"""

import os
import os.path as osp
import zipfile
from typing import List, Tuple


def mkdir(output_dir : str) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def zipdir(path, result_path):
    zipf = zipfile.ZipFile(result_path, "w")
    for root, dirs, files in os.walk(path):
        for file in files:
            zipf.write(
                osp.join(root, file),
                osp.relpath(osp.join(root, file), osp.join(path, '..'))
            )
    zipf.close()


def load_list(path : str) -> List[str]:
    """load list from text file"""
    assert osp.exists(path), "{} does not exist".format(path)

    ret = []
    with open(path, "r") as f:
        for line in f:
            ret.append(line.strip())
    return ret


def load_leison_classes(path : str) -> Tuple[List[str], List[str]]:
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
