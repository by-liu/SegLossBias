import unittest
import os.path as osp
import torch

from seglossbias.modeling.compound_losses import CrossEntropyWithL1, CrossEntropyWithKL


torch.manual_seed(101)
batch_size = 8
num_classes = 10
width, height = 512, 512
rand_max, rand_min = 2.5, -2.5


class TestCompoundLoss(unittest.TestCase):
    def test_ce_l1(self):
        mode = "binary"
        loss_func = CrossEntropyWithL1(mode)
        logits = (rand_max - rand_min) * torch.rand((batch_size, 1, height, width)) + rand_min
        labels = torch.randint(0, 2, (batch_size, height, width))
        loss, loss_ce, loss_reg = loss_func(logits, labels)

        mode = "multiclass"
        loss_func = CrossEntropyWithL1(mode)
        logits = (rand_max - rand_min) * torch.rand((batch_size, num_classes, height, width)) + rand_min
        labels = torch.randint(0, num_classes, (batch_size, height, width))
        loss, loss_ce, loss_reg = loss_func(logits, labels)

        self.assertTrue(True)

    def test_ce_kl(self):
        mode = "binary"
        loss_func = CrossEntropyWithKL(mode)
        logits = (rand_max - rand_min) * torch.rand((batch_size, 1, height, width)) + rand_min
        labels = torch.randint(0, 2, (batch_size, height, width))
        loss, loss_ce, loss_reg = loss_func(logits, labels)

        mode = "multiclass"
        loss_func = CrossEntropyWithKL(mode)
        logits = (rand_max - rand_min) * torch.rand((batch_size, num_classes, height, width)) + rand_min
        labels = torch.randint(0, num_classes, (batch_size, height, width))
        loss, loss_ce, loss_reg = loss_func(logits, labels)

        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
