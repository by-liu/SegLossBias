import unittest

from seglossbias.data.polyp_dataset import PolypDataset, data_transformation


class TestPolypDataset(unittest.TestCase):
    def test_testset(self):
        data_root = "data/polyp"

        for set_name in ["CVC-300", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "Kvasir"]:
            print("===== {} =====".format(set_name))
            dataset = PolypDataset(data_root, split="test", set_name=set_name, data_transformer=data_transformation(is_train=False))
            for i in range(len(dataset)):
                img, target = dataset[i]
                self.assertEqual(img.shape, (3, 512, 512))
                self.assertEqual(target.shape, (512, 512))

    # def test_trainset(self):
    #     data_root = "data/polyp"
    #     dataset = PolypDataset(data_root, split="train", data_transformer=data_transformation(is_train=True))
    #     for i in range(len(dataset)):
    #         img, target = dataset[i]
    #         self.assertEqual(img.shape, (3, 512, 512))
    #         self.assertEqual(target.shape, (512, 512))


if __name__ == "__main__":
    unittest.main()
