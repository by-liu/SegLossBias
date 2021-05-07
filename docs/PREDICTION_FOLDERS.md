# Inference with an image folder

This is a guide on performing inference with a given image folder.

- Installation : please refer to [Prerequisites](../README.md#Prerequisites)

- Download a trained segmentation model on Retinal Lesions : [model](https://drive.google.com/file/d/1c9CsFf8AlPKsDGHcjvvSNTN-4Zh2MX2w/view?usp=sharing)

- Download the examples : [test_images](https://drive.google.com/file/d/19SYRydo1icw5PvpkrxGRjw6jJhZF9l6m/view?usp=sharing)

- Run the scripts (you may need to change the path parameters of image-path, save-path and TEST.CHECKPOINT_PATH in the command)
  
  ```
  python tools/test_net_with_folder.py --image-path ./data/test_images --save-path ./prediction \\
    --config-file ./configs/retinal-lesions/unet_bce-l1_896x896.yaml \\
    TEST.CHECKPOINT_PATH ./trained/retinal-lesions_r50unet_896x896_bce-l1.pth
  ```

