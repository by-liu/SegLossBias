"""
File: visualizer.py
Author: Bingyuan Liu
Date: Dec 20, 2020
Brief: visualization functions : some are copied from detectron2
"""
import logging
import matplotlib.colors as mplc
import matplotlib.figure as mplfigure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
import numpy as np
import torch
from typing import Union, List, Optional
from enum import Enum, unique
import cv2

from .colormap import random_color

logger = logging.getLogger(__name__)


@unique
class ColorMode(Enum):
    """
    Enum of different color modes to use for instance visualizations.
    """

    IMAGE = 0
    """
    Picks a random color for every instance and overlay segmentations with low opacity.
    """
    SEGMENTATION = 1
    """
    Let instances of the same category have similar colors
    (from metadata.thing_colors), and overlay them with
    high opacity. This provides more attention on the quality of segmentation.
    """
    IMAGE_BW = 2
    """
    Same as IMAGE, but convert all areas without masks to gray-scale.
    Only available for drawing per-instance mask predictions.
    """


class VisImage:
    def __init__(self, img, scale=1.0):
        """
        Args:
            img (ndarray): an RGB image of shape (H, W, 3).
            scale (float): scale the input image
        """
        self.img = img
        self.scale = scale
        self.width, self.height = img.shape[1], img.shape[0]
        self._setup_figure(img)

    def _setup_figure(self, img):
        """
        Args:
            Same as in :meth:`__init__()`.

        Returns:
            fig (matplotlib.pyplot.figure): top level container for all the image plot elements.
            ax (matplotlib.pyplot.Axes): contains figure elements and sets the coordinate system.
        """
        fig = mplfigure.Figure(frameon=False)
        self.dpi = fig.get_dpi()
        # add a small 1e-2 to avoid precision lost due to matplotlib's truncation
        # (https://github.com/matplotlib/matplotlib/issues/15363)
        fig.set_size_inches(
            (self.width * self.scale + 1e-2) / self.dpi,
            (self.height * self.scale + 1e-2) / self.dpi,
        )
        self.canvas = FigureCanvasAgg(fig)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        # Need to imshow this first so that other patches can be drawn on top
        ax.imshow(img, extent=(0, self.width, self.height, 0), interpolation="nearest")

        self.fig = fig
        self.ax = ax

    def save(self, filepath):
        """
        Args:
            filepath (str): a string that contains the absolute path, including the file name, where
                the visualized image will be saved.
        """
        self.fig.savefig(filepath)

    def get_image(self) -> np.ndarray:
        """
        Returns:
            ndarray:
                the visualized image of shape (H, W, 3) (RGB) in uint8 type.
                The shape is scaled w.r.t the input image using the given `scale` argument.
        """
        canvas = self.canvas
        s, (width, height) = canvas.print_to_buffer()
        # buf = io.BytesIO()  # works for cairo backend
        # canvas.print_rgba(buf)
        # width, height = self.width, self.height
        # s = buf.getvalue()

        buffer = np.frombuffer(s, dtype="uint8")

        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        return rgb.astype("uint8")

    def show_image(self, window_name : str = "unnamed"):
        # img = Image.fromarray(self.get_image())
        # img.show(title=window_name)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, self.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class Visualizer:

    def __init__(
        self,
        img_rgb : np.ndarray,
        scale : float = 1.0,
        instance_mode : ColorMode = ColorMode.IMAGE
    ) -> None:
        """
        Args:
            img_rgb: a numpy array of shape (H, W, C), where H and W correspond to
                the height and width of the image respectively. C is the number of
                color channels. The image is required to be in RGB format since that
                is a requirement of the Matplotlib library. The image is also expected
                to be in the range [0, 255].
            instance_mode (ColorMode): defines one of the pre-defined style for drawing
                instances on an image.
        """
        self.img = np.asarray(img_rgb).clip(0, 255).astype(np.uint8)
        self.output = VisImage(self.img, scale=scale)
        self.cpu_device = torch.device("cpu")

        # too small texts are useless, therefore clamp to 9
        self._default_font_size = max(
            np.sqrt(self.output.height * self.output.width) // 90, 10 // scale
        )
        self._instance_mode = instance_mode

    def draw_binary_masks(
        self,
        masks : Union[np.ndarray, List[np.ndarray]],
        labels : List[int],
        assigned_colors : Optional[List[np.ndarray]] = None,
        alpha : float = 0.5
    ) -> VisImage:
        # if masks.shape[0] == len(labels):
        #     masks = np.einsum('kij->ijk', masks)
        if type(masks) == list:
            shape2d = masks[0].shape
        else:
            shape2d = (masks.shape[1], masks.shape[2])

        if assigned_colors is None:
            assigned_colors = [random_color(rgb=True, maximum=1) for _ in range(len(labels))]

        for i, l in enumerate(labels):
            color = mplc.to_rgb(assigned_colors[l])
            rgba = np.zeros(shape2d + (4,), dtype="float32")
            rgba[:, :, :3] = color
            rgba[:, :, 3] = (masks[i] == 1).astype("float32") * alpha
            self.output.ax.imshow(rgba, extent=(0, self.output.width, self.output.height, 0))

        return self.output


def show_image_from_arrary(arr : Union[np.ndarray, torch.Tensor]) -> None:
    if type(arr) == torch.Tensor:
        arr = arr.numpy()
    if len(arr.shape) == 3:
        arr = np.einsum('kij->ijk', (arr - arr.min()) / (arr.max() - arr.min()))

    plt.imshow(np.uint8(arr))
    plt.show()

    # img = Image.fromarray(np.uint8(arr * 255))
    # img.show()


def show_segmentation_label(arr : Union[np.ndarray, torch.Tensor]) -> None:
    if type(arr) == torch.Tensor:
        arr = arr.numpy()

    img = Image.fromarray(np.uint8(arr * 255))
    img.show()


def show_images(images, cols=1, sub_titles=None, title=None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((sub_titles is None)or (len(images) == len(sub_titles)))
    n_images = len(images)
    if sub_titles is None:
        sub_titles = [None for i in range(1, n_images + 1)]
    dpi = 100
    fig = plt.figure(figsize=(500 / dpi, 800 / dpi), dpi=dpi)
    if title is not None:
        fig.suptitle(title, y=0.61)
    for n, (image, sub_title) in enumerate(zip(images, sub_titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        a.axis("off")
        # a.get_xaxis().set_visible(False)
        # a.get_yaxis().set_visible(False)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        # size = fig.get_size_inches() * fig.dpi
        if sub_title is not None:
            a.set_title(sub_title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    fig.tight_layout()

    return fig


def save_image_gt_pred(img, mask, pred=None, save_path=None, title=None, show=False):
    img = img.numpy()
    img = np.einsum("cij->ijc", (img - img.min()) / (img.max() - img.min()))
    img = np.uint8(img * 255)
    mask = mask.numpy()
    mask = np.uint8(mask * 255)

    show_lists = [img, mask]

    if pred is not None:
        pred = pred.numpy()
        pred = np.uint8(pred * 255)
        show_lists.append(pred)
    fig = show_images(show_lists, title=title)

    if show:
        plt.show()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")


def colorize_mask(mask, palette):
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    labels = np.unique(mask)
    for l in labels:
        if l == 255:
            continue
        ind = np.nonzero(mask == l)
        colored_mask[ind[0], ind[1], :] = palette[l]
    return colored_mask


def image_mask_show(image, mask, original_image=None, original_mask=None, palette=None):
    fontsize = 18
    image = np.uint8(image)
    mask = np.uint8(mask)
    if palette is not None:
        mask = colorize_mask(mask, palette)

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        if palette is not None:
            original_mask = colorize_mask(original_mask, palette)

        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)

        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)

        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)

        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)
    
    plt.show()
