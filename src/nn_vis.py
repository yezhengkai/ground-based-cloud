from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from fastai.callback.hook import Hook

from .utils.vis_utils import get_rcParams


# ===== Grad-CAM =====
# References:
#   https://github.com/fastai/fastbook/blob/master/18_CAM.ipynb
#   https://github.com/henripal/maps/blob/master/nbs/big_resnet50-interpret-gradcam-dogs.ipynb
#   https://github.com/ai-fast-track/timeseries/blob/master/timeseries/cam.py#L233
#   https://github.com/stefannc/GradCAM-Pytorch/blob/master/gradcam.py
#   https://dhruvs.space/posts/generating-class-discriminative-heat-maps/
def hook_func(module, input, output):
    return output


def get_gradcam(tensor_image, tensor_category, target_layer, model, use_relu=True):
    with Hook(
        target_layer, hook_func, is_forward=False, detach=True, cpu=False, gather=False
    ) as hook_g:
        with Hook(
            target_layer,
            hook_func,
            is_forward=True,
            detach=True,
            cpu=False,
            gather=False,
        ) as hook_a:
            preds = model.eval()(tensor_image.unsqueeze(0).cuda())
            acts = hook_a.stored[0]
        preds[0, int(tensor_category)].backward()
        grad = hook_g.stored[0][0]
    weight = grad.mean(dim=[1, 2], keepdim=True)
    cam_map = (weight * acts).sum(0)
    if use_relu:
        cam_map = F.relu(cam_map)
    return cam_map


def plot_gradcam(
    image, labels, cam_map, ax=None, mode=None, filepath="gradcam.png", params=None
):

    _params = {
        "font.size": 11,
        "image.aspect": "auto",
        "image.cmap": "magma",
        "image.interpolation": "bilinear",
    }
    if isinstance(params, dict):
        _params.update(params)
    get_rcParams(_params, figsize="s")

    img_shape = image.shape
    if ax is None:
        fig, ax = plt.subplots()
    else:
        if not isinstance(ax, mpl.axes.Axes):
            raise TypeError("ax must be an matplotlib.axes.Axes")
        else:
            fig = plt.gcf()
    image.show(ctx=ax)
    ax.imshow(
        cam_map.detach().cpu(), alpha=0.6, extent=(0, img_shape[2], img_shape[1], 0),
    )
    ax.set_title(f"{labels}")

    if mode == "save":
        if not filepath.parent.is_dir():
            filepath.parent.mkdir(parents=True)
        fig.savefig(filepath)
        plt.close(fig)
        mpl.rcdefaults()
    elif mode == "show":
        plt.draw()
        plt.show()
        mpl.rcdefaults()
    else:
        plt.draw()
        mpl.rcdefaults()
        return fig
