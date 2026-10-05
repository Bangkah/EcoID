"""Shared PyTorch bits for train/export. Requires torch + torchvision (see requirements-train.txt).

The eval transform here MUST stay numerically identical to app/ai/preprocessing/preprocess.py.
export_model.py verifies that on real images.
"""
from pathlib import Path

import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms as T

import _common  # noqa: F401
from app.ai.contract.labels import CLASSES, NUM_CLASSES
from app.ai.data.checks import CLASS_DIRS, list_images
from app.ai.preprocessing.preprocess import (
    IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE, load_image,
)

BILINEAR = T.InterpolationMode.BILINEAR
NORMALIZE = T.Normalize(IMAGENET_MEAN.tolist(), IMAGENET_STD.tolist())

# Serving-equivalent transform (direct resize, no crop) -- SRS FR-002.
EVAL_TF = T.Compose([T.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=BILINEAR), T.ToTensor(), NORMALIZE])

# Training augmentation (SRS 5.2): rotation, flip, brightness/contrast, random crop.
# Deliberately NO shear / perspective / elastic warps (they distort leaf shape).
TRAIN_TF = T.Compose([
    T.RandomRotation(20, interpolation=BILINEAR),
    T.RandomResizedCrop(INPUT_SIZE, scale=(0.6, 1.0), ratio=(0.85, 1.18), interpolation=BILINEAR),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
    T.ToTensor(),
    NORMALIZE,
])


class FolderDataset(Dataset):
    """root/<Class_name>/*.jpg  ->  (tensor, class_index). Uses the SAME decoder as serving."""

    def __init__(self, root: Path, transform):
        self.items = [(p, i) for i, d in enumerate(CLASS_DIRS) for p in list_images(Path(root) / d)]
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return self.transform(load_image(p)), y

    def counts(self):
        c = [0] * NUM_CLASSES
        for _, y in self.items:
            c[y] += 1
        return dict(zip(CLASSES, c))


ARCHS = {
    "small": ("mobilenet_v3_small", "MobileNet_V3_Small_Weights"),
    "large": ("mobilenet_v3_large", "MobileNet_V3_Large_Weights"),
}


def build_model(arch: str, pretrained: bool, weights_path: str | None = None) -> nn.Module:
    """pretrained=True is REQUIRED for training (SRS 5.2 forbids training from scratch).
    pretrained=False is only used by export to load our own fine-tuned checkpoint."""
    fn_name, w_name = ARCHS[arch]
    fn = getattr(torchvision.models, fn_name)
    if pretrained and weights_path:
        model = fn(weights=None)
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    elif pretrained:
        model = fn(weights=getattr(torchvision.models, w_name).IMAGENET1K_V1)
    else:
        model = fn(weights=None)
    in_f = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_f, NUM_CLASSES)   # new 5-class head -> raw logits
    return model
