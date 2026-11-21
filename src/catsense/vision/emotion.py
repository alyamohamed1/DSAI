"""Cat emotion classification from images.

Two approaches are compared against a majority-class baseline:
  1. Frozen ImageNet ResNet-50 embeddings + Logistic Regression (transfer learning, cheap)
  2. Fine-tuned ResNet-18 (end-to-end, class-weighted loss)
"""
from collections import Counter
from functools import lru_cache

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

from .. import config as C

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

EVAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
TRAIN_TF = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@lru_cache(maxsize=1)
def embedder() -> nn.Module:
    net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    net.fc = nn.Identity()
    return net.eval().to(DEVICE)


@torch.no_grad()
def embed(images: list[Image.Image], batch_size: int = 64) -> np.ndarray:
    net, out = embedder(), []
    for i in range(0, len(images), batch_size):
        batch = torch.stack([EVAL_TF(im.convert("RGB")) for im in images[i:i + batch_size]]).to(DEVICE)
        out.append(net(batch).cpu().numpy())
    return np.concatenate(out)


def finetune_net(n_classes: int = len(C.EMOTIONS)) -> nn.Module:
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Linear(net.fc.in_features, n_classes)
    return net


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, images: list[Image.Image], labels: np.ndarray, tf):
        self.images, self.labels, self.tf = images, labels, tf

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.tf(self.images[i].convert("RGB")), int(self.labels[i])


def train_finetune(images, labels, epochs: int = 6, lr: float = 3e-4, batch_size: int = 64, log=print) -> nn.Module:
    net = finetune_net().to(DEVICE)
    counts = np.bincount(labels, minlength=len(C.EMOTIONS))
    weights = torch.tensor(len(labels) / (len(counts) * np.maximum(counts, 1)), dtype=torch.float32, device=DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = torch.utils.data.DataLoader(ImageDataset(images, labels, TRAIN_TF), batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        net.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(net(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        sched.step()
        log(f"epoch {epoch + 1}/{epochs} loss={total / len(labels):.4f}")
    return net.eval()


@torch.no_grad()
def predict_proba_net(net: nn.Module, images: list[Image.Image], batch_size: int = 64) -> np.ndarray:
    net.eval().to(DEVICE)
    out = []
    for i in range(0, len(images), batch_size):
        batch = torch.stack([EVAL_TF(im.convert("RGB")) for im in images[i:i + batch_size]]).to(DEVICE)
        out.append(torch.softmax(net(batch), dim=1).cpu().numpy())
    return np.concatenate(out)


def majority_vote(labels: list[str]) -> tuple[str | None, float]:
    """Most frequent per-frame label and its share (DeepCat's `*_behavior` counters, generalised)."""
    if not labels:
        return None, 0.0
    label, count = Counter(labels).most_common(1)[0]
    return label, count / len(labels)
