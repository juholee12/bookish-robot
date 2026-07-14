import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from scipy.linalg import sqrtm
from tqdm import tqdm
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.transforms.functional import resize
from torchmetrics.image.fid import FrechetInceptionDistance
import data_helper

def _preprocess_fid_batch(batch: torch.Tensor) -> torch.Tensor:
    # Expect float in [0,1], 3x299x299
    if batch.dim() == 3:
        batch = batch.unsqueeze(0)                       # (1,C,H,W) or (1,H,W)
    if batch.dim() == 4 and batch.shape[1] not in {1, 3}:
        batch = batch.permute(0, 3, 1, 2)                # NHWC -> NCHW
    batch = batch.float()
    if batch.max() > 1.0:                                # if uint8 or [0,255]
        batch = batch / 255.0
    if batch.shape[1] == 1:
        batch = batch.repeat(1, 3, 1, 1)
    batch = torch.nn.functional.interpolate(
        batch, size=(299, 299), mode="bilinear", align_corners=False, antialias=True
    )
    return batch                                         # float32 in [0,1]


@torch.no_grad()
def calculate_fid_score(real_ds, synth_ds, batch_size: int = 256, device: torch.device = None) -> float:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    for loader, is_real in [
        (DataLoader(real_ds,  batch_size=batch_size, shuffle=False), True),
        (DataLoader(synth_ds, batch_size=batch_size, shuffle=False), False),
    ]:
        for batch in loader:
            imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
            imgs = _preprocess_fid_batch(imgs).to(device, non_blocking=True)
            fid.update(imgs, real=is_real)

    score = float(fid.compute())
    fid.reset()
    return score


@torch.no_grad()
def build_cached_real_fid(real_ds, batch_size: int = 256, device: torch.device = None) -> FrechetInceptionDistance:
    """
    Build a FrechetInceptionDistance metric preloaded with real_ds's features.
    reset_real_features=False means calling .reset() after each .compute() only
    clears the synthetic-side accumulator, so real_ds never needs to be re-run
    through Inception again. Reuse the returned object across many
    calculate_fid_score_cached calls against the same real_ds.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fid = FrechetInceptionDistance(feature=2048, normalize=True, reset_real_features=False).to(device)

    for batch in DataLoader(real_ds, batch_size=batch_size, shuffle=False):
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        imgs = _preprocess_fid_batch(imgs).to(device, non_blocking=True)
        fid.update(imgs, real=True)

    return fid


@torch.no_grad()
def calculate_fid_score_cached(fid_metric: FrechetInceptionDistance, synth_ds, batch_size: int = 256, device: torch.device = None) -> float:
    """
    Compute FID against synth_ds using a metric already preloaded with real
    features via build_cached_real_fid, skipping the redundant real-image pass.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for batch in DataLoader(synth_ds, batch_size=batch_size, shuffle=False):
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        imgs = _preprocess_fid_batch(imgs).to(device, non_blocking=True)
        fid_metric.update(imgs, real=False)

    score = float(fid_metric.compute())
    fid_metric.reset()  # only resets the fake-side accumulator; real stats stay cached
    return score


def calculate_fid_from_model(real_ds, model, batch_size: int = 128, device: torch.device = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    synthetic_data_size = len(real_ds)
    gen_imgs_before_filter, y_before_filter = data_helper.generate_balanced_synthetic_data(
        synthetic_model=model,
        target_size=synthetic_data_size,
        binary_format=False,
        device=device
    )
    synthetic_ds = torch.utils.data.TensorDataset(
        gen_imgs_before_filter, y_before_filter)
    fid = calculate_fid_score(
        real_ds, synthetic_ds, batch_size=batch_size, device=device)

    return fid
