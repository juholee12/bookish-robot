import torch
import torch.nn as nn
import torch.nn.functional as F
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


@torch.no_grad()
def extract_inception_features(fid_metric: FrechetInceptionDistance, dataset, batch_size: int = 256, device: torch.device = None) -> torch.Tensor:
    """
    Extract raw (N, 2048) Inception features for a dataset, reusing the same
    Inception network already loaded inside fid_metric (built by
    build_cached_real_fid) so no separate feature-extraction model is needed.
    Used for PRDC (precision/recall/density/coverage) metrics.

    Returned as a GPU tensor (kept on `device`, not moved to CPU/NumPy) so the
    pairwise-distance computation in build_cached_real_dc_radii/
    compute_density_coverage can run via torch.cdist on the GPU instead of
    scipy.cdist on the CPU.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features = []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        imgs = _preprocess_fid_batch(imgs).to(device, non_blocking=True)
        # fid_metric.inception expects torch.uint8 in [0, 255]; FrechetInceptionDistance.update()
        # normally does this (imgs * 255).byte() conversion internally before calling .inception,
        # but calling .inception directly (to get raw features instead of just the FID scalar)
        # bypasses that, so it has to be done here explicitly.
        imgs = (imgs * 255).byte()
        features.append(fid_metric.inception(imgs))

    return torch.cat(features, dim=0)


@torch.no_grad()
def extract_embedding_features(embedding_model, dataset, batch_size: int = 256, device: torch.device = None) -> torch.Tensor:
    """
    Extract raw (N, embedding_dim) features using any model exposing an
    .encode(x) method (e.g. SimpleAutoencoder) - a drop-in replacement for
    extract_inception_features when using a domain-specific embedding instead
    of InceptionV3. No 299x299 resize or fake-RGB tripling needed here, since
    the embedding model is trained directly on 28x28 grayscale digits.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_model.eval()

    features = []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        imgs = imgs.float().to(device, non_blocking=True)
        if imgs.max() > 1.0:
            imgs = imgs / 255.0
        features.append(embedding_model.encode(imgs))

    return torch.cat(features, dim=0)


@torch.no_grad()
def compute_real_fid_stats(real_features: torch.Tensor):
    """Mean and covariance of real_features, computed once and reused - the
    other half (besides real_dc_radii) of what's needed to avoid a second
    embedding pass over the same batch just to get a separate FID number."""
    mu = real_features.mean(dim=0)
    centered = real_features - mu
    sigma = (centered.T @ centered) / (real_features.shape[0] - 1)
    return mu, sigma


@torch.no_grad()
def calculate_fid_from_features(real_mu: torch.Tensor, real_sigma: torch.Tensor, fake_features: torch.Tensor) -> float:
    """
    Standard Frechet distance formula, computed directly from features already
    extracted for PRDC - avoids running the same batch through the embedding
    model a second time just to get FID via a separate incremental metric.
    """
    fake_mu = fake_features.mean(dim=0)
    centered = fake_features - fake_mu
    fake_sigma = (centered.T @ centered) / (fake_features.shape[0] - 1)

    diff = real_mu - fake_mu
    # sqrtm has no direct GPU equivalent, but this matrix is only
    # (embedding_dim x embedding_dim) - tiny regardless of how many samples
    # N was - so moving just this one step to CPU is negligible cost.
    covmean = sqrtm((real_sigma @ fake_sigma).cpu().numpy(), disp=False)[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    covmean = torch.from_numpy(covmean).to(real_sigma.device, dtype=real_sigma.dtype)

    fid = diff @ diff + torch.trace(real_sigma + fake_sigma - 2 * covmean)
    return fid.item()


@torch.no_grad()
def build_cached_real_dc_radii(real_features: torch.Tensor, nearest_k: int) -> torch.Tensor:
    """
    One-time computation of each real sample's k-th nearest-neighbor distance
    among the *other* real samples (Naeem et al. 2020 / the `prdc` package's
    density & coverage formulas). This only depends on real_features, which
    never changes across iterations, so compute it once and reuse it -
    avoids recomputing an (N_real x N_real) pairwise distance matrix every call.

    Uses torch.cdist (GPU, if real_features is a GPU tensor) instead of
    scipy.spatial.distance.cdist (CPU-only) - this pairwise-distance step was
    the dominant cost in the per-iteration metrics timing breakdown, not the
    Inception forward pass, so moving it to the GPU is the actual lever here.
    """
    distances = torch.cdist(real_features, real_features)
    # kthvalue is 1-indexed and distance-to-self (0) always occupies rank 1,
    # so kthvalue(k=nearest_k+1) correctly yields the k-th nearest *other*
    # point's distance (matches the old np.partition(distances, k)[:, k] logic).
    return torch.kthvalue(distances, nearest_k + 1, dim=-1).values


@torch.no_grad()
def compute_density_coverage(real_dc_radii: torch.Tensor, real_features: torch.Tensor, fake_features: torch.Tensor, nearest_k: int) -> dict:
    """
    Density and Coverage only - deliberately skips precision/recall. Recall is
    the only one of the four PRDC metrics that needs a fake-vs-fake pairwise
    distance matrix; density/coverage only need real-vs-real (passed in
    pre-cached via build_cached_real_dc_radii) and real-vs-fake distances, so
    skipping precision/recall avoids computing that extra matrix entirely.
    """
    distance_real_fake = torch.cdist(real_features, fake_features)  # (n_real, n_fake)

    density = (1.0 / nearest_k) * (
        distance_real_fake < real_dc_radii.unsqueeze(1)
    ).sum(dim=0).float().mean()

    coverage = (
        distance_real_fake.min(dim=1).values < real_dc_radii
    ).float().mean()

    return {"density": density.item(), "coverage": coverage.item()}


@torch.no_grad()
def extract_labels(dataset, batch_size: int = 256) -> torch.Tensor:
    """
    Pull every label out of a dataset that yields (image, label) pairs,
    concatenated into one tensor in dataset order - a label-only counterpart
    to extract_embedding_features/extract_inception_features, used to build
    the per-class masks needed by the stratified density/coverage functions
    below (e.g. DirectoryBasedSyntheticDataset exposes labels only through
    __getitem__, one file at a time, with no single all-labels attribute).
    """
    labels = []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        lbl = batch[1] if isinstance(batch, (list, tuple)) else batch
        labels.append(torch.as_tensor(lbl))
    return torch.cat(labels, dim=0)


@torch.no_grad()
def build_cached_real_dc_radii_per_class(real_features: torch.Tensor, real_labels: torch.Tensor, nearest_k: int, num_classes: int = 10) -> dict:
    """
    Per-class counterpart to build_cached_real_dc_radii: computes each real
    sample's k-th nearest-neighbor distance among *other real samples of the
    same class only*, instead of pooling all classes into one manifold.

    Motivation: pooled density/coverage can't tell "the fake population spans
    more of the real manifold because it's more diverse" apart from "the fake
    population spans more of the real manifold because digits became sharper
    and better separated *between* classes" - both inflate the pooled metric
    the same way. Stratifying by class removes that confound, since between-
    class separation can no longer contribute; only within-class spread can.
    """
    radii_by_class = {}
    for c in range(num_classes):
        mask = real_labels == c
        radii_by_class[c] = build_cached_real_dc_radii(real_features[mask], nearest_k)
    return radii_by_class


@torch.no_grad()
def compute_density_coverage_stratified(
    real_dc_radii_per_class: dict,
    real_features: torch.Tensor,
    real_labels: torch.Tensor,
    fake_features: torch.Tensor,
    fake_labels: torch.Tensor,
    nearest_k: int,
    num_classes: int = 10,
) -> dict:
    """
    Per-class density/coverage, each class's fake samples only ever compared
    against real samples of that same class. Returns per-class scores plus a
    macro-average (equal weight per class, matching generate_balanced_synthetic_data's
    equal-per-digit sampling) - the macro-average is the number to compare
    directly against the pooled density/coverage from compute_density_coverage.
    """
    per_class = {}
    for c in range(num_classes):
        real_mask = real_labels == c
        fake_mask = fake_labels == c
        if fake_mask.sum() == 0 or real_mask.sum() == 0:
            per_class[c] = {"density": float("nan"), "coverage": float("nan")}
            continue
        per_class[c] = compute_density_coverage(
            real_dc_radii_per_class[c],
            real_features[real_mask],
            fake_features[fake_mask],
            nearest_k,
        )

    densities = [v["density"] for v in per_class.values() if not np.isnan(v["density"])]
    coverages = [v["coverage"] for v in per_class.values() if not np.isnan(v["coverage"])]
    return {
        "per_class": per_class,
        "density_stratified": float(np.mean(densities)) if densities else float("nan"),
        "coverage_stratified": float(np.mean(coverages)) if coverages else float("nan"),
    }


@torch.no_grad()
def compute_kid(
    real_features: torch.Tensor,
    fake_features: torch.Tensor,
    subset_size: int = 1000,
    num_subsets: int = 10,
    degree: int = 3,
    coef0: float = 1.0,
) -> float:
    """
    Kernel Inception Distance (Binkowski et al., 2018): unbiased polynomial-
    kernel MMD^2 between real and fake embeddings. Unlike FID, this makes no
    Gaussianity assumption on the embedding distribution - relevant here
    since nothing guarantees the 64-d autoencoder embedding is close to
    Gaussian - and its estimator is unbiased at any sample size, whereas
    FID's plug-in mean/covariance estimate is biased at small N.

    Computed via the standard subset-averaging estimator (repeatedly drawing
    a `subset_size` sample from each population and averaging the unbiased
    MMD^2 estimate across `num_subsets` draws) rather than a single pass over
    all N samples, since a full N x N kernel matrix at N=10,000 would be
    expensive to rebuild every iteration for no accuracy benefit.
    """
    m_total = real_features.shape[0]
    n_total = fake_features.shape[0]
    subset_size = min(subset_size, m_total, n_total)
    gamma = 1.0 / real_features.shape[1]

    scores = []
    for _ in range(num_subsets):
        real_idx = torch.randperm(m_total, device=real_features.device)[:subset_size]
        fake_idx = torch.randperm(n_total, device=fake_features.device)[:subset_size]
        x = real_features[real_idx]
        y = fake_features[fake_idx]

        k_xx = (gamma * (x @ x.T) + coef0) ** degree
        k_yy = (gamma * (y @ y.T) + coef0) ** degree
        k_xy = (gamma * (x @ y.T) + coef0) ** degree

        m = subset_size
        sum_xx = (k_xx.sum() - k_xx.diagonal().sum()) / (m * (m - 1))
        sum_yy = (k_yy.sum() - k_yy.diagonal().sum()) / (m * (m - 1))
        sum_xy = k_xy.mean()

        scores.append((sum_xx + sum_yy - 2 * sum_xy).item())

    return float(np.mean(scores))


@torch.no_grad()
def compute_vendi_score(features: torch.Tensor) -> float:
    """
    Vendi Score (Friedman & Dieng, 2023): exp(Shannon entropy of the
    eigenvalues of the normalized cosine-similarity matrix over a set of
    embeddings). Reference-free - it never looks at real data - so it
    measures purely how many effectively distinct samples the fake set
    contains. This is the axis density/coverage cannot see: a generator that
    collapses onto a single mode per class but sits exactly on the densest
    part of the real manifold can still score well on coverage, since
    coverage only asks "is some fake sample near this real sample," not
    "are the fake samples different from each other."
    """
    n = features.shape[0]
    normed = F.normalize(features, dim=1)
    K = (normed @ normed.T) / n
    eigenvalues = torch.linalg.eigvalsh(K).clamp(min=0)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    p = eigenvalues / eigenvalues.sum()
    entropy = -(p * p.log()).sum()
    return torch.exp(entropy).item()


@torch.no_grad()
def compute_vendi_score_per_class(features: torch.Tensor, labels: torch.Tensor, num_classes: int = 10) -> dict:
    """
    Vendi score computed separately within each digit class, then averaged -
    directly targets within-class style collapse (e.g. always drawing the
    same "7"), which class-pooled diagnostics (a classifier's predicted-label
    entropy, or pooled density/coverage) cannot detect since a single
    dominant style per class still classifies correctly and still overlaps
    the real manifold.
    """
    per_class = {}
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() < 2:
            per_class[c] = float("nan")
            continue
        per_class[c] = compute_vendi_score(features[mask])
    valid = [v for v in per_class.values() if not np.isnan(v)]
    return {
        "per_class": per_class,
        "vendi_mean": float(np.mean(valid)) if valid else float("nan"),
    }


@torch.no_grad()
def compute_class_mode_coverage(
    classifier,
    dataset,
    device: torch.device,
    num_classes: int = 10,
    batch_size: int = 512,
) -> dict:
    """
    Oracle mode-coverage check via an independently-trained digit classifier
    (never the adversarial discriminator, never the autoencoder embedding -
    conflating either would entangle this check with the models actually
    being evaluated/trained). Reports:
      - class_entropy: Shannon entropy of the predicted-label histogram,
        normalized by log(num_classes) so 1.0 = perfectly uniform over all
        10 digits and 0.0 = collapsed onto a single digit. Catches *between*-
        class mode dropping directly - near-ground-truth given how accurate
        MNIST classifiers are - but is blind to *within*-class style
        collapse (pair with compute_vendi_score_per_class for that).
      - label_match_rate: fraction of samples whose predicted digit matches
        the label the generator was conditioned on - a fidelity check
        independent of FID/KID.
    """
    classifier.eval()
    all_preds, all_intended = [], []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        imgs, intended = batch
        imgs = imgs.float().to(device)
        if imgs.max() > 1.0:
            imgs = imgs / 255.0
        logits = classifier(imgs)
        all_preds.append(logits.argmax(dim=1).cpu())
        all_intended.append(torch.as_tensor(intended).cpu())
    preds = torch.cat(all_preds, dim=0)
    intended = torch.cat(all_intended, dim=0)

    counts = torch.bincount(preds, minlength=num_classes).float()
    probs = counts / counts.sum()
    nonzero = probs[probs > 0]
    entropy = -(nonzero * nonzero.log()).sum().item()
    normalized_entropy = entropy / np.log(num_classes)
    match_rate = (preds == intended).float().mean().item()

    return {"class_entropy": normalized_entropy, "label_match_rate": match_rate}


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
