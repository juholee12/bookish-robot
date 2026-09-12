"""
Iterative retraining WITHOUT a verifier (unfiltered lineage).

Companion to ELBO_experiment.py. The baseline script trains each generation on
the top-10%-by-discriminator-score subset of its predecessor's output; this one
removes the verifier from the loop entirely, so every generation trains directly
on its predecessor's raw, unfiltered synthetic batch. This is the CVAE analogue
of the "no verifier" condition in the Gaussian toy model.

Because there is no filtering step, there is no unfiltered/filtered distinction
to report: each iteration produces exactly one batch, which is both the batch
that gets measured and the batch the next generation trains on. Metric columns
are therefore unsuffixed (fid, coverage, vendi, ...) rather than carrying the
_unfiltered/_filtered suffixes of the baseline CSV. No discriminator is trained
at any point, so the disc_* columns are absent as well.
"""
import sys
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, Subset
import torch.nn.functional as F

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import random
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

KST = ZoneInfo("Asia/Seoul")

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent / "src"
sys.path.append(str(SRC_DIR))

import models as models
import train_helper as train_helper
import utils as utils
import data_helper as data_helper
import fid as fid_helper

# ---------------------------------------------------------------------------
# Device, seed, paths
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_seed = 0
torch.manual_seed(base_seed)
torch.cuda.manual_seed_all(base_seed)
np.random.seed(base_seed)
random.seed(base_seed)

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    drive_mountpoint = Path("/content/drive")
    if not (drive_mountpoint / "MyDrive").is_dir():
        # drive.mount() talks to the notebook's IPython kernel to do the
        # OAuth/mount handshake. That link doesn't exist when this script is
        # run as a subprocess (e.g. `!python unfiltered_experiment.py`), so it
        # will crash here in that case. Mount Drive from an actual notebook cell
        # first (`from google.colab import drive; drive.mount('/content/drive')`)
        # before launching this script that way.
        from google.colab import drive
        drive.mount(str(drive_mountpoint))
    DRIVE_BASE = drive_mountpoint / "MyDrive" / "verified_synthetic_data" / "MNIST"
else:
    DRIVE_BASE = THIS_DIR.parent

# Each run gets its own timestamped subfolder so reruns never overwrite or
# get appended on top of a previous run's data/results. Set the RUN_ID env
# var yourself before running if you want to deliberately continue writing
# into an existing run's folder instead of starting a new one.
RUN_ID = os.environ.get("RUN_ID") or datetime.now(KST).strftime("%Y%m%d_%H%M%S")
ROOT = DRIVE_BASE / "runs" / f"unfiltered_{RUN_ID}"
print(f"RUN_ID: {RUN_ID}  (outputs -> {ROOT})")

model_saved_path = os.path.join(ROOT, "model_saved")
results_saved_path = os.path.join(ROOT, "results_saved")
picture_saved_path = os.path.join(ROOT, "picture_saved")
plots_saved_path = os.path.join(results_saved_path, "plots")
os.makedirs(results_saved_path, exist_ok=True)
os.makedirs(model_saved_path, exist_ok=True)
os.makedirs(picture_saved_path, exist_ok=True)
os.makedirs(plots_saved_path, exist_ok=True)


def save_metric_plots(test_results, plots_dir):
    """One line-graph PNG per numeric metric, overwritten every iteration so a
    disconnect mid-run still leaves usable plots up to the last completed iteration."""
    iterations = range(1, len(test_results["model_name"]) + 1)
    for column, values in test_results.items():
        if column == "model_name":
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(iterations, values, marker="o")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(column)
        ax.set_title(f"{column} vs Iteration")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"{column}.png"), dpi=150)
        plt.close(fig)


def save_preview_grid(images, labels, save_path, per_class=5, num_classes=10):
    """Cheap PNG preview: a handful of samples per digit, not the full batch."""
    images = images.detach().cpu()
    labels = labels.detach().cpu()
    if images.dim() == 2 and images.shape[1] == 784:
        images = images.view(-1, 1, 28, 28)
    fig, axes = plt.subplots(num_classes, per_class, figsize=(1.5 * per_class, 1.5 * num_classes))
    for c in range(num_classes):
        idx = (labels == c).nonzero(as_tuple=True)[0][:per_class]
        for j in range(per_class):
            ax = axes[c, j]
            ax.axis("off")
            if j < len(idx):
                ax.imshow(images[idx[j]].squeeze().numpy(), cmap="gray")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
full_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
full_digit_indices = utils.create_balanced_subset_indices(full_dataset, seed=base_seed)

# ---------------------------------------------------------------------------
# Embedding function for FID / Density / Coverage
# ---------------------------------------------------------------------------
# InceptionV3 (ImageNet-pretrained) is a poor fit for small grayscale MNIST
# digits - domain mismatch, forced 28->299 upsampling, and its features are
# shaped to discriminate natural photos, not stroke style. Instead, train a
# plain autoencoder directly on real MNIST digits, once, and freeze it - its
# bottleneck becomes the embedding space for every FID/density/coverage call
# below. Reconstruction loss (unlike a classifier's cross-entropy loss) keeps
# within-class style information (slant, stroke width) in the embedding,
# which is exactly the axis this experiment is trying to measure.
EMBEDDING_DIM = 64
embedding_model = models.SimpleAutoencoder(embedding_dim=EMBEDDING_DIM).to(device)
embedding_optimizer = torch.optim.Adam(embedding_model.parameters(), lr=1e-3)
embedding_train_loader = DataLoader(full_dataset, batch_size=256, shuffle=True)

print("Training frozen embedding autoencoder on real MNIST images...")
embedding_model.train()
for _epoch in range(30):
    _epoch_loss = 0.0
    for x, _ in embedding_train_loader:
        x = x.to(device)
        recon, _ = embedding_model(x)
        loss = F.binary_cross_entropy(recon, x)
        embedding_optimizer.zero_grad()
        loss.backward()
        embedding_optimizer.step()
        _epoch_loss += loss.item() * x.size(0)
    _epoch_loss /= len(full_dataset)
print(f"Embedding autoencoder final reconstruction loss: {_epoch_loss:.4f}")
embedding_model.eval()

# Independent, once-trained, frozen oracle digit classifier - used only for
# the mode-coverage check (predicted-label histogram entropy + label match
# rate) below. Deliberately separate from SimpleAutoencoder (an embedding, not
# a classifier) so this check never entangles with the model being evaluated.
digit_classifier = models.MNISTClassifier().to(device)
classifier_optimizer = torch.optim.Adam(digit_classifier.parameters(), lr=1e-3)
classifier_train_loader = DataLoader(full_dataset, batch_size=256, shuffle=True)

print("Training frozen oracle digit classifier on real MNIST images...")
digit_classifier.train()
for _epoch in range(5):
    for x, y in classifier_train_loader:
        x, y = x.to(device), y.to(device)
        logits = digit_classifier(x)
        loss = F.cross_entropy(logits, y)
        classifier_optimizer.zero_grad()
        loss.backward()
        classifier_optimizer.step()
digit_classifier.eval()
_classifier_test_acc = (
    digit_classifier(torch.stack([test_dataset[i][0] for i in range(len(test_dataset))]).to(device)).argmax(dim=1)
    == torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))], device=device)
).float().mean().item()
print(f"Oracle digit classifier test accuracy: {_classifier_test_acc:.4f}")

# Real-image embedding features never change across iterations, so compute
# them once and reuse for every metric call instead of re-running the embedding
# model over the same 10,000 real test images every time. real_dc_radii (each
# real sample's k-th nearest-neighbor distance to other real samples) and
# real_fid_mu/real_fid_sigma are cached the same way.
PRDC_NEAREST_K = 10
real_embedding_features = fid_helper.extract_embedding_features(embedding_model, test_dataset, device=device)
real_dc_radii = fid_helper.build_cached_real_dc_radii(real_embedding_features, PRDC_NEAREST_K)
real_fid_mu, real_fid_sigma = fid_helper.compute_real_fid_stats(real_embedding_features)

# Per-class real labels/radii for the stratified density/coverage check: pooled
# density/coverage can't distinguish "fake population got more diverse" from
# "fake digits got sharper and more separated between classes" - both inflate
# the pooled metric identically. Stratifying by class removes that confound.
real_labels_for_dc = torch.tensor(
    [test_dataset[i][1] for i in range(len(test_dataset))], dtype=torch.long, device=device,
)
real_dc_radii_per_class = fid_helper.build_cached_real_dc_radii_per_class(
    real_embedding_features, real_labels_for_dc, PRDC_NEAREST_K,
)

# ---------------------------------------------------------------------------
# Train init model on 500 real samples
# ---------------------------------------------------------------------------
init_size = 500
init_subset = utils.get_balanced_subset(full_digit_indices, init_size)
init_dataset = Subset(full_dataset, init_subset)
init_train_loader = DataLoader(init_dataset, batch_size=128, shuffle=True)

init_model = models.CVAE(input_dim=784, label_dim=10, latent_dim=20, name="cvae_real_500", arch="conv").to(device)
train_helper.train_model(model=init_model, train_loader=init_train_loader, device=device, epochs=200, lr=1e-3, patience=5, verbose=False)
val_loss, val_recon, val_kl = train_helper.calculate_validation_loss(init_model, test_loader, device)
init_synth_images, init_synth_labels = data_helper.generate_balanced_synthetic_data(
    synthetic_model=init_model, target_size=len(test_dataset), device=device,
)
init_synth_features = fid_helper.extract_embedding_features(
    embedding_model, TensorDataset(init_synth_images, init_synth_labels), device=device,
)
fid = fid_helper.calculate_fid_from_features(real_fid_mu, real_fid_sigma, init_synth_features)
del init_synth_images, init_synth_labels, init_synth_features
print("init model fid", fid, "val_NELBO", val_loss, "val_recon", val_recon, "val_kl", val_kl)

# ---------------------------------------------------------------------------
# Initialize results dict and size schedule
# ---------------------------------------------------------------------------
# Fixed size per iteration ("Replace" setting: each model is trained only on a
# freshly resampled, constant-size synthetic batch from the previous model,
# discarding all prior generations). Matched to ELBO_experiment.py's 10,000 so
# the two runs are directly comparable generation-for-generation.
delta_size = 10_000
total_iterations = 50
test_results = {
    "model_name": [], "fid": [],
    "density": [], "coverage": [],
    "density_stratified": [], "coverage_stratified": [],
    "kid": [], "vendi": [],
    "class_entropy": [], "label_match_rate": [],
    "val_loss": [], "val_recon": [], "val_kl": [],
}
size_schedule = [delta_size] * total_iterations
all_models = []

csv_path = os.path.join(results_saved_path, f"unfiltered_D{delta_size}_results.csv")


def append_result(csv_path, row: dict):
    row_df = pd.DataFrame([row])
    header_needed = not os.path.exists(csv_path)
    row_df.to_csv(csv_path, mode="a", header=header_needed, index=False)

# ---------------------------------------------------------------------------
# Iterative retraining loop (no verifier anywhere in the chain)
# ---------------------------------------------------------------------------
this_model = init_model

for i, synthetic_size in enumerate(size_schedule):
    i = i + 1
    synthetic_size = int(synthetic_size)
    iter_start = time.time()

    # Generate the raw, unfiltered synthetic batch. Unlike the filtered run this
    # never touches disk: the same in-memory batch is both what gets measured
    # and what the next generation trains on.
    t0 = time.time()
    synth_images, synth_labels = data_helper.generate_balanced_synthetic_data(
        synthetic_model=this_model, target_size=synthetic_size, device=device,
    )
    synth_dataset = TensorDataset(synth_images, synth_labels)
    t_generate = time.time() - t0

    t0 = time.time()
    synth_features = fid_helper.extract_embedding_features(embedding_model, synth_dataset, device=device)
    fid_score = fid_helper.calculate_fid_from_features(real_fid_mu, real_fid_sigma, synth_features)
    dc = fid_helper.compute_density_coverage(real_dc_radii, real_embedding_features, synth_features, PRDC_NEAREST_K)
    dc_stratified = fid_helper.compute_density_coverage_stratified(
        real_dc_radii_per_class, real_embedding_features, real_labels_for_dc,
        synth_features, synth_labels.to(device), PRDC_NEAREST_K,
    )
    kid_score = fid_helper.compute_kid(real_embedding_features, synth_features)
    vendi = fid_helper.compute_vendi_score_per_class(synth_features, synth_labels.to(device))
    mode_coverage = fid_helper.compute_class_mode_coverage(digit_classifier, synth_dataset, device)
    save_preview_grid(synth_images, synth_labels, os.path.join(picture_saved_path, f"iter{i}_unfiltered.png"))
    del synth_features
    t_metrics = time.time() - t0

    t0 = time.time()
    synthetic_loader = DataLoader(synth_dataset, batch_size=128, shuffle=True)
    synthetic_model = models.CVAE(
        input_dim=784, label_dim=10, latent_dim=20,
        name=f"cvae_nofilter_iter{i}_{synthetic_size}", arch="conv",
    ).to(device)
    train_helper.train_model(synthetic_model, synthetic_loader, device, epochs=200, lr=1e-3, patience=5, verbose=False)
    t_cvae_train = time.time() - t0

    this_model = synthetic_model
    all_models.append(this_model)

    t0 = time.time()
    val_loss, val_recon, val_kl = train_helper.calculate_validation_loss(this_model, test_loader, device)
    t_validation = time.time() - t0

    iter_total = time.time() - iter_start
    print(f"[timing] iter {i}: generate={t_generate:.1f}s, metrics={t_metrics:.1f}s, "
          f"cvae_train={t_cvae_train:.1f}s, validation={t_validation:.1f}s, TOTAL={iter_total:.1f}s")

    test_results["model_name"].append(this_model.get_name())
    test_results["fid"].append(fid_score)
    test_results["density"].append(dc["density"])
    test_results["coverage"].append(dc["coverage"])
    test_results["density_stratified"].append(dc_stratified["density_stratified"])
    test_results["coverage_stratified"].append(dc_stratified["coverage_stratified"])
    test_results["kid"].append(kid_score)
    test_results["vendi"].append(vendi["vendi_mean"])
    test_results["class_entropy"].append(mode_coverage["class_entropy"])
    test_results["label_match_rate"].append(mode_coverage["label_match_rate"])
    test_results["val_loss"].append(val_loss)
    test_results["val_recon"].append(val_recon)
    test_results["val_kl"].append(val_kl)

    append_result(csv_path, {
        "model_name": test_results["model_name"][-1],
        "fid": test_results["fid"][-1],
        "density": test_results["density"][-1],
        "coverage": test_results["coverage"][-1],
        "density_stratified": test_results["density_stratified"][-1],
        "coverage_stratified": test_results["coverage_stratified"][-1],
        "kid": test_results["kid"][-1],
        "vendi": test_results["vendi"][-1],
        "class_entropy": test_results["class_entropy"][-1],
        "label_match_rate": test_results["label_match_rate"][-1],
        "val_loss": test_results["val_loss"][-1],
        "val_recon": test_results["val_recon"][-1],
        "val_kl": test_results["val_kl"][-1],
    })
    save_metric_plots(test_results, plots_saved_path)

    print(f"Iteration {i} - Ending model: {this_model.get_name()}, FID: {test_results['fid'][-1]:.2f}, "
          f"KID: {test_results['kid'][-1]:.4f}, "
          f"Coverage: {test_results['coverage'][-1]:.3f} (stratified: {test_results['coverage_stratified'][-1]:.3f}), "
          f"Density: {test_results['density'][-1]:.3f} (stratified: {test_results['density_stratified'][-1]:.3f}), "
          f"Vendi: {test_results['vendi'][-1]:.2f}, "
          f"Class entropy: {test_results['class_entropy'][-1]:.3f}, Label match: {test_results['label_match_rate'][-1]:.3f}, "
          f"Test NELBO: {test_results['val_loss'][-1]:.2f}")

    del synthetic_loader
    del synth_dataset
    del synth_images, synth_labels

# ---------------------------------------------------------------------------
# Results were appended to csv_path after every iteration above.
# ---------------------------------------------------------------------------
res_table = pd.DataFrame.from_dict(test_results, orient="columns")
print(f"\nResults saved incrementally to {csv_path}")
print(f"Per-metric line plots saved to {plots_saved_path}")
print(res_table)
