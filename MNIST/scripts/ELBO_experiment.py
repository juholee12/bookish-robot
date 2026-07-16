"""
Iterative retraining with label-smoothed conditional discriminator.
"""
import sys
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, Subset
import torch.nn.functional as F

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import glob
import random
import tempfile
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
        # run as a subprocess (e.g. `!python ELBO_experiment.py`), so it will
        # crash here in that case. Mount Drive from an actual notebook cell
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
ROOT = DRIVE_BASE / "runs" / RUN_ID
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

# Real-image FID features never change across iterations, so compute them once
# and reuse for every fid_unfiltered/fid_filtered call instead of re-running
# Inception over the same 10,000 real test images every time.
real_fid_metric = fid_helper.build_cached_real_fid(test_dataset, device=device)

# Same real-image Inception features, reused for Density/Coverage every
# iteration instead of re-extracting them each time. real_dc_radii (each real
# sample's k-th nearest-neighbor distance to other real samples) is also
# cached once here since it only depends on real_prdc_features, which never
# changes - this is the expensive (N_real x N_real) part of the computation.
PRDC_NEAREST_K = 10
real_prdc_features = fid_helper.extract_inception_features(real_fid_metric, test_dataset, device=device)
real_dc_radii = fid_helper.build_cached_real_dc_radii(real_prdc_features, PRDC_NEAREST_K)


def _build_real_half(real_dataset, device):
    """One-time extraction of a real dataset's images + one-hot(digit, is_real=1) labels."""
    real_images = torch.stack([real_dataset[i][0] for i in range(len(real_dataset))]).to(device)
    real_labels = torch.tensor([real_dataset[i][1] for i in range(len(real_dataset))], dtype=torch.long, device=device)
    y_real_labels = torch.cat([
        F.one_hot(real_labels, num_classes=10).float(),
        torch.ones(len(real_dataset), 1, dtype=torch.long, device=device),
    ], dim=1)
    return real_images, y_real_labels


def build_discriminator_dataset_cached(real_images, y_real_labels, synthetic_model, device):
    """Same output as data_helper.prepare_discriminator_dataset_with_labels, but
    reuses a precomputed real-image half instead of re-extracting it every call."""
    real_size = real_images.shape[0]
    synthetic_images, synthetic_labels = data_helper.generate_balanced_synthetic_data(
        synthetic_model, real_size, device=device,
    )
    synthetic_images = synthetic_images.to(device)
    synthetic_labels = synthetic_labels.to(device)
    y_synthetic_labels = torch.cat([
        F.one_hot(synthetic_labels, num_classes=10).float(),
        torch.zeros(len(synthetic_images), 1, dtype=torch.long, device=device),
    ], dim=1)
    X_all = torch.cat([real_images, synthetic_images], dim=0)
    y_all = torch.cat([y_real_labels, y_synthetic_labels], dim=0)
    return TensorDataset(X_all, y_all)


# The real half of the discriminator's training/validation data never changes
# across iterations (only this_model's synthetic half does), so extract it
# from full_dataset/test_dataset once here instead of every iteration.
print("Precomputing real-image halves for discriminator datasets (cached once, reused every iteration)...")
real_images_full, y_real_labels_full = _build_real_half(full_dataset, device)
real_images_test, y_real_labels_test = _build_real_half(test_dataset, device)

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
fid = fid_helper.calculate_fid_from_model(real_ds=test_dataset, model=init_model, device=device)
print("init model fid", fid, "val_NELBO", val_loss, "val_recon", val_recon, "val_kl", val_kl)

# ---------------------------------------------------------------------------
# Initialize results dict and size schedule
# ---------------------------------------------------------------------------
# Fixed size per iteration ("Replace" setting: each model is trained only on a
# freshly resampled, constant-size synthetic batch from the previous model,
# discarding all prior generations - matches Gerstgrasser et al. 2024's
# classic model-collapse baseline, as opposed to a growing "Replace-Multiple"
# schedule). 10,000 keeps FID reasonably well-conditioned (Inception features
# are 2048-dim; too few samples biases FID upward from sampling noise alone)
# while staying far cheaper per-iteration than the old growing schedule.
delta_size = 10_000
total_iterations = 50
test_results = {
    "model_name": [], "fid_unfiltered": [], "fid_filtered": [],
    "density_unfiltered": [], "coverage_unfiltered": [],
    "density_filtered": [], "coverage_filtered": [],
    "val_loss": [], "val_recon": [], "val_kl": [],
    "disc_train_loss": [], "disc_val_loss": [], "disc_test_accuracy": [],
}
size_schedule = [delta_size] * total_iterations
all_models = []

csv_path = os.path.join(results_saved_path, f"label_smoothing_D{delta_size}_results.csv")


def append_result(csv_path, row: dict):
    row_df = pd.DataFrame([row])
    header_needed = not os.path.exists(csv_path)
    row_df.to_csv(csv_path, mode="a", header=header_needed, index=False)

# ---------------------------------------------------------------------------
# Iterative retraining loop
# ---------------------------------------------------------------------------
this_model = init_model

for i, synthetic_size in enumerate(size_schedule):
    filter_thres = 0.1
    i = i + 1
    synthetic_size = int(synthetic_size)
    iter_start = time.time()

    t0 = time.time()
    discriminator_dataset = build_discriminator_dataset_cached(real_images_full, y_real_labels_full, this_model, device)
    disc_loader = DataLoader(discriminator_dataset, batch_size=128, shuffle=True)

    disc_test_dataset = build_discriminator_dataset_cached(real_images_test, y_real_labels_test, this_model, device)
    disc_test_loader = DataLoader(disc_test_dataset, batch_size=128, shuffle=True)

    # Train Discriminator with Label Smoothing and dropout
    disc_model = models.ConditionalDiscriminator(input_dim=784, name="disc_mlp_" + str(synthetic_size), arch="mlp", dropout=0.1, label_smoothing=0.05)
    disc_history = train_helper.train_model_with_validation(
        model=disc_model, train_loader=disc_loader, val_loader=disc_test_loader,
        device=device, epochs=200, lr=1e-3, wd=0, patience=5, verbose=False,
    )
    t_discriminator = time.time() - t0

    print(f"Iteration {i}, disc_epochs_trained: {disc_history['epochs_trained']}, disc_best_train_loss: {disc_history['best_train_loss']}, disc_best_val_loss: {disc_history['best_val_loss']}")
    print("disc_train_last_summary:", disc_history['train_last_summary'])
    print("disc_val_last_summary:", disc_history['val_last_summary'])
    print(f"filter_thres: {filter_thres}")

    # Generate unfiltered synthetic data from the current generator (kept in memory only,
    # never written to disk) and measure its FID directly
    t0 = time.time()
    unfiltered_images, unfiltered_labels = data_helper.generate_balanced_synthetic_data(
        synthetic_model=this_model, target_size=synthetic_size, device=device,
    )
    fid_unfiltered = fid_helper.calculate_fid_score_cached(
        real_fid_metric, TensorDataset(unfiltered_images, unfiltered_labels), device=device,
    )
    unfiltered_features = fid_helper.extract_inception_features(
        real_fid_metric, TensorDataset(unfiltered_images, unfiltered_labels), device=device,
    )
    dc_unfiltered = fid_helper.compute_density_coverage(real_dc_radii, real_prdc_features, unfiltered_features, PRDC_NEAREST_K)
    save_preview_grid(unfiltered_images, unfiltered_labels, os.path.join(picture_saved_path, f"iter{i}_unfiltered.png"))
    del unfiltered_images, unfiltered_labels, unfiltered_features
    t_unfiltered = time.time() - t0

    # Generate filtered synthetic data into a local scratch directory (not under Drive's
    # persistent ROOT) since generate_balanced_images_with_filtering/create_directory_based_dataloader
    # require a directory of .pt shards to work from - this gets deleted at the end of the
    # iteration instead of being kept in data_saved.
    t0 = time.time()
    synthetic_data_load_path = tempfile.mkdtemp(prefix=f"iter{i}_filtered_")
    data_helper.generate_balanced_images_with_filtering(
        model=this_model, save_directory=synthetic_data_load_path,
        total_samples=synthetic_size, discriminator=disc_model,
        selection_threshold=filter_thres, verbose=False, use_quantile_filtering=True,
    )
    t_filtered_gen = time.time() - t0

    # Preview grid from the first saved shard (already written to disk, no extra generation)
    first_shard = sorted(glob.glob(os.path.join(synthetic_data_load_path, "*.pt")))[0]
    shard_data = torch.load(first_shard, map_location="cpu")
    save_preview_grid(shard_data["images"], shard_data["labels"], os.path.join(picture_saved_path, f"iter{i}_filtered.png"))
    del shard_data

    # Train Synthetic Model (keep_data=False: scratch directory is deleted once synthetic_loader
    # goes out of scope below, instead of persisting to Drive)
    synthetic_loader = data_helper.create_directory_based_dataloader(synthetic_data_load_path, batch_size=128, keep_data=False)

    # Measure FID directly on the filtered data used to train the next model
    t0 = time.time()
    fid_filtered = fid_helper.calculate_fid_score_cached(real_fid_metric, synthetic_loader.dataset, device=device)
    filtered_features = fid_helper.extract_inception_features(real_fid_metric, synthetic_loader.dataset, device=device)
    dc_filtered = fid_helper.compute_density_coverage(real_dc_radii, real_prdc_features, filtered_features, PRDC_NEAREST_K)
    del filtered_features
    t_filtered_metrics = time.time() - t0

    t0 = time.time()
    synthetic_model = models.CVAE(
        input_dim=784, label_dim=10, latent_dim=20,
        name=f"cvae_q{filter_thres}_iter{i}_{synthetic_size}", arch="conv",
    ).to(device)
    train_helper.train_model(synthetic_model, synthetic_loader, device, epochs=200, lr=1e-3, patience=5, verbose=False)
    t_cvae_train = time.time() - t0

    this_model = synthetic_model
    all_models.append(this_model)

    t0 = time.time()
    val_loss, val_recon, val_kl = train_helper.calculate_validation_loss(this_model, test_loader, device)
    t_validation = time.time() - t0

    iter_total = time.time() - iter_start
    print(f"[timing] iter {i}: discriminator={t_discriminator:.1f}s, unfiltered_gen+metrics={t_unfiltered:.1f}s, "
          f"filtered_gen={t_filtered_gen:.1f}s, filtered_metrics={t_filtered_metrics:.1f}s, "
          f"cvae_train={t_cvae_train:.1f}s, validation={t_validation:.1f}s, TOTAL={iter_total:.1f}s")

    test_results["model_name"].append(this_model.get_name())
    test_results["fid_unfiltered"].append(fid_unfiltered)
    test_results["fid_filtered"].append(fid_filtered)
    test_results["density_unfiltered"].append(dc_unfiltered["density"])
    test_results["coverage_unfiltered"].append(dc_unfiltered["coverage"])
    test_results["density_filtered"].append(dc_filtered["density"])
    test_results["coverage_filtered"].append(dc_filtered["coverage"])
    test_results["val_loss"].append(val_loss)
    test_results["val_recon"].append(val_recon)
    test_results["val_kl"].append(val_kl)
    test_results["disc_train_loss"].append(disc_history['best_train_loss'])
    test_results["disc_val_loss"].append(disc_history['best_val_loss'])
    test_results["disc_test_accuracy"].append(disc_history['val_last_summary']['accuracy'])

    append_result(csv_path, {
        "model_name": test_results["model_name"][-1],
        "fid_unfiltered": test_results["fid_unfiltered"][-1],
        "fid_filtered": test_results["fid_filtered"][-1],
        "density_unfiltered": test_results["density_unfiltered"][-1],
        "coverage_unfiltered": test_results["coverage_unfiltered"][-1],
        "density_filtered": test_results["density_filtered"][-1],
        "coverage_filtered": test_results["coverage_filtered"][-1],
        "val_loss": test_results["val_loss"][-1],
        "val_recon": test_results["val_recon"][-1],
        "val_kl": test_results["val_kl"][-1],
        "disc_train_loss": test_results["disc_train_loss"][-1],
        "disc_val_loss": test_results["disc_val_loss"][-1],
        "disc_test_accuracy": test_results["disc_test_accuracy"][-1],
    })
    save_metric_plots(test_results, plots_saved_path)

    print(f"Iteration {i} - Ending model: {this_model.get_name()}, FID unfiltered: {test_results['fid_unfiltered'][-1]:.2f}, FID filtered: {test_results['fid_filtered'][-1]:.2f}, "
          f"Coverage filtered: {test_results['coverage_filtered'][-1]:.3f}, Density filtered: {test_results['density_filtered'][-1]:.3f}, Test NELBO: {test_results['val_loss'][-1]:.2f}")

    del synthetic_loader
    del disc_model
    del discriminator_dataset
    del disc_loader
    del disc_test_dataset
    del disc_test_loader

# ---------------------------------------------------------------------------
# Results were appended to csv_path after every iteration above.
# ---------------------------------------------------------------------------
res_table = pd.DataFrame.from_dict(test_results, orient="columns")
print(f"\nResults saved incrementally to {csv_path}")
print(f"Per-metric line plots saved to {plots_saved_path}")
print(res_table)
