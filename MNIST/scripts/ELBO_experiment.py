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
data_saved_path = os.path.join(ROOT, "data_saved")
results_saved_path = os.path.join(ROOT, "results_saved")
picture_saved_path = os.path.join(ROOT, "picture_saved")
os.makedirs(results_saved_path, exist_ok=True)
os.makedirs(model_saved_path, exist_ok=True)
os.makedirs(data_saved_path, exist_ok=True)
os.makedirs(picture_saved_path, exist_ok=True)


def save_preview_grid(images, labels, save_path, per_class=5, num_classes=10):
    """Cheap PNG preview: a handful of samples per digit, not the full batch."""
    images = images.detach().cpu()
    labels = labels.detach().cpu()
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
delta_size = 5_000
total_iterations = 50
test_results = {
    "model_name": [], "fid_unfiltered": [], "fid_filtered": [],
    "val_loss": [], "val_recon": [], "val_kl": [],
    "disc_train_loss": [], "disc_val_loss": [], "disc_test_accuracy": [],
}
size_schedule = [delta_size * i for i in range(1, total_iterations + 1)]
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

    discriminator_dataset = data_helper.prepare_discriminator_dataset_with_labels(full_dataset, this_model, device)
    disc_loader = DataLoader(discriminator_dataset, batch_size=128, shuffle=True)

    disc_test_dataset = data_helper.prepare_discriminator_dataset_with_labels(test_dataset, this_model, device)
    disc_test_loader = DataLoader(disc_test_dataset, batch_size=128, shuffle=True)

    # Train Discriminator with Label Smoothing and dropout
    disc_model = models.ConditionalDiscriminator(input_dim=784, name="disc_mlp_" + str(synthetic_size), arch="mlp", dropout=0.1, label_smoothing=0.05)
    disc_history = train_helper.train_model_with_validation(
        model=disc_model, train_loader=disc_loader, val_loader=disc_test_loader,
        device=device, epochs=200, lr=1e-3, wd=0, patience=5, verbose=False,
    )

    print(f"Iteration {i}, disc_epochs_trained: {disc_history['epochs_trained']}, disc_best_train_loss: {disc_history['best_train_loss']}, disc_best_val_loss: {disc_history['best_val_loss']}")
    print("disc_train_last_summary:", disc_history['train_last_summary'])
    print("disc_val_last_summary:", disc_history['val_last_summary'])
    print(f"filter_thres: {filter_thres}")

    # Generate + save unfiltered synthetic data from the current generator, and measure its FID directly
    unfiltered_data_path = os.path.join(data_saved_path, f'iter{i}_unfiltered')
    os.makedirs(unfiltered_data_path, exist_ok=True)
    unfiltered_images, unfiltered_labels = data_helper.generate_balanced_synthetic_data(
        synthetic_model=this_model, target_size=synthetic_size, device=device,
    )
    torch.save(
        {"images": unfiltered_images.cpu(), "labels": unfiltered_labels.cpu()},
        os.path.join(unfiltered_data_path, "unfiltered.pt"),
    )
    fid_unfiltered = fid_helper.calculate_fid_score_cached(
        real_fid_metric, TensorDataset(unfiltered_images, unfiltered_labels), device=device,
    )
    save_preview_grid(unfiltered_images, unfiltered_labels, os.path.join(picture_saved_path, f"iter{i}_unfiltered.png"))
    del unfiltered_images, unfiltered_labels

    # Generate Synthetic Data
    synthetic_data_load_path = os.path.join(data_saved_path, f'iter{i}_filtered')
    data_helper.generate_balanced_images_with_filtering(
        model=this_model, save_directory=synthetic_data_load_path,
        total_samples=synthetic_size, discriminator=disc_model,
        selection_threshold=filter_thres, verbose=False, use_quantile_filtering=True,
    )

    # Preview grid from the first saved shard (already written to disk, no extra generation)
    first_shard = sorted(glob.glob(os.path.join(synthetic_data_load_path, "*.pt")))[0]
    shard_data = torch.load(first_shard, map_location="cpu")
    save_preview_grid(shard_data["images"], shard_data["labels"], os.path.join(picture_saved_path, f"iter{i}_filtered.png"))
    del shard_data

    # Train Synthetic Model
    synthetic_loader = data_helper.create_directory_based_dataloader(synthetic_data_load_path, batch_size=128, keep_data=True)

    # Measure FID directly on the filtered data used to train the next model
    fid_filtered = fid_helper.calculate_fid_score_cached(real_fid_metric, synthetic_loader.dataset, device=device)

    synthetic_model = models.CVAE(
        input_dim=784, label_dim=10, latent_dim=20,
        name=f"cvae_q{filter_thres}_iter{i}_{synthetic_size}", arch="conv",
    ).to(device)
    train_helper.train_model(synthetic_model, synthetic_loader, device, epochs=200, lr=1e-3, patience=5, verbose=False)

    this_model = synthetic_model
    all_models.append(this_model)

    val_loss, val_recon, val_kl = train_helper.calculate_validation_loss(this_model, test_loader, device)

    test_results["model_name"].append(this_model.get_name())
    test_results["fid_unfiltered"].append(fid_unfiltered)
    test_results["fid_filtered"].append(fid_filtered)
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
        "val_loss": test_results["val_loss"][-1],
        "val_recon": test_results["val_recon"][-1],
        "val_kl": test_results["val_kl"][-1],
        "disc_train_loss": test_results["disc_train_loss"][-1],
        "disc_val_loss": test_results["disc_val_loss"][-1],
        "disc_test_accuracy": test_results["disc_test_accuracy"][-1],
    })

    print(f"Iteration {i} - Ending model: {this_model.get_name()}, FID unfiltered: {test_results['fid_unfiltered'][-1]:.2f}, FID filtered: {test_results['fid_filtered'][-1]:.2f}, Test NELBO: {test_results['val_loss'][-1]:.2f}")

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
print(res_table)
