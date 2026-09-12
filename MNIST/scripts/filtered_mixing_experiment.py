"""
Iterative retraining with a verifier AND real-data mixing.

Companion to ELBO_experiment.py. The baseline script trains each generation
purely on the top-10%-by-discriminator-score subset of its predecessor's output
(strict "replace"). This one keeps that verifier exactly as-is but additionally
mixes a fixed fraction rho of fresh real MNIST images into every generation's
training set, the CVAE analogue of the real-data-mixing condition in the
Gaussian toy model.

The mixed training set size is held at delta_size, matching the baseline run, so
the two are comparable generation-for-generation: rho * delta_size real images
replace (rather than add to) that many filtered synthetic images. Real images
are drawn fresh from the full 60,000-image MNIST training set every generation,
balanced across digits.

Metric columns are deliberately identical to ELBO_experiment.py's CSV so the two
runs can be compared or concatenated directly. Note that the _filtered metrics
here are measured on the synthetic portion ONLY (before real images are mixed
in), so they remain a like-for-like measurement of what the generator produced
rather than being inflated by the real images blended into training.
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
        # run as a subprocess (e.g. `!python filtered_mixing_experiment.py`), so
        # it will crash here in that case. Mount Drive from an actual notebook
        # cell first (`from google.colab import drive; drive.mount('/content/drive')`)
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
ROOT = DRIVE_BASE / "runs" / f"filtered_mixing_{RUN_ID}"
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
# Real-data mixing configuration
# ---------------------------------------------------------------------------
# rho: fraction of each generation's training set drawn from real MNIST, matching
# the Gaussian toy model's mixing parameter. Real images REPLACE synthetic ones
# rather than being added on top, so the total training set size per generation
# stays at delta_size and stays comparable with the no-mixing baseline run.
REAL_MIX_RATIO = 0.1

# Synthetic samples come out of CVAE.sample_x_given_y as Bernoulli draws, i.e.
# strictly {0, 1} pixels, while raw MNIST is continuous in [0, 1]. Binarizing the
# mixed-in real images keeps the training distribution homogeneous, so what the
# mixing injects is true-distribution *content* (stroke style, slant, shape
# variety) rather than a pixel-format difference the model could key on. Set
# this to False to mix in the continuous-valued originals instead.
BINARIZE_REAL_MIX = True


def write_real_mix_shard(save_directory, n_real, generation, digit_indices, dataset, num_classes=10):
    """Sample a fresh, digit-balanced batch of real MNIST images and write it into
    save_directory as one more .pt shard, in the same {'images','labels'} format
    the synthetic shards use, so create_directory_based_dataloader picks it up
    alongside them with no special-casing.

    Sampling is fresh every generation (drawn from the full 60k training pool,
    without replacement within a generation), so the mixing stream never degrades
    into repeatedly showing the model the same handful of real digits.
    """
    per_digit = n_real // num_classes
    chosen = []
    for digit in range(num_classes):
        pool = digit_indices[digit]
        chosen.extend(random.sample(pool, per_digit))

    images = torch.stack([dataset[j][0] for j in chosen])
    labels = torch.tensor([dataset[j][1] for j in chosen], dtype=torch.long)
    if BINARIZE_REAL_MIX:
        images = (images > 0.5).float()

    shard_path = os.path.join(save_directory, f"realmix_{len(images)}_g{generation}.pt")
    torch.save({"images": images, "labels": labels}, shard_path)
    return images, labels

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
# rate) below. Deliberately separate from ConditionalDiscriminator (real/fake,
# retrained adversarially every iteration) and SimpleAutoencoder (an
# embedding, not a classifier) so this check never entangles with the models
# actually being evaluated.
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
# them once and reuse for every fid_unfiltered/fid_filtered call instead of
# re-running the embedding model over the same 10,000 real test images every
# time. real_dc_radii (each real sample's k-th nearest-neighbor distance to
# other real samples) and real_fid_mu/real_fid_sigma are cached the same way.
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
# Fixed training-set size per iteration, matched to ELBO_experiment.py's 10,000
# so the mixing run is comparable generation-for-generation with the no-mixing
# baseline. Of these, REAL_MIX_RATIO * delta_size come from real MNIST and the
# rest from verifier-filtered synthetic output.
delta_size = 10_000
total_iterations = 50
real_per_iteration = int(round(delta_size * REAL_MIX_RATIO))
synthetic_per_iteration = delta_size - real_per_iteration
print(f"Training set per generation: {synthetic_per_iteration} filtered synthetic "
      f"+ {real_per_iteration} real (rho={REAL_MIX_RATIO}), total {delta_size}")

test_results = {
    "model_name": [], "fid_unfiltered": [], "fid_filtered": [],
    "density_unfiltered": [], "coverage_unfiltered": [],
    "density_filtered": [], "coverage_filtered": [],
    "density_stratified_unfiltered": [], "coverage_stratified_unfiltered": [],
    "density_stratified_filtered": [], "coverage_stratified_filtered": [],
    "kid_unfiltered": [], "kid_filtered": [],
    "vendi_unfiltered": [], "vendi_filtered": [],
    "class_entropy_unfiltered": [], "label_match_rate_unfiltered": [],
    "class_entropy_filtered": [], "label_match_rate_filtered": [],
    "val_loss": [], "val_recon": [], "val_kl": [],
    "disc_train_loss": [], "disc_val_loss": [], "disc_test_accuracy": [],
}
size_schedule = [synthetic_per_iteration] * total_iterations
all_models = []

csv_path = os.path.join(results_saved_path, f"filtered_mixing_rho{REAL_MIX_RATIO}_D{delta_size}_results.csv")


def append_result(csv_path, row: dict):
    row_df = pd.DataFrame([row])
    header_needed = not os.path.exists(csv_path)
    row_df.to_csv(csv_path, mode="a", header=header_needed, index=False)

# ---------------------------------------------------------------------------
# Iterative retraining loop (verifier filtering + real-data mixing)
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
    # never written to disk) and measure its FID directly. Sized to delta_size rather than
    # synthetic_size so this diagnostic stays comparable with the baseline run's.
    t0 = time.time()
    unfiltered_images, unfiltered_labels = data_helper.generate_balanced_synthetic_data(
        synthetic_model=this_model, target_size=delta_size, device=device,
    )
    unfiltered_features = fid_helper.extract_embedding_features(
        embedding_model, TensorDataset(unfiltered_images, unfiltered_labels), device=device,
    )
    fid_unfiltered = fid_helper.calculate_fid_from_features(real_fid_mu, real_fid_sigma, unfiltered_features)
    dc_unfiltered = fid_helper.compute_density_coverage(real_dc_radii, real_embedding_features, unfiltered_features, PRDC_NEAREST_K)
    dc_stratified_unfiltered = fid_helper.compute_density_coverage_stratified(
        real_dc_radii_per_class, real_embedding_features, real_labels_for_dc,
        unfiltered_features, unfiltered_labels.to(device), PRDC_NEAREST_K,
    )
    kid_unfiltered = fid_helper.compute_kid(real_embedding_features, unfiltered_features)
    vendi_unfiltered = fid_helper.compute_vendi_score_per_class(unfiltered_features, unfiltered_labels.to(device))
    mode_unfiltered = fid_helper.compute_class_mode_coverage(
        digit_classifier, TensorDataset(unfiltered_images, unfiltered_labels), device,
    )
    save_preview_grid(unfiltered_images, unfiltered_labels, os.path.join(picture_saved_path, f"iter{i}_unfiltered.png"))
    del unfiltered_images, unfiltered_labels, unfiltered_features
    t_unfiltered = time.time() - t0

    # Generate filtered synthetic data into a local scratch directory (not under Drive's
    # persistent ROOT) since generate_balanced_images_with_filtering/create_directory_based_dataloader
    # require a directory of .pt shards to work from - this gets deleted at the end of the
    # iteration instead of being kept in data_saved.
    t0 = time.time()
    synthetic_data_load_path = tempfile.mkdtemp(prefix=f"iter{i}_filtered_mix_")
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

    # Measure the filtered metrics on the synthetic portion ONLY. This loader is built
    # before any real images land in the directory, and DirectoryBasedSyntheticDataset
    # fixes its shard list at construction time, so it stays synthetic-only even after
    # the real shard is written below. keep_data=True so its cleanup does not delete the
    # directory out from under the training loader.
    t0 = time.time()
    metrics_loader = data_helper.create_directory_based_dataloader(synthetic_data_load_path, batch_size=128, keep_data=True)
    filtered_features = fid_helper.extract_embedding_features(embedding_model, metrics_loader.dataset, device=device)
    filtered_labels = fid_helper.extract_labels(metrics_loader.dataset).to(device)
    fid_filtered = fid_helper.calculate_fid_from_features(real_fid_mu, real_fid_sigma, filtered_features)
    dc_filtered = fid_helper.compute_density_coverage(real_dc_radii, real_embedding_features, filtered_features, PRDC_NEAREST_K)
    dc_stratified_filtered = fid_helper.compute_density_coverage_stratified(
        real_dc_radii_per_class, real_embedding_features, real_labels_for_dc,
        filtered_features, filtered_labels, PRDC_NEAREST_K,
    )
    kid_filtered = fid_helper.compute_kid(real_embedding_features, filtered_features)
    vendi_filtered = fid_helper.compute_vendi_score_per_class(filtered_features, filtered_labels)
    mode_filtered = fid_helper.compute_class_mode_coverage(digit_classifier, metrics_loader.dataset, device)
    del filtered_features, filtered_labels, metrics_loader
    t_filtered_metrics = time.time() - t0

    # Mix in a fresh, digit-balanced batch of real MNIST images, then build the training
    # loader over synthetic + real together (keep_data=False: this loader owns the scratch
    # directory and deletes it once it goes out of scope at the end of the iteration).
    t0 = time.time()
    real_mix_images, real_mix_labels = write_real_mix_shard(
        synthetic_data_load_path, real_per_iteration, i, full_digit_indices, full_dataset,
    )
    save_preview_grid(real_mix_images, real_mix_labels, os.path.join(picture_saved_path, f"iter{i}_realmix.png"))
    del real_mix_images, real_mix_labels
    synthetic_loader = data_helper.create_directory_based_dataloader(synthetic_data_load_path, batch_size=128, keep_data=False)
    t_real_mix = time.time() - t0
    print(f"  mixed training set size: {len(synthetic_loader.dataset)} "
          f"({synthetic_size} filtered synthetic + {real_per_iteration} real)")

    t0 = time.time()
    synthetic_model = models.CVAE(
        input_dim=784, label_dim=10, latent_dim=20,
        name=f"cvae_q{filter_thres}_mix{REAL_MIX_RATIO}_iter{i}_{delta_size}", arch="conv",
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
          f"real_mix={t_real_mix:.1f}s, cvae_train={t_cvae_train:.1f}s, validation={t_validation:.1f}s, "
          f"TOTAL={iter_total:.1f}s")

    test_results["model_name"].append(this_model.get_name())
    test_results["fid_unfiltered"].append(fid_unfiltered)
    test_results["fid_filtered"].append(fid_filtered)
    test_results["density_unfiltered"].append(dc_unfiltered["density"])
    test_results["coverage_unfiltered"].append(dc_unfiltered["coverage"])
    test_results["density_filtered"].append(dc_filtered["density"])
    test_results["coverage_filtered"].append(dc_filtered["coverage"])
    test_results["density_stratified_unfiltered"].append(dc_stratified_unfiltered["density_stratified"])
    test_results["coverage_stratified_unfiltered"].append(dc_stratified_unfiltered["coverage_stratified"])
    test_results["density_stratified_filtered"].append(dc_stratified_filtered["density_stratified"])
    test_results["coverage_stratified_filtered"].append(dc_stratified_filtered["coverage_stratified"])
    test_results["kid_unfiltered"].append(kid_unfiltered)
    test_results["kid_filtered"].append(kid_filtered)
    test_results["vendi_unfiltered"].append(vendi_unfiltered["vendi_mean"])
    test_results["vendi_filtered"].append(vendi_filtered["vendi_mean"])
    test_results["class_entropy_unfiltered"].append(mode_unfiltered["class_entropy"])
    test_results["label_match_rate_unfiltered"].append(mode_unfiltered["label_match_rate"])
    test_results["class_entropy_filtered"].append(mode_filtered["class_entropy"])
    test_results["label_match_rate_filtered"].append(mode_filtered["label_match_rate"])
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
        "density_stratified_unfiltered": test_results["density_stratified_unfiltered"][-1],
        "coverage_stratified_unfiltered": test_results["coverage_stratified_unfiltered"][-1],
        "density_stratified_filtered": test_results["density_stratified_filtered"][-1],
        "coverage_stratified_filtered": test_results["coverage_stratified_filtered"][-1],
        "kid_unfiltered": test_results["kid_unfiltered"][-1],
        "kid_filtered": test_results["kid_filtered"][-1],
        "vendi_unfiltered": test_results["vendi_unfiltered"][-1],
        "vendi_filtered": test_results["vendi_filtered"][-1],
        "class_entropy_unfiltered": test_results["class_entropy_unfiltered"][-1],
        "label_match_rate_unfiltered": test_results["label_match_rate_unfiltered"][-1],
        "class_entropy_filtered": test_results["class_entropy_filtered"][-1],
        "label_match_rate_filtered": test_results["label_match_rate_filtered"][-1],
        "val_loss": test_results["val_loss"][-1],
        "val_recon": test_results["val_recon"][-1],
        "val_kl": test_results["val_kl"][-1],
        "disc_train_loss": test_results["disc_train_loss"][-1],
        "disc_val_loss": test_results["disc_val_loss"][-1],
        "disc_test_accuracy": test_results["disc_test_accuracy"][-1],
    })
    save_metric_plots(test_results, plots_saved_path)

    print(f"Iteration {i} - Ending model: {this_model.get_name()}, FID unfiltered: {test_results['fid_unfiltered'][-1]:.2f}, FID filtered: {test_results['fid_filtered'][-1]:.2f}, "
          f"KID filtered: {test_results['kid_filtered'][-1]:.4f}, "
          f"Coverage filtered: {test_results['coverage_filtered'][-1]:.3f} (stratified: {test_results['coverage_stratified_filtered'][-1]:.3f}), "
          f"Density filtered: {test_results['density_filtered'][-1]:.3f} (stratified: {test_results['density_stratified_filtered'][-1]:.3f}), "
          f"Vendi filtered: {test_results['vendi_filtered'][-1]:.2f}, "
          f"Class entropy filtered: {test_results['class_entropy_filtered'][-1]:.3f}, Label match filtered: {test_results['label_match_rate_filtered'][-1]:.3f}, "
          f"Test NELBO: {test_results['val_loss'][-1]:.2f}")

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
