import sys
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# train_full_dataset_model.py 位于 MNIST/scripts/
THIS_DIR = Path(__file__).resolve().parent
MNIST_ROOT = THIS_DIR.parent
PROJECT_ROOT = MNIST_ROOT.parent
SRC_DIR = MNIST_ROOT / "src"

sys.path.append(str(SRC_DIR))

import models
import train_helper


def main():
    seed = 0

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # 完整 MNIST training set，共 60,000 张图片
    full_dataset = datasets.MNIST(
        root=PROJECT_ROOT / "data",
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    full_loader = DataLoader(
        full_dataset,
        batch_size=128,
        shuffle=True
    )

    # 必须与 full_pipeline.py 中加载 checkpoint 时的结构一致
    full_real_model = models.CVAE(
        input_dim=784,
        label_dim=10,
        latent_dim=20,
        name="full_dataset_model",
        arch="conv"
    ).to(device)

    history = train_helper.train_model(
        model=full_real_model,
        train_loader=full_loader,
        device=device,
        epochs=200,
        lr=1e-3,
        patience=5,
        verbose=True
    )

    checkpoint_dir = (
        MNIST_ROOT
        / "conv_cvae"
        / "model_saved_full_dataset"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint_path = (
        checkpoint_dir
        / "full_dataset_model.pth"
    )

    # 与 full_pipeline.py 的 load_state_dict() 对应
    torch.save(
        full_real_model.state_dict(),
        checkpoint_path
    )

    print(f"\nTraining finished after {history['epochs_trained']} epochs.")
    print(f"Best training loss: {history['best_loss']:.4f}")
    print(f"Checkpoint saved to:\n{checkpoint_path}")


if __name__ == "__main__":
    main()