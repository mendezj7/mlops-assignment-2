import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pil_augment(img: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.5:
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    angle = rng.uniform(-15, 15)
    img = img.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False)

    brightness = ImageEnhance.Brightness(img)
    img = brightness.enhance(rng.uniform(0.8, 1.2))

    contrast = ImageEnhance.Contrast(img)
    img = contrast.enhance(rng.uniform(0.8, 1.2))

    return img


class ImageFolderDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        img_size: int,
        augment: bool,
        seed: int,
        sample_frac: float = 1.0,
        max_per_class: int | None = None,
        max_total: int | None = None,
    ):
        self.root = Path(root) / split
        self.img_size = img_size
        self.augment = augment
        self.rng = random.Random(seed)

        if not self.root.exists():
            raise FileNotFoundError(f"Missing split folder: {self.root}")

        self.classes = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        if not self.classes:
            raise ValueError(f"No class folders found in {self.root}")
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = []
        for cls in self.classes:
            cls_dir = self.root / cls
            cls_samples = []
            for p in cls_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    cls_samples.append((p, self.class_to_idx[cls]))

            if not cls_samples:
                continue

            if sample_frac < 1.0:
                keep_n = max(1, int(len(cls_samples) * sample_frac))
                cls_samples = self.rng.sample(cls_samples, keep_n)

            if max_per_class is not None:
                cls_samples = cls_samples[: max_per_class]

            self.samples.extend(cls_samples)

        if not self.samples:
            raise ValueError(f"No images found under {self.root}")

        if max_total is not None and max_total > 0:
            if max_total < len(self.samples):
                self.samples = self.rng.sample(self.samples, max_total)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")

        if self.augment:
            img = pil_augment(img, self.rng)

        if img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)

        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        x = torch.from_numpy(arr)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


def accuracy_from_logits(logits, labels):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    return correct / len(labels)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        running_loss += loss.item() * batch_size
        running_acc += accuracy_from_logits(logits, y) * batch_size
        total += batch_size

    return running_loss / total, running_acc / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            batch_size = y.size(0)
            running_loss += loss.item() * batch_size
            running_acc += accuracy_from_logits(logits, y) * batch_size
            total += batch_size

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    return running_loss / total, running_acc / total, all_labels, all_preds


def plot_confusion_matrix(cm, classes, out_path: Path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_loss_curve(train_losses, val_losses, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(train_losses, label="train")
    ax.plot(val_losses, label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed_224")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--experiment", default="cats-dogs-baseline")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sample-frac", type=float, default=1.0)
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--max-total", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    train_ds = ImageFolderDataset(
        args.data_dir,
        "train",
        args.img_size,
        args.augment,
        args.seed,
        sample_frac=args.sample_frac,
        max_per_class=args.max_per_class,
        max_total=args.max_total,
    )
    val_ds = ImageFolderDataset(
        args.data_dir,
        "val",
        args.img_size,
        False,
        args.seed,
        sample_frac=args.sample_frac,
        max_per_class=args.max_per_class,
        max_total=args.max_total,
    )
    test_ds = ImageFolderDataset(
        args.data_dir,
        "test",
        args.img_size,
        False,
        args.seed,
        sample_frac=args.sample_frac,
        max_per_class=args.max_per_class,
        max_total=args.max_total,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SimpleCNN(num_classes=len(train_ds.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(args.experiment)

    train_losses = []
    val_losses = []

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params({
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "img_size": args.img_size,
            "augment": args.augment,
            "sample_frac": args.sample_frac,
            "max_per_class": args.max_per_class,
            "max_total": args.max_total,
            "train_size": len(train_ds),
            "val_size": len(val_ds),
            "test_size": len(test_ds),
            "device": str(device),
        })

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_acc", train_acc, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)

        test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_acc", test_acc)

        cm = confusion_matrix(y_true, y_pred)
        cm_path = plots_dir / "confusion_matrix.png"
        plot_confusion_matrix(cm, train_ds.classes, cm_path)
        mlflow.log_artifact(str(cm_path))

        loss_path = plots_dir / "loss_curve.png"
        plot_loss_curve(train_losses, val_losses, loss_path)
        mlflow.log_artifact(str(loss_path))

        model_path = models_dir / "baseline_cnn.pt"
        torch.save({
            "model_state": model.state_dict(),
            "class_to_idx": train_ds.class_to_idx,
            "img_size": args.img_size,
        }, model_path)
        mlflow.log_artifact(str(model_path))

        meta_path = out_dir / "run_metadata.json"
        meta = {
            "classes": train_ds.classes,
            "class_to_idx": train_ds.class_to_idx,
            "train_size": len(train_ds),
            "val_size": len(val_ds),
            "test_size": len(test_ds),
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        mlflow.log_artifact(str(meta_path))

        print(f"Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")
        print(f"Artifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
