"""Training pipeline: PyTorch Lightning module + audio dataset.

Self-supervised: each track yields two random augmented crops (positives).
Supervised heads: regression targets come from the heuristic extractor —
the model distills + smooths heuristics into a learned space, and can later
be fine-tuned on listening-behavior pairs (same-session tracks as positives).

Run:
    python -m app.training.train --data-dir /data/audio --features-csv features.csv
Distributed:
    torchrun --nproc_per_node=4 -m app.training.train ... (Lightning DDP handles it)
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from app.models.encoder import REGRESSION_TARGETS, SongEncoder, nt_xent_loss

SR = 22050
CROP_SECONDS = 10.0
N_MELS = 96


# ── augmentations ─────────────────────────────────────────────────────────────

def random_crop(y: np.ndarray, seconds: float = CROP_SECONDS) -> np.ndarray:
    n = int(seconds * SR)
    if len(y) <= n:
        return np.pad(y, (0, n - len(y)))
    start = random.randint(0, len(y) - n)
    return y[start : start + n]


def augment(y: np.ndarray) -> np.ndarray:
    # gain
    y = y * random.uniform(0.7, 1.3)
    # additive noise
    if random.random() < 0.5:
        y = y + np.random.randn(len(y)).astype(np.float32) * 0.005
    # time stretch (cheap: resample-based)
    if random.random() < 0.3:
        rate = random.uniform(0.9, 1.1)
        y = librosa.effects.time_stretch(y, rate=rate)
    # pitch shift
    if random.random() < 0.3:
        y = librosa.effects.pitch_shift(y, sr=SR, n_steps=random.uniform(-2, 2))
    return y


def to_mel(y: np.ndarray) -> torch.Tensor:
    n = int(CROP_SECONDS * SR)
    y = y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, hop_length=512)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db + 80) / 80  # normalize to ~[0, 1]
    return torch.from_numpy(mel_db).float().unsqueeze(0)  # [1, 96, T]


def spec_augment(mel: torch.Tensor, n_masks: int = 2) -> torch.Tensor:
    mel = mel.clone()
    _, n_freq, n_time = mel.shape
    for _ in range(n_masks):
        f0 = random.randint(0, n_freq - 12)
        mel[:, f0 : f0 + random.randint(4, 12), :] = 0
        t0 = random.randint(0, max(n_time - 30, 1))
        mel[:, :, t0 : t0 + random.randint(8, 30)] = 0
    return mel


# ── dataset ───────────────────────────────────────────────────────────────────

class ContrastiveSongDataset(Dataset):
    """Expects:
    - audio files under data_dir
    - features_csv with columns: path, danceability, energy, ... (heuristic targets)
    """

    def __init__(self, data_dir: str, features_csv: str):
        self.df = pd.read_csv(features_csv)
        self.data_dir = Path(data_dir)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        y, _ = librosa.load(self.data_dir / row["path"], sr=SR, mono=True)

        view1 = spec_augment(to_mel(augment(random_crop(y))))
        view2 = spec_augment(to_mel(augment(random_crop(y))))

        targets = torch.tensor(
            [row[t] for t in REGRESSION_TARGETS], dtype=torch.float32
        )
        return view1, view2, targets


# ── lightning module ──────────────────────────────────────────────────────────

class EmbeddingTrainer(pl.LightningModule):
    def __init__(
        self,
        lr: float = 3e-4,
        temperature: float = 0.1,
        w_contrastive: float = 1.0,
        w_features: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = SongEncoder()

    def training_step(self, batch, batch_idx):
        v1, v2, targets = batch
        out1 = self.model(v1)
        out2 = self.model(v2)

        loss_c = nt_xent_loss(out1["embedding"], out2["embedding"], self.hparams.temperature)
        loss_f = 0.5 * (
            F.mse_loss(out1["features"], targets) + F.mse_loss(out2["features"], targets)
        )
        loss = self.hparams.w_contrastive * loss_c + self.hparams.w_features * loss_f

        self.log_dict(
            {"train/loss": loss, "train/contrastive": loss_c, "train/features": loss_f},
            prog_bar=True, sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        v1, v2, targets = batch
        out1 = self.model(v1)
        out2 = self.model(v2)
        loss_c = nt_xent_loss(out1["embedding"], out2["embedding"], self.hparams.temperature)
        loss_f = F.mse_loss(out1["features"], targets)
        self.log_dict({"val/contrastive": loss_c, "val/features": loss_f}, sync_dist=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
        return {"optimizer": opt, "lr_scheduler": sched}


# ── entrypoint ────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--features-csv", required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    args = p.parse_args()

    ds = ContrastiveSongDataset(args.data_dir, args.features_csv)
    n_val = max(int(0.05 * len(ds)), 1)
    train_ds, val_ds = torch.utils.data.random_split(ds, [len(ds) - n_val, n_val])

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.workers)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=args.devices,
        strategy="ddp" if args.devices > 1 else "auto",
        precision="16-mixed",
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=args.checkpoint_dir, monitor="val/contrastive",
                save_top_k=3, filename="encoder-{epoch}-{val/contrastive:.3f}",
            ),
            pl.callbacks.LearningRateMonitor(),
        ],
        log_every_n_steps=20,
    )
    trainer.fit(EmbeddingTrainer(), train_dl, val_dl)


if __name__ == "__main__":
    main()
