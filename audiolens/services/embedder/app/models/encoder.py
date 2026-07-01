"""Song embedding model.

Architecture:
    mel-spectrogram (96 x T)
      -> CNN frontend (4 conv blocks)
      -> temporal Transformer encoder (4 layers)
      -> attention pooling
      -> projection head  -> 128-dim L2-normalized embedding

Training objectives (multi-task):
  1. Contrastive (NT-Xent / SimCLR): two augmented crops of the same track
     are positives; all other tracks in batch are negatives.
  2. Feature regression heads: predict danceability/energy/valence/etc from
     the embedding -> forces the space to encode Spotify-like semantics.
  3. (optional) Genre classification head when genre labels available.

Result: an embedding usable for similarity search, mood clustering,
playlist generation, and as a feature predictor for new tracks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_MELS = 96
EMBED_DIM = 128

REGRESSION_TARGETS = [
    "danceability", "energy", "valence", "speechiness",
    "acousticness", "instrumentalness", "liveness",
]


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.GELU(),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.GELU(),
            nn.MaxPool2d((2, 2)),
        )

    def forward(self, x):
        return self.net(x)


class AttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x):  # x: [B, T, D]
        w = torch.softmax(self.attn(x), dim=1)  # [B, T, 1]
        return (w * x).sum(dim=1)  # [B, D]


class SongEncoder(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM, n_genres: int = 0):
        super().__init__()
        self.frontend = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        # after 4 pools: mel 96 -> 6, so feature dim = 256 * 6
        self.to_seq = nn.Linear(256 * 6, 512)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=512, nhead=8, dim_feedforward=1024,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=4)
        self.pool = AttentionPool(512)

        self.projection = nn.Sequential(
            nn.Linear(512, 256), nn.GELU(), nn.Linear(256, embed_dim),
        )

        # multi-task heads operate on the pooled representation (not projection)
        self.feature_head = nn.Sequential(
            nn.Linear(512, 128), nn.GELU(),
            nn.Linear(128, len(REGRESSION_TARGETS)), nn.Sigmoid(),
        )
        self.genre_head = (
            nn.Linear(512, n_genres) if n_genres > 0 else None
        )

    def encode(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: [B, 1, 96, T] -> pooled representation [B, 512]"""
        h = self.frontend(mel)                    # [B, 256, 6, T']
        b, c, f, t = h.shape
        h = h.permute(0, 3, 1, 2).reshape(b, t, c * f)  # [B, T', 256*6]
        h = self.to_seq(h)                        # [B, T', 512]
        h = self.transformer(h)
        return self.pool(h)                       # [B, 512]

    def forward(self, mel: torch.Tensor) -> dict[str, torch.Tensor]:
        rep = self.encode(mel)
        z = F.normalize(self.projection(rep), dim=-1)   # [B, 128]
        out = {"embedding": z, "features": self.feature_head(rep)}
        if self.genre_head is not None:
            out["genre_logits"] = self.genre_head(rep)
        return out


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """SimCLR contrastive loss for paired views z1[i] <-> z2[i]."""
    b = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)                       # [2B, D]
    sim = z @ z.T / temperature                          # [2B, 2B]
    sim.fill_diagonal_(-1e9)

    targets = torch.cat([
        torch.arange(b, 2 * b), torch.arange(0, b)
    ]).to(z.device)
    return F.cross_entropy(sim, targets)
