from __future__ import annotations

import torch
from torch import nn


class TransformerKT(nn.Module):

    def __init__(
        self,
        num_skills: int,
        num_items: int,
        max_len: int,
        embedding_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads."
            )

        self.skill_embedding = nn.Embedding(
            num_skills,
            embedding_dim,
            padding_idx=0,
        )

        self.item_embedding = nn.Embedding(
            num_items,
            embedding_dim,
            padding_idx=0,
        )

        self.response_embedding = nn.Embedding(
            2,
            embedding_dim,
        )

        self.position_embedding = nn.Embedding(
            max_len + 1,
            embedding_dim,
        )

        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(
            embedding_dim
        )

        self.predictor = nn.Sequential(
            nn.Linear(
                embedding_dim * 2,
                embedding_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                embedding_dim,
                1,
            ),
        )

    def forward(
        self,
        skills,
        items,
        correct,
        **_,
    ):

        batch_size, seq_len = items.shape

        positions = torch.arange(
            seq_len,
            device=items.device,
        ).unsqueeze(0).expand(
            batch_size,
            -1,
        )

        x = (
            self.skill_embedding(skills)
            + self.item_embedding(items)
            + self.response_embedding(
                correct.clamp(0, 1)
            )
            + self.position_embedding(positions)
        )

        x = torch.where(
            items.unsqueeze(-1).eq(0),
            torch.zeros_like(x),
            x,
        )

        causal_mask = torch.triu(
            torch.ones(
                (seq_len, seq_len),
                device=items.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        padding_mask = items.eq(0)

        encoded = self.encoder(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )

        encoded = self.norm(encoded)

        target = (
            self.skill_embedding(skills[:, 1:])
            + self.item_embedding(items[:, 1:])
        )

        features = torch.cat(
            [
                encoded[:, :-1],
                target,
            ],
            dim=-1,
        )

        return self.predictor(features).squeeze(-1)
