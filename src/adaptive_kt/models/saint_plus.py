from __future__ import annotations

import torch
from torch import nn


class SAINTPlus(nn.Module):
    """SAINT+-style encoder/decoder KT with causal time alignment."""

    def __init__(
        self,
        num_skills: int,
        num_items: int,
        max_len: int,
        embedding_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.skill_embedding = nn.Embedding(num_skills, embedding_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.response_embedding = nn.Embedding(2, embedding_dim)
        self.position_embedding = nn.Embedding(max_len + 1, embedding_dim)
        self.temporal_projection = nn.Sequential(
            nn.Linear(2, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embedding_dim)
        self.output = nn.Linear(embedding_dim, 1)

    def forward(self, skills, items, correct, elapsed=None, lag=None, **_) -> torch.Tensor:
        batch_size, seq_len = items.shape
        positions = torch.arange(seq_len, device=items.device).unsqueeze(0).expand(batch_size, -1)

        if elapsed is None:
            elapsed = torch.zeros_like(items, dtype=torch.float32)
        if lag is None:
            lag = torch.zeros_like(items, dtype=torch.float32)

        questions = (
            self.skill_embedding(skills)
            + self.item_embedding(items)
            + self.position_embedding(positions)
        )
        questions = torch.where(items.unsqueeze(-1).eq(0), torch.zeros_like(questions), questions)

        full_causal = torch.triu(
            torch.ones((seq_len, seq_len), device=items.device, dtype=torch.bool),
            diagonal=1,
        )
        memory = self.encoder(
            questions,
            mask=full_causal,
            src_key_padding_mask=items.eq(0),
        )

        # Only already-observed timing from the previous interaction is permitted.
        # lag[:, 1:] is anchored at the target response timestamp and is therefore not
        # guaranteed to be known when predicting the target response.
        previous_elapsed = elapsed[:, :-1]
        previous_lag = lag[:, :-1]
        temporal = torch.stack([previous_elapsed, previous_lag], dim=-1)

        decoder_input = (
            self.response_embedding(correct[:, :-1].clamp(0, 1))
            + self.temporal_projection(temporal)
            + self.position_embedding(positions[:, 1:])
        )
        decoder_input = torch.where(
            items[:, :-1].unsqueeze(-1).eq(0),
            torch.zeros_like(decoder_input),
            decoder_input,
        )

        target_len = seq_len - 1
        tgt_mask = torch.triu(
            torch.ones((target_len, target_len), device=items.device, dtype=torch.bool),
            diagonal=1,
        )

        q_index = torch.arange(target_len, device=items.device).unsqueeze(1) + 1
        m_index = torch.arange(seq_len, device=items.device).unsqueeze(0)
        memory_mask = m_index > q_index

        decoded = self.decoder(
            tgt=decoder_input,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=items[:, :-1].eq(0),
            memory_key_padding_mask=items.eq(0),
        )
        return self.output(self.norm(decoded)).squeeze(-1)
