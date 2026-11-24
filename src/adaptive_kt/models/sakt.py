from __future__ import annotations

import torch
from torch import nn


class SAKT(nn.Module):
    """Self-Attentive Knowledge Tracing with item-aware queries."""

    def __init__(
        self,
        num_skills: int,
        num_items: int,
        max_len: int,
        embedding_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        self.skill_embedding = nn.Embedding(num_skills, embedding_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.response_embedding = nn.Embedding(2, embedding_dim)
        self.position_embedding = nn.Embedding(max_len + 1, embedding_dim)
        self.attention = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim),
        )
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, skills, items, correct, **_) -> torch.Tensor:
        seq_len = items.size(1)
        positions = torch.arange(seq_len, device=items.device).unsqueeze(0)

        history = (
            self.skill_embedding(skills[:, :-1])
            + self.item_embedding(items[:, :-1])
            + self.response_embedding(correct[:, :-1].clamp(0, 1))
            + self.position_embedding(positions[:, :-1])
        )
        query = (
            self.skill_embedding(skills[:, 1:])
            + self.item_embedding(items[:, 1:])
            + self.position_embedding(positions[:, 1:])
        )
        history = torch.where(items[:, :-1].unsqueeze(-1).eq(0), torch.zeros_like(history), history)
        query = torch.where(items[:, 1:].unsqueeze(-1).eq(0), torch.zeros_like(query), query)

        target_len = query.size(1)
        causal_mask = torch.triu(
            torch.ones((target_len, target_len), device=items.device, dtype=torch.bool), diagonal=1
        )
        key_padding = items[:, :-1].eq(0)
        attended, _ = self.attention(
            query=query,
            key=history,
            value=history,
            attn_mask=causal_mask,
            key_padding_mask=key_padding,
            need_weights=False,
        )
        x = self.norm1(query + self.dropout(attended))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return self.predictor(torch.cat([x, query], dim=-1)).squeeze(-1)
