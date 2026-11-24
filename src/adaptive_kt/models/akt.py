from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class ContextAwareMonotonicAttention(nn.Module):
    """AKT-style context-aware monotonic multi-head attention."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float,
        share_qk: bool = True,
    ) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.share_qk = bool(share_qk)

        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.q_proj = None if self.share_qk else nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

        self.gamma = nn.Parameter(torch.empty(num_heads, 1, 1))
        nn.init.xavier_uniform_(self.gamma)
        self.dropout = nn.Dropout(dropout)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        allow_current: bool,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, tq, _ = query.shape
        tk = key.size(1)

        q_linear = self.k_proj(query) if self.share_qk else self.q_proj(query)
        q = self._split(q_linear)
        k = self._split(self.k_proj(key))
        v = self._split(self.v_proj(value))

        raw_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        q_pos = torch.arange(tq, device=query.device).view(tq, 1)
        k_pos = torch.arange(tk, device=query.device).view(1, tk)
        causal_allowed = (k_pos <= q_pos) if allow_current else (k_pos < q_pos)

        allowed = causal_allowed.view(1, 1, tq, tk).expand(
            b, self.num_heads, -1, -1
        )
        if key_padding_mask is not None:
            allowed = allowed & (~key_padding_mask[:, None, None, :])

        with torch.no_grad():
            preliminary = raw_scores.masked_fill(~allowed, -1e9)
            preliminary = torch.softmax(preliminary, dim=-1) * allowed.float()
            cumulative = torch.cumsum(preliminary, dim=-1)
            total = preliminary.sum(dim=-1, keepdim=True)
            position_effect = (q_pos - k_pos).abs().float().view(1, 1, tq, tk)
            distance = torch.clamp(
                (total - cumulative) * position_effect,
                min=0.0,
            )
            distance = torch.sqrt(distance).detach()

        decay_rate = -F.softplus(self.gamma).unsqueeze(0)
        total_effect = torch.exp(decay_rate * distance).clamp(min=1e-5, max=1e5)

        scores = (raw_scores * total_effect).masked_fill(~allowed, -1e9)
        weights = torch.softmax(scores, dim=-1) * allowed.float()
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        weights = self.dropout(weights / denom)

        context = torch.matmul(weights, v)
        context = context.transpose(1, 2).contiguous().view(
            b, tq, self.embedding_dim
        )
        return self.out_proj(context)


class AKTBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float,
        *,
        apply_ffn: bool,
    ) -> None:
        super().__init__()
        self.attention = ContextAwareMonotonicAttention(
            embedding_dim,
            num_heads,
            dropout,
            share_qk=True,
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.apply_ffn = bool(apply_ffn)

        if self.apply_ffn:
            self.ffn = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim * 4, embedding_dim),
            )
            self.norm2 = nn.LayerNorm(embedding_dim)
            self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        query,
        key,
        value,
        *,
        allow_current: bool,
        key_padding_mask=None,
    ):
        attended = self.attention(
            query,
            key,
            value,
            allow_current=allow_current,
            key_padding_mask=key_padding_mask,
        )
        x = self.norm1(query + self.dropout1(attended))

        if self.apply_ffn:
            x = self.norm2(x + self.dropout2(self.ffn(x)))

        return x


class AKT(nn.Module):
    """Closer-to-reference AKT baseline with Rasch-style item difficulty."""

    def __init__(
        self,
        num_skills: int,
        num_items: int,
        embedding_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.skill_embedding = nn.Embedding(num_skills, embedding_dim, padding_idx=0)
        self.skill_variation = nn.Embedding(num_skills, embedding_dim, padding_idx=0)
        self.response_embedding = nn.Embedding(2, embedding_dim)
        self.response_variation = nn.Embedding(2, embedding_dim)

        self.item_difficulty = nn.Embedding(num_items, 1, padding_idx=0)
        nn.init.zeros_(self.item_difficulty.weight)

        self.response_blocks = nn.ModuleList(
            [
                AKTBlock(
                    embedding_dim,
                    num_heads,
                    dropout,
                    apply_ffn=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.question_blocks = nn.ModuleList(
            [
                AKTBlock(
                    embedding_dim,
                    num_heads,
                    dropout,
                    apply_ffn=False,
                )
                for _ in range(num_layers)
            ]
        )
        self.retrieval_blocks = nn.ModuleList(
            [
                AKTBlock(
                    embedding_dim,
                    num_heads,
                    dropout,
                    apply_ffn=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.output = nn.Sequential(
            nn.Linear(embedding_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def _question_embedding(self, skills, items):
        difficulty = self.item_difficulty(items)
        base = self.skill_embedding(skills)
        variation = self.skill_variation(skills)
        question = base + difficulty * variation
        return question, difficulty, variation

    def forward(self, skills, items, correct, **_) -> torch.Tensor:
        question, difficulty, q_variation = self._question_embedding(skills, items)
        responses = correct.clamp(0, 1)

        interaction = (
            question
            + self.response_embedding(responses)
            + difficulty
            * (
                self.response_variation(responses)
                + q_variation
            )
        )

        padding = items.eq(0)
        question = torch.where(
            padding.unsqueeze(-1),
            torch.zeros_like(question),
            question,
        )
        interaction = torch.where(
            padding.unsqueeze(-1),
            torch.zeros_like(interaction),
            interaction,
        )

        y = interaction
        for block in self.response_blocks:
            y = block(
                y,
                y,
                y,
                allow_current=True,
                key_padding_mask=padding,
            )

        x = question
        for q_block, r_block in zip(
            self.question_blocks,
            self.retrieval_blocks,
        ):
            x = q_block(
                x,
                x,
                x,
                allow_current=True,
                key_padding_mask=padding,
            )
            x = r_block(
                x,
                x,
                y,
                allow_current=False,
                key_padding_mask=padding,
            )

        logits = self.output(
            torch.cat([x, question], dim=-1)
        ).squeeze(-1)

        return logits[:, 1:]
