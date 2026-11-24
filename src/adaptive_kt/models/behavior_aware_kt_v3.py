"""Proposed-V3: V2 + difficulty-adjusted mastery (Elo block in the context) + part/tag embeddings.

Same architecture and sizes as V2. Differences:
  1. Context vector has 29 values (20 from V2 + 9 difficulty-adjusted mastery values).
  2. The question's TOEIC part and its single knowledge tags are embedded (static metadata).
  3. Single profile features can be switched off for the ablations named in the proposal
     (mastery score, response time, recent accuracy).
Target correctness, response time and answer changes are never inputs to their own prediction.
"""
from __future__ import annotations

import torch
from torch import nn

from adaptive_kt.models.behavior_aware_kt_v2 import HISTORY_KEEP, PROFILE_KEEP
from adaptive_kt.models.sequence import BEHAVIOR_DIM
from adaptive_kt.models.sequence_v3 import CTX_DIM_V3, N_PARTS


def _projection(in_dim: int, dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout))


def _fusion(in_dim: int, dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, dim * 2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim * 2, dim),
        nn.LayerNorm(dim),
    )


class BehaviorAwareKTV3(nn.Module):
    def __init__(
        self,
        num_skills: int,
        num_items: int,
        num_sources: int,
        num_platforms: int,
        num_tags: int,
        item_part: torch.Tensor,
        item_tags: torch.Tensor,
        max_len: int,
        embedding_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        ablate_ctx_indices: tuple[int, ...] = (),
        ablate_behavior_indices: tuple[int, ...] = (),
        use_source: bool = True,
        use_profile: bool = True,
        use_meta: bool = True,
        use_elapsed: bool = True,
    ) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if item_part.shape[0] != num_items or item_tags.shape[0] != num_items:
            raise ValueError("item_part/item_tags must have one row per item vocabulary code")
        dim = embedding_dim
        self.use_source = bool(use_source)
        self.use_profile = bool(use_profile)
        self.use_meta = bool(use_meta)
        self.use_elapsed = bool(use_elapsed)

        ctx_mask = torch.ones(CTX_DIM_V3)
        for index in ablate_ctx_indices:
            ctx_mask[int(index)] = 0.0
        behavior_mask = torch.ones(BEHAVIOR_DIM)
        for index in ablate_behavior_indices:
            behavior_mask[int(index)] = 0.0
        self.register_buffer("ctx_mask", ctx_mask, persistent=False)
        self.register_buffer("behavior_mask", behavior_mask, persistent=False)
        self.register_buffer("profile_keep", torch.tensor(PROFILE_KEEP, dtype=torch.long), persistent=False)
        self.register_buffer("history_keep", torch.tensor(HISTORY_KEEP, dtype=torch.long), persistent=False)
        self.register_buffer("item_part", item_part.long().clone(), persistent=False)
        self.register_buffer("item_tags", item_tags.long().clone(), persistent=False)

        self.skill_embedding = nn.Embedding(num_skills, dim, padding_idx=0)
        self.item_embedding = nn.Embedding(num_items, dim, padding_idx=0)
        self.response_embedding = nn.Embedding(2, dim)
        self.position_embedding = nn.Embedding(max_len + 1, dim)
        self.source_embedding = nn.Embedding(max(int(num_sources), 2), dim, padding_idx=0)
        self.platform_embedding = nn.Embedding(max(int(num_platforms), 2), dim, padding_idx=0)
        self.part_embedding = nn.Embedding(N_PARTS, dim, padding_idx=0)
        self.tag_embedding = nn.Embedding(int(num_tags) + 1, dim, padding_idx=0)

        self.behavior_projection = _projection(len(HISTORY_KEEP), dim, dropout)
        self.target_profile_projection = _projection(len(PROFILE_KEEP), dim, dropout)
        self.ctx_projection = _projection(CTX_DIM_V3, dim, dropout)
        self.temporal_projection = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))

        self.history_fusion = _fusion(dim * 9, dim, dropout)
        self.target_fusion = _fusion(dim * 7, dim, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.history_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cross_attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(dim * 3, dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(dim)
        self.predictor = nn.Sequential(
            nn.Linear(dim * 5, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )
        self.wide = nn.Linear(len(PROFILE_KEEP) + CTX_DIM_V3, 1)
        nn.init.zeros_(self.wide.weight)
        nn.init.zeros_(self.wide.bias)
        self.item_bias = nn.Embedding(num_items, 1, padding_idx=0)
        nn.init.zeros_(self.item_bias.weight)

    def _meta(self, items: torch.Tensor, source: torch.Tensor, platform: torch.Tensor) -> torch.Tensor:
        meta = self.source_embedding(source) + self.platform_embedding(platform)
        if self.use_meta:
            tags = self.item_tags[items]
            tag_sum = self.tag_embedding(tags).sum(dim=2)
            tag_count = tags.ne(0).sum(dim=2, keepdim=True).clamp(min=1)
            meta = meta + self.part_embedding(self.item_part[items]) + tag_sum / tag_count
        return meta

    def forward(
        self,
        skills,
        items,
        correct,
        behavior=None,
        elapsed=None,
        lag=None,
        ctx=None,
        source=None,
        platform=None,
        **_,
    ) -> torch.Tensor:
        if behavior is None or ctx is None:
            raise ValueError("BehaviorAwareKTV3 needs behavior and ctx tensors (use collate_sequences_v3)")
        if behavior.size(-1) != BEHAVIOR_DIM or ctx.size(-1) != CTX_DIM_V3:
            raise ValueError(f"bad feature dims: behavior={tuple(behavior.shape)} ctx={tuple(ctx.shape)}")
        batch_size, seq_len = items.shape
        positions = torch.arange(seq_len, device=items.device).unsqueeze(0).expand(batch_size, -1)
        if elapsed is None or not self.use_elapsed:
            elapsed = torch.zeros_like(items, dtype=torch.float32)
        if lag is None:
            lag = torch.zeros_like(items, dtype=torch.float32)
        if source is None or not self.use_source:
            source = torch.zeros_like(items)
        if platform is None or not self.use_source:
            platform = torch.zeros_like(items)
        if not self.use_profile:
            behavior = torch.zeros_like(behavior)

        ctx = ctx * self.ctx_mask
        behavior = behavior * self.behavior_mask
        profile = behavior.index_select(2, self.profile_keep)
        history_behavior = behavior.index_select(2, self.history_keep)
        meta = self._meta(items, source, platform)
        ctx_emb = self.ctx_projection(ctx)

        # History token t: interaction t with its observed response (t = 0 .. L-2).
        h_skill = self.skill_embedding(skills[:, :-1])
        h_item = self.item_embedding(items[:, :-1])
        history = self.history_fusion(torch.cat([
            h_skill,
            h_item,
            self.response_embedding(correct[:, :-1].clamp(0, 1)),
            self.position_embedding(positions[:, :-1]),
            self.behavior_projection(history_behavior[:, :-1]),
            self.temporal_projection(torch.stack([elapsed[:, :-1], lag[:, :-1]], dim=-1)),
            h_skill * h_item,
            ctx_emb[:, :-1],
            meta[:, :-1],
        ], dim=-1))
        history_pad = items[:, :-1].eq(0)
        history = history.masked_fill(history_pad.unsqueeze(-1), 0.0)

        # Target query t+1: only information available before the learner answers.
        target_profile = profile[:, 1:]
        target_ctx = ctx[:, 1:]
        target_profile_emb = self.target_profile_projection(target_profile)
        target_ctx_emb = ctx_emb[:, 1:]
        t_skill = self.skill_embedding(skills[:, 1:])
        t_item = self.item_embedding(items[:, 1:])
        target = self.target_fusion(torch.cat([
            t_skill,
            t_item,
            self.position_embedding(positions[:, 1:]),
            target_profile_emb,
            t_skill * t_item,
            target_ctx_emb,
            meta[:, 1:],
        ], dim=-1))
        target = target.masked_fill(items[:, 1:].eq(0).unsqueeze(-1), 0.0)

        target_len = seq_len - 1
        causal = torch.triu(torch.ones((target_len, target_len), device=items.device, dtype=torch.bool), diagonal=1)
        encoded = self.history_encoder(history, mask=causal, src_key_padding_mask=history_pad)
        context, _ = self.cross_attention(
            query=target,
            key=encoded,
            value=encoded,
            attn_mask=causal,
            key_padding_mask=history_pad,
            need_weights=False,
        )
        gate = self.gate(torch.cat([target, context, target_profile_emb], dim=-1))
        fused = self.norm(gate * context + (1.0 - gate) * target)
        logits = self.predictor(torch.cat([
            fused,
            target,
            context,
            target_profile_emb,
            target_ctx_emb,
        ], dim=-1)).squeeze(-1)
        wide = self.wide(torch.cat([target_profile, target_ctx], dim=-1)).squeeze(-1)
        return logits + wide + self.item_bias(items[:, 1:]).squeeze(-1)
