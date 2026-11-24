from __future__ import annotations

import torch
from torch import nn

from adaptive_kt.models.sequence import CAUSAL_BEHAVIOR_DIM


class BehaviorAwareKT(nn.Module):
    """Behavior-enriched KT model with target-safe learner-state conditioning.

    Historical interaction representations include the observed response plus behavior that was
    available at/after that historical interaction.  For the current target, only the first
    ``CAUSAL_BEHAVIOR_DIM`` profile features are used; these are computed strictly from prior
    history.  Current response time, attempt count, and correctness are never inputs to their own
    prediction.
    """

    def __init__(
        self,
        num_skills: int,
        num_items: int,
        max_len: int,
        behavior_dim: int,
        embedding_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        ablate_behavior_indices: tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        self.ablate_behavior_indices = tuple(sorted(set(int(i) for i in ablate_behavior_indices)))
        self.skill_embedding = nn.Embedding(num_skills, embedding_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.response_embedding = nn.Embedding(2, embedding_dim)
        self.position_embedding = nn.Embedding(max_len + 1, embedding_dim)
        self.behavior_projection = nn.Sequential(
            nn.Linear(behavior_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.target_profile_projection = nn.Sequential(
            nn.Linear(CAUSAL_BEHAVIOR_DIM, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal_projection = nn.Sequential(
            nn.Linear(2, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        # Learn how to combine knowledge, item, response, behavior and time
        # instead of simply adding all signals together.
        self.history_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 7, embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        self.target_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 5, embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
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
        self.history_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cross_attention = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True
        )
        # Behavior directly participates in deciding how much historical
        # knowledge versus current target state should be used.
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(embedding_dim)

        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim * 4, embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, 1),
        )

        # Learn question-specific difficulty/intercept.
        self.item_bias = nn.Embedding(num_items, 1, padding_idx=0)
        nn.init.zeros_(self.item_bias.weight)

    def _apply_ablation(self, behavior: torch.Tensor) -> torch.Tensor:
        if not self.ablate_behavior_indices:
            return behavior
        behavior = behavior.clone()
        valid = [idx for idx in self.ablate_behavior_indices if 0 <= idx < behavior.size(-1)]
        if valid:
            behavior[..., valid] = 0.0
        return behavior

    def forward(self, skills, items, correct, behavior=None, elapsed=None, lag=None, **_) -> torch.Tensor:
        batch_size, seq_len = items.shape
        positions = torch.arange(seq_len, device=items.device).unsqueeze(0).expand(batch_size, -1)
        if behavior is None:
            raise ValueError("BehaviorAwareKT requires causal behavior features")
        if elapsed is None:
            elapsed = torch.zeros_like(items, dtype=torch.float32)
        if lag is None:
            lag = torch.zeros_like(items, dtype=torch.float32)

        behavior = self._apply_ablation(behavior)

        history_temporal = torch.stack([elapsed[:, :-1], lag[:, :-1]], dim=-1)

        h_skill = self.skill_embedding(skills[:, :-1])
        h_item = self.item_embedding(items[:, :-1])
        h_response = self.response_embedding(correct[:, :-1].clamp(0, 1))
        h_position = self.position_embedding(positions[:, :-1])
        h_behavior = self.behavior_projection(behavior[:, :-1])
        h_temporal = self.temporal_projection(history_temporal)
        h_interaction = h_skill * h_item

        history = self.history_fusion(torch.cat([
            h_skill,
            h_item,
            h_response,
            h_position,
            h_behavior,
            h_temporal,
            h_interaction,
        ], dim=-1))
        history = torch.where(items[:, :-1].unsqueeze(-1).eq(0), torch.zeros_like(history), history)

        # Target profile is safe because these dimensions are computed before observing the
        # target response.  The current elapsed time/attempt count live in later behavior
        # dimensions and are intentionally excluded here.
        target_profile = behavior[:, 1:, :CAUSAL_BEHAVIOR_DIM]
        target_profile_emb = self.target_profile_projection(target_profile)

        t_skill = self.skill_embedding(skills[:, 1:])
        t_item = self.item_embedding(items[:, 1:])
        t_position = self.position_embedding(positions[:, 1:])
        t_interaction = t_skill * t_item

        target = self.target_fusion(torch.cat([
            t_skill,
            t_item,
            t_position,
            target_profile_emb,
            t_interaction,
        ], dim=-1))
        target = torch.where(items[:, 1:].unsqueeze(-1).eq(0), torch.zeros_like(target), target)

        target_len = seq_len - 1
        causal = torch.triu(
            torch.ones((target_len, target_len), device=items.device, dtype=torch.bool), diagonal=1
        )
        encoded_history = self.history_encoder(
            history,
            mask=causal,
            src_key_padding_mask=items[:, :-1].eq(0),
        )
        context, _ = self.cross_attention(
            query=target,
            key=encoded_history,
            value=encoded_history,
            attn_mask=causal,
            key_padding_mask=items[:, :-1].eq(0),
            need_weights=False,
        )
        gate = self.gate(torch.cat(
            [target, context, target_profile_emb], dim=-1
        ))
        fused = self.norm(gate * context + (1.0 - gate) * target)

        logits = self.predictor(torch.cat([
            fused,
            target,
            context,
            target_profile_emb,
        ], dim=-1)).squeeze(-1)

        return logits + self.item_bias(items[:, 1:]).squeeze(-1)
