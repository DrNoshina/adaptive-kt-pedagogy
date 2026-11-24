from __future__ import annotations

import math

import torch
from torch import nn

from adaptive_kt.models.sequence import CAUSAL_BEHAVIOR_DIM


class LocalGraphAttention(nn.Module):
    """
    Multi-head graph attention over each question's local educational hierarchy.

    Nodes:
        part -> tags -> skill -> question

    The layer supports K heads even when embedding_dim is not divisible by K.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 6,
        head_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        projected = num_heads * head_dim

        self.q_proj = nn.Linear(embedding_dim, projected, bias=False)
        self.k_proj = nn.Linear(embedding_dim, projected, bias=False)
        self.v_proj = nn.Linear(embedding_dim, projected, bias=False)

        self.out_proj = nn.Linear(projected, embedding_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        valid_nodes: torch.Tensor,
    ) -> torch.Tensor:

        # x: [B,T,N,D]
        b, t, n, d = x.shape

        flat = x.reshape(b * t, n, d)
        valid = valid_nodes.reshape(b * t, n)

        q = self.q_proj(flat)
        k = self.k_proj(flat)
        v = self.v_proj(flat)

        q = q.view(b * t, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b * t, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b * t, n, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)

        graph_mask = adjacency.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(~graph_mask, -1e4)

        key_valid = valid[:, None, None, :]
        scores = scores.masked_fill(~key_valid, -1e4)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)

        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(b * t, n, self.num_heads * self.head_dim)
        )

        out = self.out_proj(out)

        query_valid = valid.unsqueeze(-1)
        out = out * query_valid

        out = self.norm(flat + self.dropout(out))
        out = out * query_valid

        return out.reshape(b, t, n, d)


class HierarchicalGraphEncoder(nn.Module):

    def __init__(
        self,
        num_skills: int,
        num_items: int,
        num_parts: int,
        num_tags: int,
        item_to_part: torch.Tensor,
        item_to_tags: torch.Tensor,
        embedding_dim: int = 128,
        num_heads: int = 6,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.max_tags = int(item_to_tags.shape[1])

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

        self.part_embedding = nn.Embedding(
            num_parts,
            embedding_dim,
            padding_idx=0,
        )

        self.tag_embedding = nn.Embedding(
            num_tags,
            embedding_dim,
            padding_idx=0,
        )

        self.register_buffer(
            "item_to_part",
            item_to_part.long(),
            persistent=True,
        )

        self.register_buffer(
            "item_to_tags",
            item_to_tags.long(),
            persistent=True,
        )

        node_count = self.max_tags + 3

        # node layout:
        # 0 = part
        # 1..M = tags
        # M+1 = skill
        # M+2 = question

        adj = torch.eye(node_count, dtype=torch.bool)

        part_idx = 0
        skill_idx = self.max_tags + 1
        question_idx = self.max_tags + 2

        # part <-> tags
        for tag_node in range(1, self.max_tags + 1):
            adj[part_idx, tag_node] = True
            adj[tag_node, part_idx] = True

            # tag <-> skill
            adj[tag_node, skill_idx] = True
            adj[skill_idx, tag_node] = True

            # tag <-> question
            adj[tag_node, question_idx] = True
            adj[question_idx, tag_node] = True

        # part <-> skill
        adj[part_idx, skill_idx] = True
        adj[skill_idx, part_idx] = True

        # skill <-> question
        adj[skill_idx, question_idx] = True
        adj[question_idx, skill_idx] = True

        self.register_buffer("adjacency", adj, persistent=True)

        self.layers = nn.ModuleList([
            LocalGraphAttention(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                head_dim=32,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        skills: torch.Tensor,
        items: torch.Tensor,
    ) -> torch.Tensor:

        safe_items = items.clamp(
            min=0,
            max=self.item_to_part.shape[0] - 1,
        )

        part_ids = self.item_to_part[safe_items]
        tag_ids = self.item_to_tags[safe_items]

        part_emb = self.part_embedding(part_ids).unsqueeze(-2)
        tag_emb = self.tag_embedding(tag_ids)
        skill_emb = self.skill_embedding(skills).unsqueeze(-2)
        item_emb = self.item_embedding(items).unsqueeze(-2)

        nodes = torch.cat(
            [
                part_emb,
                tag_emb,
                skill_emb,
                item_emb,
            ],
            dim=-2,
        )

        valid = torch.cat(
            [
                part_ids.ne(0).unsqueeze(-1),
                tag_ids.ne(0),
                skills.ne(0).unsqueeze(-1),
                items.ne(0).unsqueeze(-1),
            ],
            dim=-1,
        )

        for layer in self.layers:
            nodes = layer(
                nodes,
                self.adjacency,
                valid,
            )

        question_node = nodes[..., -1, :]

        valid_f = valid.unsqueeze(-1).float()
        pooled = (
            (nodes * valid_f).sum(dim=-2)
            / valid_f.sum(dim=-2).clamp_min(1.0)
        )

        return self.final_norm(
            question_node + pooled
        )


class HGAKTKT3(nn.Module):
    """
    HGAKT-style KT3 adaptation.

    Components:
      hierarchical graph representation
      multi-head graph attention
      dynamic gating
      LSTM knowledge-state modeling
      task/item difficulty
      shared target representation
    """

    def __init__(
        self,
        num_skills: int,
        num_items: int,
        num_parts: int,
        num_tags: int,
        item_to_part: torch.Tensor,
        item_to_tags: torch.Tensor,
        item_difficulty: torch.Tensor,
        behavior_dim: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        graph_heads: int = 6,
        graph_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.graph_encoder = HierarchicalGraphEncoder(
            num_skills=num_skills,
            num_items=num_items,
            num_parts=num_parts,
            num_tags=num_tags,
            item_to_part=item_to_part,
            item_to_tags=item_to_tags,
            embedding_dim=embedding_dim,
            num_heads=graph_heads,
            num_layers=graph_layers,
            dropout=dropout,
        )

        self.response_embedding = nn.Embedding(
            2,
            embedding_dim,
        )

        self.behavior_projection = nn.Sequential(
            nn.Linear(behavior_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )

        self.target_profile_projection = nn.Sequential(
            nn.Linear(CAUSAL_BEHAVIOR_DIM, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )

        self.register_buffer(
            "item_difficulty",
            item_difficulty.float(),
            persistent=True,
        )

        self.difficulty_projection = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.GELU(),
        )

        # Dynamic gating between graph structure and learner interaction state.
        self.dynamic_gate = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.Sigmoid(),
        )

        self.history_norm = nn.LayerNorm(embedding_dim)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.knowledge_projection = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )

        self.shared_gate = nn.Sequential(
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.Sigmoid(),
        )

        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim * 5, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        skills,
        items,
        correct,
        behavior=None,
        elapsed=None,
        lag=None,
        **_,
    ):

        if behavior is None:
            raise ValueError(
                "HGAKTKT3 requires causal learner behavior features"
            )

        # ----- historical interactions -----

        history_graph = self.graph_encoder(
            skills[:, :-1],
            items[:, :-1],
        )

        history_response = self.response_embedding(
            correct[:, :-1].clamp(0, 1)
        )

        history_behavior = self.behavior_projection(
            behavior[:, :-1]
        )

        gate = self.dynamic_gate(
            torch.cat(
                [
                    history_graph,
                    history_response,
                    history_behavior,
                ],
                dim=-1,
            )
        )

        interaction_state = (
            gate * history_graph
            + (1.0 - gate)
            * (history_response + history_behavior)
        )

        interaction_state = self.history_norm(
            interaction_state
        )

        interaction_state = self.dropout(
            interaction_state
        )

        interaction_state = torch.where(
            items[:, :-1].unsqueeze(-1).eq(0),
            torch.zeros_like(interaction_state),
            interaction_state,
        )

        knowledge_sequence, _ = self.lstm(
            interaction_state
        )

        knowledge_state = self.knowledge_projection(
            knowledge_sequence
        )

        # ----- target interaction -----

        target_graph = self.graph_encoder(
            skills[:, 1:],
            items[:, 1:],
        )

        target_profile = self.target_profile_projection(
            behavior[:, 1:, :CAUSAL_BEHAVIOR_DIM]
        )

        target_difficulty = self.item_difficulty[
            items[:, 1:]
        ].unsqueeze(-1)

        difficulty_emb = self.difficulty_projection(
            target_difficulty
        )

        shared_gate = self.shared_gate(
            torch.cat(
                [
                    knowledge_state,
                    target_graph,
                    target_profile,
                    difficulty_emb,
                ],
                dim=-1,
            )
        )

        shared_state = (
            shared_gate * knowledge_state
            + (1.0 - shared_gate)
            * (
                target_graph
                + target_profile
                + difficulty_emb
            )
        )

        logits = self.predictor(
            torch.cat(
                [
                    shared_state,
                    knowledge_state,
                    target_graph,
                    target_profile,
                    difficulty_emb,
                ],
                dim=-1,
            )
        ).squeeze(-1)

        return logits
