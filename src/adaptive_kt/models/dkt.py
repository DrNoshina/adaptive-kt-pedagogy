from __future__ import annotations

import torch
from torch import nn


class DKT(nn.Module):

    def __init__(
        self,
        num_skills: int,
        num_items: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

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

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)

        self.predictor = nn.Sequential(
            nn.Linear(
                hidden_dim + embedding_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        skills,
        items,
        correct,
        **_,
    ):

        # Current interaction:
        # question + skill + observed response
        current = (
            self.skill_embedding(skills)
            + self.item_embedding(items)
            + self.response_embedding(
                correct.clamp(0, 1)
            )
        )

        current = torch.where(
            items.unsqueeze(-1).eq(0),
            torch.zeros_like(current),
            current,
        )

        hidden, _ = self.lstm(current)

        # Next question is known at prediction time.
        # Its ANSWER is NOT used.
        target = (
            self.skill_embedding(skills[:, 1:])
            + self.item_embedding(items[:, 1:])
        )

        features = torch.cat(
            [
                self.dropout(hidden[:, :-1]),
                target,
            ],
            dim=-1,
        )

        return self.predictor(features).squeeze(-1)
