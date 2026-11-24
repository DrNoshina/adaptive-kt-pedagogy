import torch

from adaptive_kt.models.dkt import DKT
from adaptive_kt.models.transformer_kt import TransformerKT


def test_dkt_output_shape():

    model = DKT(
        num_skills=10,
        num_items=20,
        embedding_dim=8,
        hidden_dim=12,
    )

    skills = torch.tensor(
        [[2,3,4],[5,6,0]]
    )

    items = torch.tensor(
        [[2,7,9],[4,8,0]]
    )

    correct = torch.tensor(
        [[1,0,1],[0,1,0]]
    )

    output = model(
        skills,
        items,
        correct,
    )

    assert output.shape == (2,2)


def test_transformer_output_shape():

    model = TransformerKT(
        num_skills=10,
        num_items=20,
        max_len=8,
        embedding_dim=8,
        num_heads=2,
        num_layers=1,
    )

    skills = torch.tensor(
        [[2,3,4],[5,6,0]]
    )

    items = torch.tensor(
        [[2,7,9],[4,8,0]]
    )

    correct = torch.tensor(
        [[1,0,1],[0,1,0]]
    )

    output = model(
        skills,
        items,
        correct,
    )

    assert output.shape == (2,2)
