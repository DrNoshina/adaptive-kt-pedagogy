import pandas as pd

from adaptive_kt.data.splits import assert_user_disjoint, split_by_user


def test_split_by_user_has_no_overlap():
    rows = []
    for user in range(50):
        for t in range(4):
            rows.append({"user_id": f"u{user}", "timestamp": t})
    frame = pd.DataFrame(rows)
    split = split_by_user(frame, 0.8, 0.1, 0.1, seed=42)
    assert_user_disjoint(split)
    train = set(split["train"]["user_id"])
    val = set(split["validation"]["user_id"])
    test = set(split["test"]["user_id"])
    assert not (train & val)
    assert not (train & test)
    assert not (val & test)
    assert train | val | test == set(frame["user_id"])
