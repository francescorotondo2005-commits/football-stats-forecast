"""Test sulle feature: nessuna informazione dal futuro (data leakage)."""

import numpy as np
import pandas as pd
import pytest

from footstats.config import STATS
from footstats.data import add_future_match, to_team_rows
from footstats.features import build_features, feature_columns


def _synthetic_matches(n_rounds=30, seed=0):
    """Mini campionato a 4 squadre con statistiche casuali."""
    rng = np.random.default_rng(seed)
    teams = ["A", "B", "C", "D"]
    pairs = [("A", "B"), ("C", "D"), ("A", "C"), ("B", "D"), ("A", "D"), ("B", "C")]
    rows = []
    start = pd.Timestamp("2020-08-01")
    for r in range(n_rounds):
        for j, (h, a) in enumerate(pairs[(r * 2) % 6:(r * 2) % 6 + 2]):
            if r % 2:
                h, a = a, h
            row = {"league": "serie-a", "season": 2020, "date": start + pd.Timedelta(days=7 * r), "home": h, "away": a}
            for stat in STATS:
                row[f"{stat}_h"] = float(rng.poisson(10))
                row[f"{stat}_a"] = float(rng.poisson(8))
            rows.append(row)
    assert set(teams)
    return pd.DataFrame(rows)


def _all_feature_cols():
    return sorted(set().union(*(feature_columns(s) for s in STATS)))


def test_changing_a_result_does_not_change_its_own_features():
    matches = _synthetic_matches()
    base = build_features(to_team_rows(matches))

    target = 40  # una partita a metà stagione
    altered = matches.copy()
    for stat in STATS:
        altered.loc[target, f"{stat}_h"] += 50
        altered.loc[target, f"{stat}_a"] += 50
    changed = build_features(to_team_rows(altered))

    cols = _all_feature_cols()
    date = matches.loc[target, "date"]
    before_or_same = base["date"] <= date
    pd.testing.assert_frame_equal(
        base.loc[before_or_same, cols].reset_index(drop=True),
        changed.loc[changed["date"] <= date, cols].reset_index(drop=True),
    )
    # ...mentre le partite successive delle due squadre devono accorgersene
    later = base["date"] > date
    assert not np.allclose(base.loc[later, cols].to_numpy(float), changed.loc[later, cols].to_numpy(float))


def test_future_match_gets_complete_features():
    long = to_team_rows(_synthetic_matches())
    feats = build_features(add_future_match(long, "serie-a", "A", "B"))
    future = feats.tail(2)
    assert list(future["team"]) == ["A", "B"]
    assert future["shots_for"].isna().all()
    assert not future[feature_columns("shots")].isna().any().any()


def test_opponent_features_come_from_the_opponent():
    feats = build_features(to_team_rows(_synthetic_matches()))
    m = feats[feats.match_id == feats.match_id.iloc[-1]]
    home, away = m[m.is_home == 1].iloc[0], m[m.is_home == 0].iloc[0]
    assert home["o_shots_against_l"] == pytest.approx(away["t_shots_against_l"])
    assert away["o_shots_against_l"] == pytest.approx(home["t_shots_against_l"])
