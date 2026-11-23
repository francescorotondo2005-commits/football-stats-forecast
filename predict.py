"""Previsione per una singola partita futura."""

from __future__ import annotations

import difflib

import joblib
import pandas as pd

from .config import DEFAULT_LEVEL, LEAGUES, MODELS_DIR, STAT_LABELS_IT, STATS
from .data import add_future_match, load_matches, to_team_rows
from .features import build_features


def resolve_team(name: str, league: str, long: pd.DataFrame) -> str:
    """Trova il nome della squadra come compare nei dati (es. 'inter' -> 'Inter')."""
    last_season = long.loc[long.league == league, "season"].max()
    teams = sorted(long.loc[(long.league == league) & (long.season >= last_season - 1), "team"].unique())
    lookup = {t.lower(): t for t in teams}
    if name.lower() in lookup:
        return lookup[name.lower()]
    close = difflib.get_close_matches(name.lower(), list(lookup), n=1, cutoff=0.6)
    if close:
        return lookup[close[0]]
    raise SystemExit(f"Squadra '{name}' non trovata in {LEAGUES[league][0]}. Squadre disponibili:\n  "
                     + ", ".join(teams))


def predict_match(league: str, home: str, away: str, level: float = DEFAULT_LEVEL) -> pd.DataFrame:
    bundle_path = MODELS_DIR / "footstats.joblib"
    if not bundle_path.exists():
        raise SystemExit("Modelli non trovati: esegui prima `python -m footstats train`.")
    bundle = joblib.load(bundle_path)

    long = to_team_rows(load_matches())
    home, away = resolve_team(home, league, long), resolve_team(away, league, long)
    # Le feature di una partita dipendono solo dal suo campionato (medie delle squadre, dei loro
    # avversari e del campionato): calcolarle anche sugli altri quattro le lascerebbe identiche
    # (lo verifica tests/test_features.py) e costerebbe circa 5 volte tanto.
    league_rows = long[long["league"] == league]
    feats = build_features(add_future_match(league_rows, league, home, away))
    rows = feats.tail(2)  # le due righe della partita futura: casa e ospite

    out = []
    for i in range(len(rows)):
        X = rows.iloc[[i]]  # DataFrame di una riga, con i tipi originali
        row = X.iloc[0]
        for stat in STATS:
            model = bundle["models"][stat]
            mu = float(model.predict_mean(X)[0])
            lo, hi = model.predict_interval(X, level)
            prob = float(model.dist.prob_between([mu], lo, hi)[0])
            out.append({"squadra": row.team, "statistica": STAT_LABELS_IT[stat], "atteso": mu,
                        "min": int(lo[0]), "max": int(hi[0]), "prob": prob})
    result = pd.DataFrame(out)
    # la data dei dati usati per le feature: dopo un `download` può essere più recente dell'ultimo `train`
    data_until = league_rows["date"].max().strftime("%Y-%m-%d")
    result.attrs.update(home=home, away=away, level=level, data_until=data_until)
    return result


def format_prediction(result: pd.DataFrame) -> str:
    a = result.attrs
    lines = [f"\n{a['home']} – {a['away']}   (range al {a['level']:.0%}, dati fino al {a['data_until']})", ""]
    for team in (a["home"], a["away"]):
        lines.append(f"  {team}")
        for r in result[result.squadra == team].itertuples():
            lines.append(f"    {r.statistica:<14} atteso {r.atteso:5.1f}    range {r.min:>2}–{r.max:<2}"
                         f"   (probabilità {r.prob:.0%})")
        lines.append("")
    return "\n".join(lines)
