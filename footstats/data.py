"""Download e caricamento dei dati partita.

Fonte: football-data.co.uk (statistiche di fine partita dei 5 principali campionati).
Se il sito originale non è raggiungibile si usa il mirror pubblico
github.com/datasets/football-datasets (licenza ODC-PDDL), che riporta gli stessi dati.
"""

from __future__ import annotations

import io
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

from .config import FIRST_SEASON, LEAGUES, RAW_DIR, STATS, current_season, season_code

ORIGINAL_URL = "https://www.football-data.co.uk/mmz4281/{code}/{div}.csv"
MIRROR_URL = (
    "https://raw.githubusercontent.com/datasets/football-datasets/main/"
    "datasets/{league}/season-{code}.csv"
)

RAW_COLUMNS = [
    "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "HS", "AS", "HST", "AST",
    "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR",
]


def _fetch(url: str, timeout: int = 30) -> bytes | None:
    req = urllib.request.Request(url, headers={"User-Agent": "footstats/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            return body if len(body) > 200 else None
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def download(leagues=None, first_season=FIRST_SEASON, last_season=None, force=False) -> None:
    """Scarica un CSV per campionato e stagione in data/raw/."""
    leagues = leagues or list(LEAGUES)
    last_season = last_season if last_season is not None else current_season()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for league in leagues:
        div = LEAGUES[league][1]
        for year in range(first_season, last_season + 1):
            code = season_code(year)
            target = RAW_DIR / f"{league}_{code}.csv"
            is_current = year == current_season()
            if target.exists() and not force and not is_current:
                continue
            body = _fetch(ORIGINAL_URL.format(code=code, div=div))
            source = "football-data.co.uk"
            if body is None:
                body = _fetch(MIRROR_URL.format(league=league, code=code))
                source = "mirror"
            if body is None:
                print(f"  [skip] {league} {code}: non disponibile")
                continue
            target.write_bytes(body)
            print(f"  [ok]   {league} {code} ({source})")
            time.sleep(0.2)


def _parse_dates(s: pd.Series) -> pd.Series:
    """Gestisce sia '2024-08-17' (mirror) sia '17/08/2024' o '17/08/24' (originale)."""
    iso = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    long_ = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    short = pd.to_datetime(s, format="%d/%m/%y", errors="coerce")
    return iso.fillna(long_).fillna(short)


def _read_raw(path) -> pd.DataFrame:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc, on_bad_lines="skip")
            break
        except UnicodeDecodeError:
            continue
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: colonne mancanti {missing}")
    return df[RAW_COLUMNS].dropna(subset=["HomeTeam", "AwayTeam", "Date"])


def load_matches(leagues=None) -> pd.DataFrame:
    """Una riga per partita, ordinate per data. Colonne: league, season, date, home, away, stat_h/stat_a."""
    leagues = leagues or list(LEAGUES)
    frames = []
    for league in leagues:
        for path in sorted(RAW_DIR.glob(f"{league}_*.csv")):
            code = path.stem.split("_")[-1]
            df = _read_raw(path)
            df["league"] = league
            df["season"] = 2000 + int(code[:2])
            frames.append(df)
    if not frames:
        raise FileNotFoundError("Nessun dato in data/raw/: esegui prima `python -m footstats download`.")

    df = pd.concat(frames, ignore_index=True)
    df["date"] = _parse_dates(df["Date"].astype(str))
    df = df.dropna(subset=["date"])
    df["home"] = df["HomeTeam"].str.strip()
    df["away"] = df["AwayTeam"].str.strip()

    out = df[["league", "season", "date", "home", "away"]].copy()
    for stat, (hc, ac) in STATS.items():
        if stat == "cards":
            out["cards_h"] = pd.to_numeric(df["HY"], errors="coerce") + pd.to_numeric(df["HR"], errors="coerce")
            out["cards_a"] = pd.to_numeric(df["AY"], errors="coerce") + pd.to_numeric(df["AR"], errors="coerce")
        else:
            out[f"{stat}_h"] = pd.to_numeric(df[hc], errors="coerce")
            out[f"{stat}_a"] = pd.to_numeric(df[ac], errors="coerce")

    stat_cols = [c for c in out.columns if c.endswith(("_h", "_a"))]
    out = out.dropna(subset=stat_cols)  # partite senza statistiche complete
    out = out.drop_duplicates(subset=["league", "date", "home", "away"])
    return out.sort_values(["date", "league", "home"]).reset_index(drop=True)


def to_team_rows(matches: pd.DataFrame) -> pd.DataFrame:
    """Formato 'lungo': due righe per partita, una dal punto di vista di ciascuna squadra.

    Per ogni statistica: <stat>_for (prodotta dalla squadra) e <stat>_against (concessa).
    """
    rows = []
    for is_home, (me, opp, mine, theirs) in ((1, ("home", "away", "_h", "_a")), (0, ("away", "home", "_a", "_h"))):
        part = matches[["league", "season", "date"]].copy()
        part["team"] = matches[me].values
        part["opponent"] = matches[opp].values
        part["is_home"] = is_home
        for stat in STATS:
            part[f"{stat}_for"] = matches[f"{stat}{mine}"].values
            part[f"{stat}_against"] = matches[f"{stat}{theirs}"].values
        part["match_id"] = matches.index.values
        rows.append(part)
    long = pd.concat(rows, ignore_index=True)
    return long.sort_values(["date", "match_id", "is_home"], ascending=[True, True, False]).reset_index(drop=True)


def add_future_match(long: pd.DataFrame, league: str, home: str, away: str) -> pd.DataFrame:
    """Aggiunge una partita futura (statistiche ignote) per calcolarne le feature pre-partita."""
    last = long["date"].max()
    season = int(long.loc[long["league"] == league, "season"].max())
    mid = int(long["match_id"].max()) + 1
    extra = []
    for is_home, team, opp in ((1, home, away), (0, away, home)):
        row = {"league": league, "season": season, "date": last + pd.Timedelta(days=1),
               "team": team, "opponent": opp, "is_home": is_home, "match_id": mid}
        for stat in STATS:
            row[f"{stat}_for"] = np.nan
            row[f"{stat}_against"] = np.nan
        extra.append(row)
    return pd.concat([long, pd.DataFrame(extra)], ignore_index=True)
