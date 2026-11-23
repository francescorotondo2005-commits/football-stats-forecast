"""Feature pre-partita.

Regola fondamentale: la riga di una partita usa SOLO partite giocate in date precedenti.
Ogni media mobile è calcolata con shift(1), e la media del campionato usa solo giorni precedenti.
Il test tests/test_features.py verifica che cambiare il risultato di una partita non modifichi
le sue feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LEAGUES, STATS

SHORT_HALFLIFE = 3  # partite: forma recente
LONG_HALFLIFE = 10  # partite: livello di fondo della squadra
VENUE_HALFLIFE = 6  # partite giocate nello stesso contesto (solo casa / solo trasferta)
ADJ_HALFLIFE = 10  # partite: rendimento aggiustato per la forza degli avversari affrontati
LEAGUE_HALFLIFE_DAYS = 365  # giorni: memoria della media del campionato
# Pausa oltre la quale una squadra riparte senza storia (tornata dalla serie inferiore).
# Nei dati le pause di chi resta nel campionato arrivano a 194 giorni (Ligue 1 2019-20,
# interrotta a marzo per il Covid), i ritorni dalla serie inferiore partono da 384:
# 300 sta a metà di questo intervallo vuoto.
NEW_SPELL_GAP_DAYS = 300


def _ewm_prev(values: pd.Series, keys: list[pd.Series], halflife: float) -> pd.Series:
    """Media esponenziale delle partite PRECEDENTI, separata per gruppo."""
    return values.groupby(keys).transform(
        lambda s: s.shift(1).ewm(halflife=halflife, min_periods=1, ignore_na=True).mean()
    )


def _league_prior_mean(long: pd.DataFrame, col: str) -> pd.Series:
    """Media del campionato calcolata solo sui giorni precedenti (niente partite dello stesso giorno).

    Le partite pesano meno man mano che invecchiano (dimezzano il peso ogni LEAGUE_HALFLIFE_DAYS):
    una media su tutta la storia dal 2014 reagirebbe troppo lentamente ai cambiamenti del
    campionato (es. arbitraggi più o meno severi, stili di gioco che cambiano).
    """
    daily = long.groupby(["league", "date"])[col].agg(["sum", "count"]).reset_index()
    lam = np.log(2) / LEAGUE_HALFLIFE_DAYS
    parts = []
    for _, g in daily.groupby("league", sort=False):
        days = (g["date"] - g["date"].iloc[0]).dt.days.to_numpy(float)
        sums, counts = g["sum"].to_numpy(float), g["count"].to_numpy(float)
        prior = np.empty(len(g))
        s = c = 0.0
        for i in range(len(g)):
            if i:
                decay = np.exp(-lam * (days[i] - days[i - 1]))
                s, c = s * decay, c * decay
            prior[i] = s / c if c > 0 else np.nan  # prima di aggiungere il giorno stesso
            s, c = s + sums[i], c + counts[i]
        parts.append(pd.Series(prior, index=pd.MultiIndex.from_frame(g[["league", "date"]])))
    prior = pd.concat(parts).rename(f"_lg_{col}")
    return long[["league", "date"]].join(prior, on=["league", "date"])[f"_lg_{col}"]


def build_features(long: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge al formato lungo (una riga per squadra e partita) tutte le feature pre-partita."""
    df = long.sort_values(["date", "match_id", "is_home"], ascending=[True, True, False]).reset_index(drop=True)

    # "Spell": una squadra che sparisce per più di un'estate (retrocessione) riparte senza storia.
    gap = df.groupby("team")["date"].diff().dt.days
    df["spell"] = (gap.isna() | (gap > NEW_SPELL_GAP_DAYS)).groupby(df["team"]).cumsum()
    df["n_prev"] = df.groupby(["team", "spell"]).cumcount()

    team_keys = [df["team"], df["spell"]]
    venue_keys = [df["team"], df["spell"], df["is_home"]]

    team_feats = {}
    for stat in STATS:
        lg = _league_prior_mean(df, f"{stat}_for")
        df[f"lg_{stat}"] = lg
        for side in ("for", "against"):
            col = f"{stat}_{side}"
            for name, keys, hl in (
                ("s", team_keys, SHORT_HALFLIFE),
                ("l", team_keys, LONG_HALFLIFE),
                ("v", venue_keys, VENUE_HALFLIFE),
            ):
                feat = f"{stat}_{side}_{name}"
                # senza storia (neopromossa, inizio dati) si parte dalla media del campionato
                team_feats[feat] = _ewm_prev(df[col], keys, hl).fillna(lg)

    tf = pd.DataFrame(team_feats, index=df.index)
    df = pd.concat([df, tf.add_prefix("t_")], axis=1)

    # Le stesse feature viste dal lato dell'avversario: le prendo dalla riga gemella della partita.
    opp = pd.concat([df[["match_id", "team", "n_prev"]], tf], axis=1)
    opp = opp.rename(columns={"team": "opponent", "n_prev": "o_n_prev"})
    opp = opp.rename(columns={c: f"o_{c}" for c in tf.columns})
    df = df.merge(opp, on=["match_id", "opponent"], how="left")
    df = _add_opponent_adjusted(df)

    for slug in LEAGUES:
        df[f"league_{slug}"] = (df["league"] == slug).astype(int)

    return df


def _add_opponent_adjusted(df: pd.DataFrame) -> pd.DataFrame:
    """Medie aggiustate per la forza degli avversari affrontati.

    Le medie semplici non sanno contro chi sono state ottenute: 15 tiri contro la difesa
    migliore valgono più di 15 tiri contro la peggiore. Per ogni partita passata si calcola
    il rapporto tra quanto la squadra ha prodotto e quanto quell'avversario concedeva di solito
    PRIMA della partita (o_<stat>_against_l, già pre-partita); poi si fa la media esponenziale
    di questi rapporti sulle partite precedenti (shift(1), come per le altre feature) e la si
    riporta in unità della statistica moltiplicando per la media del campionato.
    Lo stesso per la difesa: quanto ha concesso rispetto a quanto quell'avversario produceva.
    """
    keys = [df["team"], df["spell"]]
    adj = {}
    for stat in STATS:
        lg = df[f"lg_{stat}"]
        floor = 0.2 * lg  # evita rapporti esplosivi con medie quasi nulle
        r_for = df[f"{stat}_for"] / df[f"o_{stat}_against_l"].clip(lower=floor)
        r_against = df[f"{stat}_against"] / df[f"o_{stat}_for_l"].clip(lower=floor)
        adj[f"{stat}_adjfor"] = _ewm_prev(r_for, keys, ADJ_HALFLIFE).fillna(1.0) * lg
        adj[f"{stat}_adjagainst"] = _ewm_prev(r_against, keys, ADJ_HALFLIFE).fillna(1.0) * lg
    at = pd.DataFrame(adj, index=df.index)
    df = pd.concat([df, at.add_prefix("t_")], axis=1)

    opp = pd.concat([df[["match_id", "team"]], at], axis=1).rename(columns={"team": "opponent"})
    opp = opp.rename(columns={c: f"o_{c}" for c in at.columns})
    return df.merge(opp, on=["match_id", "opponent"], how="left")


def feature_columns(target_stat: str) -> list[str]:
    """Colonne in input al modello per una statistica.

    Per prevedere i tiri della squadra T contro O servono soprattutto:
    quanti tiri fa T (attacco) e quanti ne concede O (difesa), più il contesto.
    Includo anche le altre statistiche: ad esempio una squadra che subisce molti tiri
    tende ad averne pochi.
    """
    cols = ["is_home", "n_prev", "o_n_prev", f"lg_{target_stat}"]
    for stat in STATS:
        for name in ("s", "l", "v"):
            cols.append(f"t_{stat}_for_{name}")
            cols.append(f"o_{stat}_against_{name}")
        cols.append(f"t_{stat}_against_l")
        cols.append(f"o_{stat}_for_l")
        # versioni aggiustate per la forza degli avversari affrontati
        cols += [f"t_{stat}_adjfor", f"o_{stat}_adjagainst", f"t_{stat}_adjagainst", f"o_{stat}_adjfor"]
    cols += [f"league_{slug}" for slug in LEAGUES]
    return cols
