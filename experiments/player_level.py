"""Esperimento: dalle squadre ai giocatori.

Domanda: sapere CHI scende in campo migliora la previsione delle statistiche di una squadra,
rispetto alle sole medie di squadra usate da footstats?

Dati: Wyscout, stagione 2017-18 dei 5 principali campionati, rilasciati in forma aperta da
  Pappalardo et al. (2019), "A public data set of spatio-temporal match events in soccer
  competitions", Scientific Data 6:236 — licenza CC BY 4.0.
  https://figshare.com/collections/Soccer_match_event_dataset/4415000

Tre versioni dello stesso modello (GLM di Poisson, come in footstats), confrontate sul girone
di ritorno (partite dal 1° gennaio 2018), addestrate sulle partite precedenti:
  A   solo medie di squadra (forma, livello, casa/trasferta, avversario, campionato)
  B1  A + rendimento dei giocatori dell'undici della partita PRECEDENTE ("formazione abituale")
  B2  A + rendimento dei giocatori dell'undici UFFICIALE della partita (noto ~1 ora prima)
B1 - A misura il valore di guardare i giocatori; B2 - B1 il valore di conoscere la formazione.

Uso:  python experiments/player_level.py
(scarica circa 80 MB in data/wyscout/ la prima volta, poi scrive reports/player_level.md)
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from footstats.data import load_matches as load_football_data  # noqa: E402
from footstats.models import PoissonGLM  # noqa: E402

DATA = ROOT / "data" / "wyscout"
REPORT = ROOT / "reports" / "player_level.md"

FILES = {  # nome -> (url, md5) dalla collezione figshare
    "events.zip": ("https://ndownloader.figshare.com/files/14464685", "7c20e8647e7eda58d7838a0c7b1ec6ab"),
    "matches.zip": ("https://ndownloader.figshare.com/files/14464622", "51d80beb17480919f69a53a0152c2d71"),
    "players.json": ("https://ndownloader.figshare.com/files/15073721", "f28ddf6326281efeda6488b2169f5609"),
}
LEAGUES = {"Italy": "serie-a", "England": "premier-league", "Spain": "la-liga",
           "Germany": "bundesliga", "France": "ligue-1"}
STATS = ["goals", "shots", "sot", "fouls", "corners", "cards"]
PLAYER_STATS = ["goals", "shots", "sot", "fouls", "cards"]  # i corner sono un evento di squadra
LABELS = {"goals": "Gol", "shots": "Tiri", "sot": "Tiri in porta", "fouls": "Falli",
          "corners": "Corner", "cards": "Cartellini"}

FOUL_TYPES = {"Foul", "Hand foul", "Late card foul", "Violent Foul"}  # falli commessi (no proteste/simulazioni)
TAG_GOAL, TAG_ACCURATE = 101, 1801

TEAM_HALFLIFE = 5  # partite
SHRINK_MATCHES = 5  # peso della media del ruolo, in "partite intere" equivalenti
TEST_FROM = pd.Timestamp("2018-01-01")
BURN_IN_GAMEWEEK = 6  # le prime giornate non hanno abbastanza storia per addestrare


# ───────────────────────── download e lettura ─────────────────────────

def download() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    for name, (url, md5) in FILES.items():
        path = DATA / name
        if not path.exists():
            print(f"  scarico {name}…", flush=True)
            urllib.request.urlretrieve(url, path)
        if hashlib.md5(path.read_bytes()).hexdigest() != md5:
            raise SystemExit(f"{name}: checksum diverso da quello pubblicato, cancella il file e riprova")


def iter_json_array(fh, chunk: int = 1 << 20):
    """Legge un array JSON un oggetto alla volta: i file eventi sono ~190 MB l'uno."""
    dec = json.JSONDecoder()
    buf = fh.read(chunk)
    pos = buf.index("[") + 1
    while True:
        while True:
            while pos < len(buf) and buf[pos] in " \r\n\t,":
                pos += 1
            if pos < len(buf):
                break
            more = fh.read(chunk)
            if not more:
                return
            buf, pos = more, 0
        if buf[pos] == "]":
            return
        try:
            obj, end = dec.raw_decode(buf, pos)
        except json.JSONDecodeError:
            more = fh.read(chunk)
            if not more:
                raise
            buf, pos = buf[pos:] + more, 0
            continue
        yield obj
        pos = end


def player_roles() -> dict[int, str]:
    players = json.loads((DATA / "players.json").read_text(encoding="utf-8"))
    return {p["wyId"]: (p.get("role") or {}).get("code2", "MD") for p in players}


def read_matches() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Una riga per squadra e partita, e una riga per giocatore e partita (minuti, cartellini)."""
    teams, players = [], []
    with zipfile.ZipFile(DATA / "matches.zip") as zf:
        for country, league in LEAGUES.items():
            for m in json.loads(zf.read(f"matches_{country}.json")):
                tds = list(m["teamsData"].values())
                for td in tds:
                    opp = next(t for t in tds if t["teamId"] != td["teamId"])
                    teams.append({"match_id": m["wyId"], "league": league, "gameweek": m["gameweek"],
                                  "date": pd.Timestamp(m["dateutc"]).normalize(), "team": td["teamId"],
                                  "opponent": opp["teamId"], "is_home": int(td["side"] == "home"),
                                  "goals_for": td["score"], "goals_against": opp["score"]})
                    players += _player_minutes(m["wyId"], td)
    return pd.DataFrame(teams), pd.DataFrame(players)


def _player_minutes(match_id: int, td: dict) -> list[dict]:
    f = td["formation"]
    start = {p["playerId"]: 0 for p in f["lineup"]}
    end = {p["playerId"]: 90 for p in f["lineup"]}
    subs = f.get("substitutions")
    for s in subs if isinstance(subs, list) else []:  # 6 squadre-partita hanno la stringa "null"
        minute = min(max(int(s["minute"]), 0), 90)
        if s["playerOut"] in end:
            end[s["playerOut"]] = minute
        start[s["playerIn"]], end[s["playerIn"]] = minute, 90
    rows = []
    for p in f["lineup"] + f["bench"]:
        pid = p["playerId"]
        red = int(p["redCards"])
        if red and pid in end:
            end[pid] = min(end[pid], red)  # espulso: i minuti finiscono lì
        minutes = max(end.get(pid, 0) - start.get(pid, 0), 0) if pid in start else 0
        rows.append({"match_id": match_id, "team": td["teamId"], "player": pid,
                     "starter": int(p in f["lineup"]), "minutes": minutes,
                     "cards": int(p["yellowCards"] != "0") + int(red > 0)})
    return rows


def read_events() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tiri, tiri in porta, gol e falli per giocatore e partita; corner per squadra e partita."""
    pstat = defaultdict(lambda: np.zeros(4))  # (match, team, player) -> goals, shots, sot, fouls
    corners = defaultdict(int)
    with zipfile.ZipFile(DATA / "events.zip") as zf:
        for country in LEAGUES:
            print(f"  eventi {country}…", flush=True)
            with io.TextIOWrapper(zf.open(f"events_{country}.json"), encoding="utf-8") as fh:
                for e in iter_json_array(fh):
                    name, sub = e["eventName"], e["subEventName"]
                    key = (e["matchId"], e["teamId"], e["playerId"])
                    if name == "Shot" or (name == "Free Kick" and sub in ("Free kick shot", "Penalty")):
                        tags = {t["id"] for t in e["tags"]}
                        v = pstat[key]
                        v[1] += 1
                        v[2] += TAG_ACCURATE in tags
                        v[0] += TAG_GOAL in tags
                    elif name == "Foul" and sub in FOUL_TYPES:
                        pstat[key][3] += 1
                    elif name == "Free Kick" and sub == "Corner":
                        corners[(e["matchId"], e["teamId"])] += 1
    p = pd.DataFrame([(*k, *v) for k, v in pstat.items()],
                     columns=["match_id", "team", "player", "goals", "shots", "sot", "fouls"])
    c = pd.DataFrame([(*k, v) for k, v in corners.items()], columns=["match_id", "team", "corners"])
    return p, c


# ───────────────────────── tabelle e feature ─────────────────────────

def build_tables():
    teams, pm = read_matches()
    pev, corners = read_events()
    pm = pm.merge(pev, on=["match_id", "team", "player"], how="left").fillna(
        {"goals": 0, "shots": 0, "sot": 0, "fouls": 0})
    # totali di squadra direttamente dagli eventi (anche quelli senza giocatore in distinta)
    ev = pev.groupby(["match_id", "team"])[["shots", "sot", "fouls"]].sum().reset_index()
    cards = pm.groupby(["match_id", "team"])["cards"].sum().reset_index()
    teams = (teams.merge(ev, on=["match_id", "team"], how="left").merge(cards, on=["match_id", "team"], how="left")
             .merge(corners, on=["match_id", "team"], how="left"))
    teams[["shots", "sot", "fouls", "cards", "corners"]] = teams[["shots", "sot", "fouls", "cards", "corners"]].fillna(0)
    teams["goals"] = teams["goals_for"]
    teams = teams.drop(columns=["goals_for", "goals_against"])  # ricostruiti sotto come <stat>_against
    against =teams[["match_id", "team"] + STATS].rename(columns={"team": "opponent", **{s: f"{s}_against" for s in STATS}})
    teams = teams.merge(against, on=["match_id", "opponent"])
    pm = pm.merge(teams[["match_id", "date"]].drop_duplicates(), on="match_id")
    return teams.sort_values(["date", "match_id", "is_home"], ascending=[True, True, False]).reset_index(drop=True), pm


def team_features(teams: pd.DataFrame) -> pd.DataFrame:
    """Medie esponenziali delle partite precedenti (shift(1)), come in footstats."""
    df = teams.copy()
    df["n_prev"] = df.groupby("team").cumcount()
    for s in STATS:
        for side in ("", "_against"):
            col = f"{s}{side}"
            df[f"_ewm_{col}"] = df.groupby("team")[col].transform(
                lambda x: x.shift(1).ewm(halflife=TEAM_HALFLIFE, min_periods=1).mean())
        daily = df.groupby(["league", "date"])[s].agg(["sum", "count"])
        cum = daily.groupby(level="league").cumsum() - daily
        lg = (cum["sum"] / cum["count"].replace(0, np.nan)).rename("_lg")
        df[f"lg_{s}"] = df[["league", "date"]].join(lg, on=["league", "date"])["_lg"]
        df[f"t_{s}_for"] = df[f"_ewm_{s}"].fillna(df[f"lg_{s}"])
        df[f"_t_{s}_against"] = df[f"_ewm_{s}_against"].fillna(df[f"lg_{s}"])
    opp = df[["match_id", "team", "n_prev"] + [f"_t_{s}_against" for s in STATS] + [f"t_{s}_for" for s in STATS]]
    opp = opp.rename(columns={"team": "opponent", "n_prev": "o_n_prev",
                              **{f"_t_{s}_against": f"o_{s}_against" for s in STATS},
                              **{f"t_{s}_for": f"o_{s}_for" for s in STATS}})
    df = df.merge(opp, on=["match_id", "opponent"])
    for slug in LEAGUES.values():
        df[f"league_{slug}"] = (df["league"] == slug).astype(int)
    return df.drop(columns=[c for c in df.columns if c.startswith("_")])


def xi_features(teams: pd.DataFrame, pm: pd.DataFrame, roles: dict[int, str]) -> pd.DataFrame:
    """Somma, sugli 11 titolari, del rendimento ogni 90 minuti di ciascun giocatore.

    Il rendimento usa SOLO le partite giocate prima di quella data (in qualunque squadra dei
    5 campionati) e viene "ristretto" verso la media del suo ruolo quando ha giocato poco:
    rate = (eventi + K * media_ruolo) / (minuti/90 + K).
    Per ogni partita si calcola con l'undici ufficiale (xi) e con quello della partita
    precedente della stessa squadra (prev).
    """
    pm = pm.copy()
    pm["role"] = pm["player"].map(roles).fillna("MD")
    totals = defaultdict(lambda: np.zeros(len(PLAYER_STATS) + 1))  # player -> eventi..., minuti
    role_tot = defaultdict(lambda: np.zeros(len(PLAYER_STATS) + 1))
    last_xi: dict[int, list[int]] = {}
    out = []
    starters = pm[pm.starter == 1].groupby(["match_id", "team"])["player"].apply(list)
    for date, day in teams.groupby("date", sort=True):
        def rate(pid):
            t, r = totals[pid], role_tot[roles.get(pid, "MD")]
            r90 = r[:-1] / max(r[-1] / 90, 1e-9)
            return (t[:-1] + SHRINK_MATCHES * r90) / (t[-1] / 90 + SHRINK_MATCHES)
        for row in day.itertuples():
            xi = starters.get((row.match_id, row.team), [])
            prev = last_xi.get(row.team, xi)  # prima partita: nessuna formazione precedente
            out.append({"match_id": row.match_id, "team": row.team,
                        **{f"xi_{s}": v for s, v in zip(PLAYER_STATS, sum((rate(p) for p in xi), np.zeros(5)))},
                        **{f"prev_{s}": v for s, v in zip(PLAYER_STATS, sum((rate(p) for p in prev), np.zeros(5)))}})
        # solo DOPO aver calcolato le feature del giorno si aggiornano i totali con le sue partite
        played = pm[pm.match_id.isin(day.match_id)]
        for r in played.itertuples():
            vec = np.array([getattr(r, s) for s in PLAYER_STATS] + [r.minutes], float)
            totals[r.player] += vec
            role_tot[r.role] += vec
        for row in day.itertuples():
            last_xi[row.team] = starters.get((row.match_id, row.team), [])
    return pd.DataFrame(out)


# ───────────────────────── confronto ─────────────────────────

def columns(target: str, variant: str) -> list[str]:
    cols = ["is_home", "n_prev", "o_n_prev", f"lg_{target}"] + [f"league_{s}" for s in LEAGUES.values()]
    for s in STATS:
        cols += [f"t_{s}_for", f"o_{s}_against"]
    if variant in ("B1", "B2"):
        src = "prev" if variant == "B1" else "xi"
        cols += [f"t_{src}_{s}" for s in PLAYER_STATS] + [f"o_{src}_{s}" for s in PLAYER_STATS]
    return cols


def poisson_deviance(y, mu):
    mu = np.maximum(mu, 1e-9)
    return 2 * np.mean(np.where(y > 0, y * np.log(np.maximum(y, 1e-9) / mu), 0) - (y - mu))


def run() -> str:
    download()
    teams, pm = build_tables()
    df = team_features(teams)
    xi = xi_features(teams, pm, player_roles())
    for side, key in (("t", "team"), ("o", "opponent")):
        df = df.merge(xi.rename(columns={"team": key, **{c: f"{side}_{c}" for c in xi.columns if c.startswith(("xi_", "prev_"))}}),
                      on=["match_id", key])
    df = df.merge(lineup_changes(teams, pm), on=["match_id", "team"])
    train = df[(df.date < TEST_FROM) & (df.gameweek >= BURN_IN_GAMEWEEK)]
    test = df[df.date >= TEST_FROM]
    big = (test["changes"] >= 4).to_numpy()  # partite con almeno 4 titolari diversi dalla precedente

    rng = np.random.default_rng(0)
    test_matches = test["match_id"].unique()
    boot = [rng.choice(len(test_matches), len(test_matches)) for _ in range(1000)]
    rows = []
    for s in STATS:
        y_tr, y_te = train[s].to_numpy(float), test[s].to_numpy(float)
        pred = {v: PoissonGLM(alpha=1e-2).fit(train[columns(s, v)], y_tr).predict(test[columns(s, v)])
                for v in ("A", "B1", "B2")}
        err = {v: np.abs(y_te - p) for v, p in pred.items()}
        # intervallo di confidenza 95% della differenza di MAE (bootstrap sulle partite)
        by_match = pd.DataFrame({"m": test["match_id"].to_numpy(), **{v: e for v, e in err.items()}}).groupby("m").sum()
        idx = {m: i for i, m in enumerate(by_match.index)}
        order = np.array([idx[m] for m in test_matches])
        diffs = []
        for b in boot:
            sel = order[b]
            base = by_match["A"].to_numpy()[sel].sum()
            diffs.append(by_match["B2"].to_numpy()[sel].sum() / base - 1)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        # diagnostica: lo scostamento dell'undici dal solito spiega l'errore del modello di squadra?
        corr = np.nan
        if s in PLAYER_STATS:
            dev = np.log((test[f"t_xi_{s}"] + 0.1) / (test[f"t_prev_{s}"] + 0.1))
            corr = float(np.corrcoef(dev, np.log((y_te + 0.5) / (pred["A"] + 0.5)))[0, 1])
        rows.append({"stat": s, **{f"mae_{v}": float(e.mean()) for v, e in err.items()},
                     **{f"dev_{v}": poisson_deviance(y_te, p) for v, p in pred.items()},
                     "ci_lo": lo, "ci_hi": hi, "n_test": len(y_te), "n_train": len(y_tr),
                     "few_A": err["A"][~big].mean(), "few_B2": err["B2"][~big].mean(),
                     "many_A": err["A"][big].mean(), "many_B2": err["B2"][big].mean(), "corr": corr})
    res = pd.DataFrame(rows)
    sanity = sanity_check(teams)
    text = write_report(res, sanity, len(train), len(test), test["changes"].mean(), big.mean())
    REPORT.write_text(text, encoding="utf-8", newline="\n")
    return text


def lineup_changes(teams: pd.DataFrame, pm: pd.DataFrame) -> pd.DataFrame:
    """Quanti titolari sono diversi rispetto alla partita precedente della stessa squadra."""
    xi = pm[pm.starter == 1].groupby(["match_id", "team"])["player"].apply(frozenset).rename("xi").reset_index()
    xi = xi.merge(teams[["match_id", "team", "date"]], on=["match_id", "team"]).sort_values("date")
    xi["prev"] = xi.groupby("team")["xi"].shift(1)
    xi["changes"] = [len(a - b) if isinstance(b, frozenset) else 0 for a, b in zip(xi["xi"], xi["prev"])]
    return xi[["match_id", "team", "changes"]]


def sanity_check(teams: pd.DataFrame) -> pd.DataFrame:
    """Le medie per partita calcolate dagli eventi Wyscout devono somigliare a quelle di football-data."""
    fd = load_football_data()
    fd = fd[fd.season == 2017]
    out = []
    for s in ["shots", "sot", "fouls", "corners", "cards", "goals"]:
        fd_mean = (fd[f"{s}_h"].mean() + fd[f"{s}_a"].mean()) / 2
        out.append({"stat": s, "wyscout": teams[s].mean(), "football_data": fd_mean})
    return pd.DataFrame(out)


def write_report(res: pd.DataFrame, sanity: pd.DataFrame, n_train: int, n_test: int,
                 mean_changes: float, share_big: float) -> str:
    f2 = lambda x: f"{x:.3f}".replace(".", ",")  # noqa: E731
    pct = lambda x: f"{x * 100:+.1f}%".replace(".", ",")  # noqa: E731
    lines = [
        "# Dalle squadre ai giocatori: esperimento sui dati aperti Wyscout 2017-18",
        "",
        "Domanda: sapere **chi scende in campo** migliora la previsione delle statistiche di una squadra,",
        "rispetto alle sole medie di squadra? Stesso tipo di modello di footstats (GLM di Poisson),",
        f"addestrato sulle partite fino al 31/12/2017 ({n_train} righe squadra-partita, dalla "
        f"{BURN_IN_GAMEWEEK}ª giornata) e valutato sul resto della stagione ({n_test} righe).",
        "",
        "- **A**: solo medie di squadra (forma, livello, casa/trasferta, avversario, campionato).",
        "- **B1**: A + rendimento dei giocatori dell'undici della partita precedente (\"formazione abituale\").",
        "- **B2**: A + rendimento dei giocatori dell'undici ufficiale (noto circa un'ora prima della partita).",
        "",
        "Il rendimento di ogni giocatore è la sua media ogni 90 minuti nelle partite precedenti, ristretta",
        f"verso la media del suo ruolo quando ha giocato poco (peso pari a {SHRINK_MATCHES} partite intere).",
        "",
        "## Errore medio assoluto (MAE) sul girone di ritorno",
        "",
        "| Statistica | A: squadra | B1: + formazione abituale | B2: + formazione ufficiale | B2 vs A | Intervallo 95% di B2 vs A |",
        "|---|---|---|---|---|---|",
    ]
    for r in res.itertuples():
        lines.append(f"| {LABELS[r.stat]} | {f2(r.mae_A)} | {f2(r.mae_B1)} | {f2(r.mae_B2)} | "
                     f"{pct(r.mae_B2 / r.mae_A - 1)} | da {pct(r.ci_lo)} a {pct(r.ci_hi)} |")
    lines += [
        "",
        "L'intervallo viene da un bootstrap sulle partite del test: se contiene lo zero, la differenza",
        "non è distinguibile dal caso con una sola stagione di dati.",
        "",
        "## Quando la formazione cambia molto",
        "",
        f"In media una squadra cambia {mean_changes:.1f} titolari rispetto alla partita precedente".replace(".", ","),
        f"e nel {share_big:.0%} delle partite ne cambia almeno 4. Analisi diagnostica, non usata per",
        "scegliere il modello: MAE di A e B2 separando le partite con pochi o molti cambi, e correlazione",
        "tra lo scostamento dell'undici dal solito (rendimento atteso dei titolari di oggi rispetto a",
        "quelli della partita precedente) e l'errore del modello di squadra.",
        "",
        "| Statistica | ≤ 3 cambi: A | ≤ 3 cambi: B2 | ≥ 4 cambi: A | ≥ 4 cambi: B2 | Correlazione scostamento-errore |",
        "|---|---|---|---|---|---|",
    ]
    for r in res.itertuples():
        corr = "–" if np.isnan(r.corr) else f"{r.corr:+.3f}".replace(".", ",")
        lines.append(f"| {LABELS[r.stat]} | {f2(r.few_A)} | {f2(r.few_B2)} ({pct(r.few_B2 / r.few_A - 1)}) | "
                     f"{f2(r.many_A)} | {f2(r.many_B2)} ({pct(r.many_B2 / r.many_A - 1)}) | {corr} |")
    lines += [
        "",
        "## Controllo dei dati",
        "",
        "Medie per squadra e partita nel 2017-18: eventi Wyscout contro football-data.co.uk.",
        "",
        "| Statistica | Wyscout | football-data |",
        "|---|---|---|",
    ]
    for r in sanity.itertuples():
        lines.append(f"| {LABELS[r.stat]} | {f2(r.wyscout)} | {f2(r.football_data)} |")
    lines += [
        "",
        "Dati: Pappalardo et al. (2019), *A public data set of spatio-temporal match events in soccer",
        "competitions*, Scientific Data 6:236, licenza CC BY 4.0",
        "([figshare](https://figshare.com/collections/Soccer_match_event_dataset/4415000)).",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    run()
    print(f"Report scritto in {REPORT.relative_to(ROOT)}")
