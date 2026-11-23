"""Addestramento, selezione del modello e valutazione su stagioni mai viste.

Protocollo (tutto in ordine temporale, niente partite "dal futuro"):
  1. training     2014-15 … 2022-23  -> si addestrano i candidati
  2. validazione  2023-24            -> si sceglie il modello e si stima la dispersione k
  3. test         2024-25, 2025-26   -> si riaddestra su training+validazione e si misura,
                                        una sola volta, su partite mai usate prima
  4. modello finale                  -> riaddestrato su tutto, usato da `predict`
"""

from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from .config import (DEFAULT_LEVEL, MODELS_DIR, REPORTS_DIR, STAT_LABELS_IT, STATS,
                     TEST_SEASONS, TRAIN_LAST_SEASON, VALID_SEASON)
from .data import load_matches, to_team_rows
from .features import build_features, feature_columns
from .models import (CountDistribution, Ensemble, LeagueAverage, PoissonGBM, PoissonGLM,
                     RollingAverage, StatModel, fit_dispersion)

LEVELS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)

# Griglie estese dopo aver visto che la validazione sceglieva sempre i valori al bordo
# (alpha più alto per il GLM, meno alberi per il boosting).
GLM_GRID = [dict(alpha=a) for a in (1e-4, 1e-3, 1e-2, 1e-1, 3e-1)]
GBM_GRID = [
    dict(max_iter=it, learning_rate=0.05, max_leaf_nodes=nl, min_samples_leaf=ml)
    for it in (100, 200) for nl in (7, 15) for ml in (100, 400)
]


def load_dataset() -> pd.DataFrame:
    df = build_features(to_team_rows(load_matches()))
    lg_cols = [f"lg_{s}" for s in STATS]
    return df.dropna(subset=lg_cols).reset_index(drop=True)


def _families(stat):
    """Famiglie di modelli candidati: (nome, lista di costruttori)."""
    return {
        "media campionato": [lambda: LeagueAverage(stat)],
        "media mobile": [lambda: RollingAverage(stat)],
        "GLM Poisson": [lambda p=p: PoissonGLM(**p) for p in GLM_GRID],
        "Gradient Boosting": [lambda p=p: PoissonGBM(**p) for p in GBM_GRID],
    }


def _metrics(y, mu, dist: CountDistribution) -> dict:
    out = {
        "mae": float(mean_absolute_error(y, mu)),
        "log_score": dist.log_score(y, mu),
    }
    for lvl in LEVELS:
        lo, hi = dist.interval(mu, lvl)
        out[f"cov_{int(lvl * 100)}"] = float(np.mean((y >= lo) & (y <= hi)))
        out[f"claimed_{int(lvl * 100)}"] = float(np.mean(dist.prob_between(mu, lo, hi)))
        out[f"width_{int(lvl * 100)}"] = float(np.mean(hi - lo))
    return out


def run(verbose: bool = True) -> dict:
    t0 = time.time()
    log = print if verbose else (lambda *a, **k: None)
    df = load_dataset()
    train = df[df.season <= TRAIN_LAST_SEASON]
    valid = df[df.season == VALID_SEASON]
    trval = df[df.season <= VALID_SEASON]
    test = df[df.season.isin(TEST_SEASONS)]
    log(f"Righe squadra-partita: training {len(train)}, validazione {len(valid)}, test {len(test)}")

    results, final_models = {}, {}
    for stat in STATS:
        cols = feature_columns(stat)
        y_tr, y_va, y_tv, y_te = (d[f"{stat}_for"].to_numpy(float) for d in (train, valid, trval, test))
        stat_res = {"families": {}}
        best_overall, chosen_build, test_mu = None, {}, {}

        def evaluate_family(fam, builders):
            nonlocal best_overall
            # 1) scelta della configurazione migliore della famiglia sulla validazione
            best = None
            for build in builders:
                m = build().fit(train[cols], y_tr)
                mu_va = m.predict(valid[cols])
                dist = CountDistribution(fit_dispersion(y_va, mu_va))
                score = dist.log_score(y_va, mu_va)
                if best is None or score > best[0]:
                    best = (score, build, dist)
            val_score, build, dist = best
            chosen_build[fam] = build

            # 2) test: riaddestro su training+validazione, valuto su stagioni mai viste
            m = build().fit(trval[cols], y_tv)
            mu_te = test_mu[fam] = m.predict(test[cols])
            test_metrics = _metrics(y_te, mu_te, dist)
            params = getattr(m, "params", None) or ({"alpha": m.alpha} if hasattr(m, "alpha") else {})
            stat_res["families"][fam] = {
                "val_log_score": val_score, "k": dist.k, "params": params, "test": test_metrics,
            }
            log(f"  {STAT_LABELS_IT[stat]:<14} {fam:<18} val log-score {val_score:7.4f}  "
                f"test MAE {test_metrics['mae']:.3f}  copertura 80% {test_metrics['cov_80']:.1%}")
            if best_overall is None or val_score > best_overall[0]:
                best_overall = (val_score, fam, build, dist)

        for fam, builders in _families(stat).items():
            evaluate_family(fam, builders)
        # ensemble: media del miglior GLM e del miglior boosting scelti sopra
        glm_b, gbm_b = chosen_build["GLM Poisson"], chosen_build["Gradient Boosting"]
        evaluate_family("Ensemble", [lambda g=glm_b, b=gbm_b: Ensemble(g(), b())])

        # 3) modello finale: famiglia scelta in validazione, riaddestrata su tutti i dati
        _, fam, build, dist = best_overall
        y_all = df[f"{stat}_for"].to_numpy(float)
        final = StatModel(stat, build().fit(df[cols], y_all), dist, cols)
        final_models[stat] = final
        stat_res["chosen"] = fam
        results[stat] = stat_res

        # serie di test del modello scelto, per i grafici
        mu_te = test_mu[fam]
        lo, hi = dist.interval(mu_te, DEFAULT_LEVEL)
        stat_res["_test_frame"] = test[["league", "season", "date", "team", "opponent", "is_home"]].assign(
            y=y_te, mu=mu_te, lo=lo, hi=hi)

    MODELS_DIR.mkdir(exist_ok=True)
    last_dates = df.groupby("league")["date"].max().dt.strftime("%Y-%m-%d").to_dict()
    joblib.dump({"models": final_models, "last_dates": last_dates}, MODELS_DIR / "footstats.joblib")
    log(f"Modelli salvati in {MODELS_DIR / 'footstats.joblib'}  ({time.time() - t0:.0f}s)")
    return results


def save_report(results: dict) -> None:
    """Scrive reports/results.json e i grafici."""
    from .report import write_report

    REPORTS_DIR.mkdir(exist_ok=True)
    clean = {s: {k: v for k, v in r.items() if not k.startswith("_")} for s, r in results.items()}
    # encoding esplicito: su Windows il default non è UTF-8 e le lettere accentate si rovinano
    (REPORTS_DIR / "results.json").write_text(
        json.dumps(clean, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    write_report(results)
