"""Modelli: valore atteso (media) + distribuzione di conteggio per gli intervalli.

Idea: un modello stima mu = valore atteso della statistica (es. 15,8 tiri).
Attorno a mu si mette una distribuzione Binomiale Negativa, con varianza
    var = mu + mu^2 / k
dove k (dispersione) è stimato per ogni statistica su dati mai visti in training.
Se i dati non mostrano sovra-dispersione si usa la Poisson (k -> infinito).
L'intervallo all'80% va dal 10° al 90° percentile di quella distribuzione.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

MAX_K = 1e4  # oltre questo valore la Binomiale Negativa è di fatto una Poisson


# ───────────────────────── modelli per la media ─────────────────────────

class LeagueAverage:
    """Baseline banale: la media del campionato, uguale per tutti."""

    name = "media campionato"

    def __init__(self, stat: str):
        self.stat = stat

    def fit(self, X, y, sample_weight=None):
        return self

    def predict(self, X):
        return X[f"lg_{self.stat}"].to_numpy(dtype=float)


class RollingAverage:
    """Baseline ragionevole: metà attacco della squadra + metà difesa dell'avversario."""

    name = "media mobile"

    def __init__(self, stat: str):
        self.stat = stat

    def fit(self, X, y, sample_weight=None):
        return self

    def predict(self, X):
        s = self.stat
        return 0.5 * (X[f"t_{s}_for_l"].to_numpy(float) + X[f"o_{s}_against_l"].to_numpy(float))


def _log_rates(X):
    """Log delle feature di frequenza: il modello di Poisson è moltiplicativo."""
    X = X.copy()
    for c in X.columns:
        if c.startswith(("t_", "o_", "lg_")):
            X[c] = np.log(X[c].clip(lower=0) + 0.5)
        elif c.endswith("n_prev"):
            X[c] = np.log1p(X[c].clip(upper=100))
    return X


class PoissonGLM:
    """Regressione di Poisson (modello lineare generalizzato, link logaritmico)."""

    name = "GLM Poisson"

    def __init__(self, alpha: float = 1e-3):
        self.alpha = alpha
        self.model = make_pipeline(
            FunctionTransformer(_log_rates),
            StandardScaler(),
            PoissonRegressor(alpha=alpha, max_iter=2000),
        )

    def fit(self, X, y, sample_weight=None):
        self.model.fit(X, y, poissonregressor__sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.model.predict(X)


class PoissonGBM:
    """Gradient boosting con perdita di Poisson (cattura interazioni non lineari)."""

    name = "Gradient Boosting"

    def __init__(self, max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200):
        self.params = dict(max_iter=max_iter, learning_rate=learning_rate,
                           max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf)
        self.model = HistGradientBoostingRegressor(
            loss="poisson", l2_regularization=1.0, early_stopping=False, random_state=0, **self.params
        )

    def fit(self, X, y, sample_weight=None):
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.model.predict(X)


class Ensemble:
    """Media delle previsioni di GLM e gradient boosting.

    I due modelli sbagliano in modo diverso (il GLM è più stabile, il boosting coglie le
    interazioni): la media dei due è spesso migliore di ciascuno preso da solo.
    """

    name = "Ensemble"

    def __init__(self, glm: PoissonGLM, gbm: PoissonGBM, w_glm: float = 0.5):
        self.glm, self.gbm, self.w_glm = glm, gbm, w_glm
        self.params = {"glm_alpha": glm.alpha, **{f"gbm_{k}": v for k, v in gbm.params.items()}, "w_glm": w_glm}

    def fit(self, X, y, sample_weight=None):
        self.glm.fit(X, y, sample_weight)
        self.gbm.fit(X, y, sample_weight)
        return self

    def predict(self, X):
        return self.w_glm * self.glm.predict(X) + (1 - self.w_glm) * self.gbm.predict(X)


# ───────────────────────── distribuzione e intervalli ─────────────────────────

def _nb_params(mu, k):
    """Parametri scipy (n, p) della Binomiale Negativa con media mu e dispersione k."""
    mu = np.maximum(np.asarray(mu, float), 1e-6)
    return k, k / (k + mu)


def fit_dispersion(y, mu) -> float:
    """Stima k per massima verosimiglianza. Restituisce MAX_K se non c'è sovra-dispersione."""
    y = np.asarray(y, float)
    mu = np.maximum(np.asarray(mu, float), 1e-6)

    def nll(log_k):
        n, p = _nb_params(mu, np.exp(log_k))
        return -stats.nbinom.logpmf(y, n, p).sum()

    res = optimize.minimize_scalar(nll, bounds=(np.log(0.1), np.log(MAX_K)), method="bounded")
    return float(min(np.exp(res.x), MAX_K))


@dataclass
class CountDistribution:
    k: float  # dispersione; MAX_K = Poisson

    @property
    def is_poisson(self) -> bool:
        return self.k >= MAX_K * 0.99

    def _dist(self, mu):
        mu = np.maximum(np.asarray(mu, float), 1e-6)
        if self.is_poisson:
            return stats.poisson(mu)
        n, p = _nb_params(mu, self.k)
        return stats.nbinom(n, p)

    def interval(self, mu, level: float):
        """Intervallo centrale [lo, hi] (estremi inclusi) con probabilità >= level."""
        tail = (1 - level) / 2
        d = self._dist(mu)
        lo = d.ppf(tail)
        hi = d.ppf(1 - tail)
        return np.asarray(lo, float), np.asarray(hi, float)

    def prob_between(self, mu, lo, hi):
        """Probabilità che il modello assegna all'intervallo [lo, hi] (estremi inclusi).

        Con valori interi un intervallo "all'80%" contiene di solito più dell'80% di
        probabilità: è questa la copertura che il modello dichiara davvero.
        """
        d = self._dist(mu)
        return np.asarray(d.cdf(hi) - d.cdf(np.asarray(lo) - 1), float)

    def log_score(self, y, mu):
        """Log-verosimiglianza media: quanto la distribuzione prevista 'credeva' nel valore vero."""
        return float(np.mean(self._dist(mu).logpmf(np.asarray(y, float))))


@dataclass
class StatModel:
    """Modello completo per una statistica: media + distribuzione."""

    stat: str
    mean_model: object
    dist: CountDistribution
    feature_cols: list = field(default_factory=list)

    def predict_mean(self, X):
        return self.mean_model.predict(X[self.feature_cols])

    def predict_interval(self, X, level):
        return self.dist.interval(self.predict_mean(X), level)
