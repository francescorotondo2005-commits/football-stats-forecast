"""Test sulla distribuzione e sugli intervalli."""

import numpy as np
import pytest
from scipy import stats

from footstats.models import MAX_K, CountDistribution, fit_dispersion


def test_dispersion_is_recovered_on_simulated_data():
    rng = np.random.default_rng(1)
    mu = rng.uniform(5, 20, 20000)
    k_true = 15.0
    y = stats.nbinom(k_true, k_true / (k_true + mu)).rvs(random_state=rng)
    assert fit_dispersion(y, mu) == pytest.approx(k_true, rel=0.15)


def test_poisson_data_gives_poisson():
    rng = np.random.default_rng(2)
    mu = rng.uniform(1, 4, 20000)
    y = rng.poisson(mu)
    assert fit_dispersion(y, mu) > 200  # nessuna sovra-dispersione visibile


@pytest.mark.parametrize("k", [10.0, MAX_K])
def test_interval_coverage_matches_declared_probability(k):
    rng = np.random.default_rng(3)
    dist = CountDistribution(k)
    mu = rng.uniform(3, 15, 50000)
    y = dist._dist(mu).rvs(random_state=rng)
    lo, hi = dist.interval(mu, 0.8)
    covered = np.mean((y >= lo) & (y <= hi))
    declared = np.mean(dist.prob_between(mu, lo, hi))
    assert declared >= 0.8
    assert covered == pytest.approx(declared, abs=0.01)


def test_wider_level_gives_wider_interval():
    dist = CountDistribution(20.0)
    lo50, hi50 = dist.interval(np.array([14.0]), 0.5)
    lo90, hi90 = dist.interval(np.array([14.0]), 0.9)
    assert lo90[0] <= lo50[0] and hi90[0] >= hi50[0]
    assert (hi90 - lo90)[0] > (hi50 - lo50)[0]
