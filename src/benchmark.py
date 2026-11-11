"""Repeated-seed experiments and deterministic Gamma references."""

from time import perf_counter

import numpy as np
import pandas as pd
import torch
from scipy.integrate import quad
from scipy.special import expit
from scipy.stats import norm, t as student_t

from . import (generate_normal_vect, vanilla_call_analytics, vanilla_call_mc_fd,
               vanilla_call_mc_aad_smooth, vanilla_call_mc_malliavin)


def expected_gamma(method, maturity, parameter=None, spot=100., strike=100.,
                   sigma=.1, rate=.05):
    """Expectation of each estimator, without Monte Carlo sampling error."""
    if method == "LR":
        return vanilla_call_analytics(spot, strike, sigma, rate, maturity)[3]
    if method == "FD":
        h = parameter * spot
        prices = [vanilla_call_analytics(s, strike, sigma, rate, maturity)[0]
                  for s in (spot - h, spot, spot + h)]
        return (prices[2] - 2 * prices[1] + prices[0]) / h**2
    if method != "AAD":
        raise ValueError(method)
    beta = parameter

    def integrand(u):
        # u = beta * (S_T - K) resolves the narrow smoothing region.
        terminal = strike + u / beta
        if terminal <= 0:
            return 0.
        z = (np.log(terminal / spot) - (rate - .5 * sigma**2) * maturity)
        z /= sigma * np.sqrt(maturity)
        density = norm.pdf(z) / (terminal * sigma * np.sqrt(maturity))
        return (np.exp(-rate * maturity) * (terminal / spot)**2
                * expit(u) * expit(-u) * density)

    # PyTorch Softplus uses its linear branch above beta*x = 20.
    lower = max(-100., -beta * strike)
    return quad(integrand, lower, 20., points=[0.], epsabs=1e-12)[0]


def estimate(z, method, maturity, parameter=None, spot=100.):
    spot = torch.as_tensor(spot, dtype=torch.float64).clone().requires_grad_(True)
    sigma = torch.tensor(.1, dtype=torch.float64, requires_grad=True)
    args = (z, spot, 100., sigma, .05, maturity)
    if method == "FD":
        return vanilla_call_mc_fd(*args, epsilon=parameter)[3]
    if method == "AAD":
        return vanilla_call_mc_aad_smooth(*args, beta=parameter)[3]
    if method == "LR":
        return vanilla_call_mc_malliavin(*args)[3]
    raise ValueError(method)


def run_trials(configs, pair_counts, maturities, seeds):
    """Share draws across methods, with independent replicates across seeds."""
    rows = []
    for maturity in maturities:
        reference = expected_gamma("LR", maturity)
        for pairs in pair_counts:
            for seed in seeds:
                z = generate_normal_vect(pairs, seed=seed)
                for label, method, parameter in configs:
                    gamma = float(np.asarray(estimate(z, method, maturity, parameter)).item())
                    rows.append((maturity, pairs, seed, label, gamma, gamma-reference))
    return pd.DataFrame(rows, columns=["T", "pairs", "seed", "method", "gamma", "error"])


def summarize(trials):
    """Bias CI across independent seeds; percentile bootstrap CI for RMSE."""
    rows = []
    rng = np.random.default_rng(1729)
    for (maturity, pairs, method), group in trials.groupby(["T", "pairs", "method"], sort=True):
        errors = group.error.to_numpy()
        n = len(errors)
        sd = errors.std(ddof=1)
        half_width = student_t.ppf(.975, n-1) * sd / np.sqrt(n)
        samples = rng.choice(errors, size=(2000, n), replace=True)
        lo, hi = np.quantile(np.sqrt(np.mean(samples**2, axis=1)), [.025, .975])
        rows.append((maturity, pairs, method, n, errors.mean(), sd,
                     np.sqrt(np.mean(errors**2)), errors.mean()-half_width,
                     errors.mean()+half_width, lo, hi))
    return pd.DataFrame(rows, columns=["T", "pairs", "method", "replicates", "bias", "sd",
                                      "rmse", "bias_ci_low", "bias_ci_high", "rmse_ci_low", "rmse_ci_high"])


def benchmark_times(configs, pair_counts, maturities, repeats=7, batch_sizes=(1,)):
    """Warm up each configuration; time calls in shuffled order, excluding RNG."""
    rows = []
    rng = np.random.default_rng(2026)
    for maturity in maturities:
        for pairs in pair_counts:
            z = generate_normal_vect(pairs, seed=99999)
            for batch in batch_sizes:
                spot = 100. if batch == 1 else np.linspace(80., 120., batch)
                for _, method, parameter in configs:
                    estimate(z, method, maturity, parameter, spot)
                for repeat in range(repeats):
                    for index in rng.permutation(len(configs)):
                        label, method, parameter = configs[index]
                        start = perf_counter()
                        estimate(z, method, maturity, parameter, spot)
                        elapsed = 1000 * (perf_counter()-start)
                        rows.append((maturity, pairs, batch, repeat, label, elapsed))
    return pd.DataFrame(rows, columns=["T", "pairs", "batch", "repeat", "method", "ms"])
