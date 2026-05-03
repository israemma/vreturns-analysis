#!/usr/bin/env python3
"""
Comparison: Exp(refl)+GΓ+GPD  vs  Pareto+GΓ+Pareto
==========================================================
Model A: Exp(refl) + GΓ + GPD       [exp_gg_gpd]
Model B: Pareto    + GΓ + Pareto    [pareto_gg_pareto]

Bâcă & Vernic Philosophy (Axioms 2024, 13, 473):
  - C0+C1 splices at θ_H and C0+C1 at θ_L.
  - Exp Model: fixed ξ_L=0, σ_L = 1/D_body(θ_L). No free parameters in L.
  - Par Model: α_L = 1 + θ_L·D_body(θ_L). No free parameters in L.
  - Both models have 2 free parameters: ξ_R and σ_R.

Date ranges (Table I):
  DJIA:   1992-01-02 → 2023-12-29
  DAX:    1992-01-02 → 2023-12-29
  IPC:    1992-01-02 → 2023-12-29
  Nikkei: 1992-01-06 → 2023-12-29

Outputs:
  plots_compare_ExpvsPar_v2/
    metrics.csv
    bootstrap_details.csv
    summary_pass_rate.txt
    plot_ccdf_{ASSET}.png
    plot_hist_{ASSET}.png
    plot_qq_{ASSET}.png
"""

import os
import sys
import glob
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize, brentq
from scipy.special import gammainc, gammaincinv, gammaln
from scipy.stats import kstest, norm
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable=None, total=None, desc=None, **kwargs):
        if iterable is not None:
            return iterable
        class _NullBar:
            def update(self, n=1): pass
            def close(self): pass
            def set_description(self, desc): pass
            def __enter__(self): return self
            def __exit__(self, *a): pass
        return _NullBar()

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 13,
    'figure.titlesize': 20,
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
})

# ═══════════════════════════════════════════════════════════════════
# CONFIG GLOBAL
# ═══════════════════════════════════════════════════════════════════
DATA_DIR = 'financial_data_v2'
OUT_DIR = 'compare_composite_models_exp_vs_pareto'
os.makedirs(OUT_DIR, exist_ok=True)

NUM_WORKERS = 30

# 4 indices
ASSETS_ALLOWED = ['DJIA', 'DAX', 'IPC', 'Nikkei']

# ─── Sweep of θ_H (percentile) — FINER GRID ─────────────────
# ─── Sweep of θ_H (percentile) ────────────────────────────────────
# Reduced grid (6 pts) including standards [92, 94, 95, 96]
# plus critical points where DAX passes: 89.0 and 96.5.
THETA_H_PCT_GRID = [89.0, 92.0, 94.0, 95.0, 96.0, 96.5]

# ─── Sweep of θ_L ───────────────────────────────────────────────
# Standard grid plus points 13% and 14% that optimize DAX.
THETA_LOW_PCT_GRID = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0,
                      10.0, 12.0, 13.0, 14.0, 15.0, 18.0, 20.0, 25.0]
DISCARD_PCT = 2.0

def min_n_L_dynamic(n_total):
    if n_total < 3500:
        return 50
    elif n_total < 6000:
        return 80
    else:
        return 100

# ─── GG Body parameters ────────────────────────
D_MIN_HARD = 0.15
D_MIN_SOFT = 0.22
D_PENALTY_LAMBDA = 0.5

# Multi-start
D_MULTISTART = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0,
                1.2, 1.5, 2.0, 2.5, 3.0]
XI_MULTISTART_FACTORS = [0.5, 0.7, 1.0, 1.4, 2.0]
SIGMA_MULTISTART_FACTORS = [0.3, 0.5, 1.0, 2.0, 3.0]
XI_MULTISTART_FIXED = [0.12, 0.2, 0.3, 0.4, 0.5]

# Multi-start for bootstrap 
D_MULTISTART_BOOT = [0.3, 0.7, 1.0, 1.5]
XI_MULTISTART_FIXED_BOOT = [0.2, 0.35]
SIGMA_MULTISTART_FACTORS_BOOT = [0.5, 1.0, 2.0]

# ─── Multi-pass refit ───────────────────────────────────────────

MAX_REFIT_ITERS = 6
MAX_REFIT_ITERS_BOOT = 1
REFIT_KS3R_TOL = 1.02

# ─── Composite sweep ────────────────────────────────
LAMBDA_BODY = 0.5
LAMBDA_LEFT = 0.15

# ─── Bootstrap Lilliefors ────────────────────────────
DO_LILLIEFORS = True
N_BOOTSTRAP = 1000
BOOTSTRAP_MAX_N = 10000

# ─── Colors  ────────────────────────────
C_REGION_L = '#1f77b4'
C_REGION_B = '#2ca02c'
C_REGION_T = '#ff7f0e'
C_EMPIRICAL = '#000000'
LINESTYLE_GPD = '-'
LINESTYLE_PAR = '--'
MARKER_GPD = 'o'
MARKER_PAR = 'x'

CASE_EXP = {'key': 'exp_gg_gpd', 'left': 'exp_refl', 'body': 'gg',
            'right': 'gpd', 'label': 'Exp(refl) + GΓ + GPD',
            'label_short': 'Exp-GΓ-GPD',
            'linestyle': LINESTYLE_GPD, 'marker': MARKER_GPD}
CASE_PAR = {'key': 'pareto_gg_pareto', 'left': 'pareto', 'body': 'gg',
            'right': 'pareto_right', 'label': 'Pareto + GΓ + Pareto',
            'label_short': 'Par-GΓ-Par',
            'linestyle': LINESTYLE_PAR, 'marker': MARKER_PAR}
CASES = [CASE_EXP, CASE_PAR]
CASE_BY_KEY = {c['key']: c for c in CASES}


# ═══════════════════════════════════════════════════════════════════
# BASE DENSITIES
# ═══════════════════════════════════════════════════════════════════
def gg_pdf(x, a, b, d):
    x = np.asarray(x, dtype=float)
    z = x / b
    log_pdf = (np.log(d) - np.log(b) - gammaln(a/d)
               + (a - 1.0) * np.log(z) - z ** d)
    return np.exp(log_pdf)

def gg_cdf(x, a, b, d):
    x = np.asarray(x, dtype=float)
    z = x / b
    return gammainc(a / d, z ** d)

def gg_ppf(p, a, b, d):
    p = np.clip(np.asarray(p, dtype=float), 1e-15, 1.0 - 1e-15)
    q = gammaincinv(a / d, p)
    return b * q ** (1.0 / d)

def gg_pdf_at_point(x, a, b, d):
    if x <= 0:
        return 0.0
    z = x / b
    log_pdf = (np.log(d) - np.log(b) - gammaln(a/d)
               + (a - 1.0) * np.log(z) - z ** d)
    return float(np.exp(log_pdf))

def gpd_pdf(x, theta, xi, sigma):
    x = np.asarray(x, dtype=float)
    z = (x - theta) / sigma
    if abs(xi) < 1e-10:
        return (1.0 / sigma) * np.exp(-z)
    inner = 1.0 + xi * z
    inner = np.maximum(inner, 1e-300)
    return (1.0 / sigma) * inner ** (-1.0 / xi - 1.0)

def gpd_cdf(x, theta, xi, sigma):
    x = np.asarray(x, dtype=float)
    z = (x - theta) / sigma
    if abs(xi) < 1e-10:
        return 1.0 - np.exp(-z)
    inner = 1.0 + xi * z
    inner = np.maximum(inner, 1e-300)
    return 1.0 - inner ** (-1.0 / xi)

def gpd_ppf(p, theta, xi, sigma):
    if abs(xi) < 1e-10:
        return theta + sigma * (-np.log(1.0 - p))
    return theta + (sigma / xi) * ((1.0 - p) ** (-xi) - 1.0)


# ═══════════════════════════════════════════════════════════════════
# LEFT TAIL L
# ═══════════════════════════════════════════════════════════════════
def pareto_L_pdf(x, params, x_min, theta_L):
    alpha = params['alpha_pl']
    denom = theta_L ** alpha - x_min ** alpha
    if denom <= 0:
        return np.zeros_like(x)
    return alpha * x ** (alpha - 1.0) / denom

def pareto_L_cdf(x, params, x_min, theta_L):
    alpha = params['alpha_pl']
    denom = theta_L ** alpha - x_min ** alpha
    if denom <= 0:
        return np.full_like(x, np.nan)
    return (x ** alpha - x_min ** alpha) / denom

def pareto_L_pdf_at(x0, params, x_min, theta_L):
    alpha = params['alpha_pl']
    denom = theta_L ** alpha - x_min ** alpha
    if denom <= 0:
        return 0.0
    return float(alpha * x0 ** (alpha - 1.0) / denom)


def fit_L_pareto(obs_discarded, theta_L, body_params=None):
    """Left Pareto with C1 at θ_L (identical to original)."""
    if body_params is None:
        raise ValueError("fit_L_pareto requires body_params (C1 at θ_L)")

    n_disc = len(obs_discarded)
    if n_disc < 20:
        return None
    x_min = float(obs_discarded.min())
    if x_min <= 0 or theta_L <= x_min:
        return None

    D_body = body_logderiv_at(theta_L, body_params)
    if not np.isfinite(D_body):
        return None
    alpha_hat = 1.0 + theta_L * D_body

    if alpha_hat <= 1e-6 or alpha_hat > 50.0:
        return None

    try:
        u = pareto_L_cdf(obs_discarded, {'alpha_pl': alpha_hat}, x_min, theta_L)
        u = np.clip(u, 1e-12, 1.0 - 1e-12)
        ks_stat, ks_p = kstest(u, 'uniform')
    except Exception:
        ks_stat, ks_p = np.nan, np.nan

    try:
        log_min = np.log10(x_min)
        log_max = np.log10(theta_L)
        if log_max - log_min >= 0.3:
            bin_edges = np.logspace(log_min, log_max, 41)
            obs_counts, _ = np.histogram(obs_discarded, bins=bin_edges)
            F_lo = pareto_L_cdf(bin_edges[:-1], {'alpha_pl': alpha_hat},
                                x_min, theta_L)
            F_hi = pareto_L_cdf(bin_edges[1:], {'alpha_pl': alpha_hat},
                                x_min, theta_L)
            exp_c = n_disc * (F_hi - F_lo)
            rmse_hat = float(np.sqrt(np.mean(
                (np.log1p(obs_counts) - np.log1p(exp_c)) ** 2)))
        else:
            rmse_hat = np.nan
    except Exception:
        rmse_hat = np.nan

    return {
        'left_kind': 'pareto',
        'alpha_pl': float(alpha_hat),
        'alpha_pl_se': np.nan,
        'x_min_pl': float(x_min),
        'threshold_pl': float(theta_L),
        'n_disc_pl': int(n_disc),
        'ks_stat_pl': float(ks_stat),
        'ks_p_pl': float(ks_p),
        'rmse_log_pl': rmse_hat,
        'c1_enforced_pl': True,
    }


def gpd_refl_L_pdf(x, params, x_min, theta_L):
    xi = params['xi_L']
    sigma = params['sigma_L']
    Z = gpd_cdf(np.array([theta_L - x_min]), 0.0, xi, sigma)[0]
    if Z <= 0:
        return np.zeros_like(x)
    y = theta_L - x
    y_safe = np.clip(y, 0.0, theta_L - x_min)
    return gpd_pdf(y_safe, 0.0, xi, sigma) / Z

def gpd_refl_L_cdf(x, params, x_min, theta_L):
    xi = params['xi_L']
    sigma = params['sigma_L']
    Z = gpd_cdf(np.array([theta_L - x_min]), 0.0, xi, sigma)[0]
    if Z <= 0:
        return np.full_like(x, np.nan)
    y = theta_L - x
    y_safe = np.clip(y, 0.0, theta_L - x_min)
    F_y = gpd_cdf(y_safe, 0.0, xi, sigma)
    return 1.0 - F_y / Z

def gpd_refl_L_pdf_at(x0, params, x_min, theta_L):
    xi = params['xi_L']
    sigma = params['sigma_L']
    Z = gpd_cdf(np.array([theta_L - x_min]), 0.0, xi, sigma)[0]
    if Z <= 0:
        return 0.0
    y = max(0.0, theta_L - x0)
    return float(gpd_pdf(np.array([y]), 0.0, xi, sigma)[0] / Z)


def fit_L_gpd_refl(obs_discarded, theta_L, body_params=None):
    """Reflected left GPD with C1 at θ_L (identical to original)."""
    if body_params is None:
        raise ValueError("fit_L_gpd_refl requires body_params (C1 at θ_L)")

    n_disc = len(obs_discarded)
    if n_disc < 20:
        return None
    x_min = float(obs_discarded.min())
    if x_min <= 0 or theta_L <= x_min:
        return None

    D_body = body_logderiv_at(theta_L, body_params)
    if not np.isfinite(D_body) or D_body <= 1e-10:
        return None

    def sigma_from_xi(xi_L):
        if 1.0 + xi_L <= 0:
            return None
        return (1.0 + xi_L) / D_body

    def neg_log_lik(xi_L):
        sigma_L = sigma_from_xi(xi_L)
        if sigma_L is None or sigma_L <= 0:
            return 1e10
        y = theta_L - obs_discarded
        y_max = theta_L - x_min
        if xi_L < 0:
            upper = -sigma_L / xi_L
            if y_max >= upper:
                return 1e10
        try:
            z = y / sigma_L
            if abs(xi_L) < 1e-10:
                log_f_y = -np.log(sigma_L) - z
            else:
                inner = 1.0 + xi_L * z
                if np.any(inner <= 0):
                    return 1e10
                log_f_y = -np.log(sigma_L) + (-1.0 / xi_L - 1.0) * np.log(inner)
            z_max = y_max / sigma_L
            if abs(xi_L) < 1e-10:
                log_Z = np.log1p(-np.exp(-z_max))
            else:
                inner_max = 1.0 + xi_L * z_max
                if inner_max <= 0:
                    return 1e10
                Z = 1.0 - inner_max ** (-1.0 / xi_L)
                if Z <= 0 or Z > 1:
                    return 1e10
                log_Z = np.log(Z)
            return -float(np.sum(log_f_y) - n_disc * log_Z)
        except Exception:
            return 1e10

    best_xi = None
    best_nll = np.inf
    for xi_try in [0.05, 0.15, 0.3, 0.5, 0.8, 1.2, 2.0, -0.1, -0.3]:
        try:
            res = minimize(neg_log_lik, [xi_try], method='Nelder-Mead',
                           options={'xatol': 1e-8, 'fatol': 1e-10,
                                    'maxiter': 2000})
            if res.success and res.fun < best_nll:
                best_nll = res.fun
                best_xi = float(res.x[0])
        except Exception:
            continue
    if best_xi is None:
        return None
    if best_xi <= -0.95 or best_xi > 10.0:
        return None

    sigma_hat = sigma_from_xi(best_xi)
    if sigma_hat is None or sigma_hat <= 0:
        return None

    try:
        u = gpd_refl_L_cdf(obs_discarded,
                           {'xi_L': best_xi, 'sigma_L': sigma_hat},
                           x_min, theta_L)
        u = np.clip(u, 1e-12, 1.0 - 1e-12)
        ks_stat, ks_p = kstest(u, 'uniform')
    except Exception:
        ks_stat, ks_p = np.nan, np.nan

    try:
        log_min = np.log10(x_min)
        log_max = np.log10(theta_L)
        if log_max - log_min >= 0.3:
            bin_edges = np.logspace(log_min, log_max, 41)
            obs_counts, _ = np.histogram(obs_discarded, bins=bin_edges)
            F_lo = gpd_refl_L_cdf(bin_edges[:-1],
                                  {'xi_L': best_xi, 'sigma_L': sigma_hat},
                                  x_min, theta_L)
            F_hi = gpd_refl_L_cdf(bin_edges[1:],
                                  {'xi_L': best_xi, 'sigma_L': sigma_hat},
                                  x_min, theta_L)
            exp_c = n_disc * (F_hi - F_lo)
            rmse_hat = float(np.sqrt(np.mean(
                (np.log1p(obs_counts) - np.log1p(exp_c)) ** 2)))
        else:
            rmse_hat = np.nan
    except Exception:
        rmse_hat = np.nan

    return {
        'left_kind': 'gpd_refl',
        'xi_L': float(best_xi),
        'sigma_L': float(sigma_hat),
        'x_min_pl': float(x_min),
        'threshold_pl': float(theta_L),
        'n_disc_pl': int(n_disc),
        'ks_stat_pl': float(ks_stat),
        'ks_p_pl': float(ks_p),
        'rmse_log_pl': rmse_hat,
        'c1_enforced_pl': True,
        'alpha_pl': np.nan, 'alpha_pl_se': np.nan,
    }


# ─── Exponencial reflejada (ξ_L = 0 fijo) ────────────────────────
# The exponential is the special case of GPD with ξ=0.
# Under C1 at θ_L: σ_L = (1 + ξ_L) / D_body(θ_L) = 1 / D_body(θ_L)
# No free parameters (analogous to Pareto).
def exp_refl_L_pdf(x, params, x_min, theta_L):
    """PDF of the reflected exponential truncated in [x_min, θ_L]."""
    sigma = params['sigma_L']
    y = theta_L - np.asarray(x, dtype=float)
    y_safe = np.clip(y, 0.0, theta_L - x_min)
    y_max = theta_L - x_min
    Z = 1.0 - np.exp(-y_max / sigma)
    if Z <= 0:
        return np.zeros_like(x, dtype=float)
    return (1.0 / sigma) * np.exp(-y_safe / sigma) / Z


def exp_refl_L_cdf(x, params, x_min, theta_L):
    """CDF of the reflected exponential truncated in [x_min, θ_L]."""
    sigma = params['sigma_L']
    y = theta_L - np.asarray(x, dtype=float)
    y_safe = np.clip(y, 0.0, theta_L - x_min)
    y_max = theta_L - x_min
    Z = 1.0 - np.exp(-y_max / sigma)
    if Z <= 0:
        return np.full_like(x, np.nan, dtype=float)
    # F_Y(y) = (1 - exp(-y/σ)) / Z  (CDF of Y = θ_L - X)
    # F_X(x) = P(X ≤ x) = 1 - F_Y(θ_L - x) / Z ... simpler:
    # P(X ≤ x) = 1 - [1 - exp(-y/σ)] / Z  where y = θ_L - x
    F_y = 1.0 - np.exp(-y_safe / sigma)
    return 1.0 - F_y / Z


def exp_refl_L_pdf_at(x0, params, x_min, theta_L):
    """Pointwise PDF of the reflected exponential."""
    sigma = params['sigma_L']
    y = max(0.0, theta_L - x0)
    y_max = theta_L - x_min
    Z = 1.0 - np.exp(-y_max / sigma)
    if Z <= 0:
        return 0.0
    return float((1.0 / sigma) * np.exp(-y / sigma) / Z)


def fit_L_exp_refl(obs_discarded, theta_L, body_params=None):
    """Reflected left exponential with C1 at θ_L.

    No free parameters: σ_L = 1 / D_body(θ_L).
    Analogous to fit_L_pareto where α_L = 1 + θ_L·D_body(θ_L).
    """
    if body_params is None:
        raise ValueError("fit_L_exp_refl requires body_params (C1 at θ_L)")

    n_disc = len(obs_discarded)
    if n_disc < 20:
        return None
    x_min = float(obs_discarded.min())
    if x_min <= 0 or theta_L <= x_min:
        return None

    D_body = body_logderiv_at(theta_L, body_params)
    if not np.isfinite(D_body) or D_body <= 1e-10:
        return None

    # C1: σ_L = (1 + ξ_L) / D_body  with ξ_L = 0
    sigma_hat = 1.0 / D_body
    if sigma_hat <= 0:
        return None

    try:
        u = exp_refl_L_cdf(obs_discarded,
                           {'sigma_L': sigma_hat},
                           x_min, theta_L)
        u = np.clip(u, 1e-12, 1.0 - 1e-12)
        ks_stat, ks_p = kstest(u, 'uniform')
    except Exception:
        ks_stat, ks_p = np.nan, np.nan

    try:
        log_min = np.log10(x_min)
        log_max = np.log10(theta_L)
        if log_max - log_min >= 0.3:
            bin_edges = np.logspace(log_min, log_max, 41)
            obs_counts, _ = np.histogram(obs_discarded, bins=bin_edges)
            F_lo = exp_refl_L_cdf(bin_edges[:-1],
                                  {'sigma_L': sigma_hat},
                                  x_min, theta_L)
            F_hi = exp_refl_L_cdf(bin_edges[1:],
                                  {'sigma_L': sigma_hat},
                                  x_min, theta_L)
            exp_c = n_disc * (F_hi - F_lo)
            rmse_hat = float(np.sqrt(np.mean(
                (np.log1p(obs_counts) - np.log1p(exp_c)) ** 2)))
        else:
            rmse_hat = np.nan
    except Exception:
        rmse_hat = np.nan

    return {
        'left_kind': 'exp_refl',
        'xi_L': 0.0,
        'sigma_L': float(sigma_hat),
        'x_min_pl': float(x_min),
        'threshold_pl': float(theta_L),
        'n_disc_pl': int(n_disc),
        'ks_stat_pl': float(ks_stat),
        'ks_p_pl': float(ks_p),
        'rmse_log_pl': rmse_hat,
        'c1_enforced_pl': True,
        'alpha_pl': np.nan, 'alpha_pl_se': np.nan,
    }


def pdf_L(x, left_params, x_min, theta_L):
    if left_params['left_kind'] == 'pareto':
        return pareto_L_pdf(x, left_params, x_min, theta_L)
    elif left_params['left_kind'] == 'exp_refl':
        return exp_refl_L_pdf(x, left_params, x_min, theta_L)
    else:
        return gpd_refl_L_pdf(x, left_params, x_min, theta_L)

def cdf_L(x, left_params, x_min, theta_L):
    if left_params['left_kind'] == 'pareto':
        return pareto_L_cdf(x, left_params, x_min, theta_L)
    elif left_params['left_kind'] == 'exp_refl':
        return exp_refl_L_cdf(x, left_params, x_min, theta_L)
    else:
        return gpd_refl_L_cdf(x, left_params, x_min, theta_L)

def pdf_L_at(x0, left_params, x_min, theta_L):
    if left_params['left_kind'] == 'pareto':
        return pareto_L_pdf_at(x0, left_params, x_min, theta_L)
    elif left_params['left_kind'] == 'exp_refl':
        return exp_refl_L_pdf_at(x0, left_params, x_min, theta_L)
    else:
        return gpd_refl_L_pdf_at(x0, left_params, x_min, theta_L)

def fit_left(kind, obs_discarded, theta_L, body_params=None):
    if body_params is None:
        raise ValueError("fit_left requires body_params")
    if kind == 'pareto':
        return fit_L_pareto(obs_discarded, theta_L, body_params=body_params)
    elif kind == 'gpd_refl':
        return fit_L_gpd_refl(obs_discarded, theta_L, body_params=body_params)
    elif kind == 'exp_refl':
        return fit_L_exp_refl(obs_discarded, theta_L, body_params=body_params)
    else:
        raise ValueError(f"Unknown left kind: {kind}")


# ═══════════════════════════════════════════════════════════════════
# BODY + RIGHT TAIL
# ═══════════════════════════════════════════════════════════════════
def solve_body_params_gg(xi_R, sigma_R, phi, theta_H, d,
                         a_bounds=(1.001, 1500)):
    if phi <= 0 or phi >= 1 or sigma_R <= 0 or theta_H <= 0 or d <= 0:
        return None

    def b_from_a(a):
        rhs_deriv = (a - 1.0) / theta_H + (1.0 + xi_R) / sigma_R
        if rhs_deriv <= 0:
            return None
        bd = d * theta_H ** (d - 1.0) / rhs_deriv
        if bd <= 0:
            return None
        return bd ** (1.0 / d)

    def continuity_residual(a):
        b = b_from_a(a)
        if b is None or b <= 0:
            return np.inf
        f_theta = gg_pdf_at_point(theta_H, a, b, d)
        F_theta = gg_cdf(np.array([theta_H]), a, b, d)[0]
        if f_theta <= 0 or F_theta <= 0 or F_theta >= 1:
            return np.inf
        lhs = (1.0 - phi) / F_theta * f_theta
        rhs = phi / sigma_R
        return np.log(lhs) - np.log(rhs)

    try:
        a_lo, a_hi = a_bounds
        f_lo = continuity_residual(a_lo)
        f_hi = continuity_residual(a_hi)
        if not (np.isfinite(f_lo) and np.isfinite(f_hi)):
            return None
        if f_lo * f_hi > 0:
            grid_a = np.logspace(np.log10(a_lo), np.log10(a_hi), 30)
            vals = np.array([continuity_residual(aa) for aa in grid_a])
            finite = np.isfinite(vals)
            if not finite.any():
                return None
            idx = np.argmin(np.abs(vals[finite]))
            a_sol = grid_a[finite][idx]
        else:
            a_sol = brentq(continuity_residual, a_lo, a_hi,
                           xtol=1e-10, maxiter=300)
        b_sol = b_from_a(a_sol)
        if b_sol is None or b_sol <= 0:
            return None
        F_theta = gg_cdf(np.array([theta_H]), a_sol, b_sol, d)[0]
        return {'body_kind': 'gg', 'a': a_sol, 'b': b_sol, 'd': d,
                'F_body_high': float(F_theta)}
    except Exception:
        return None


def body_pdf(x, body_params):
    return gg_pdf(x, body_params['a'], body_params['b'], body_params['d'])

def body_cdf(x, body_params):
    return gg_cdf(x, body_params['a'], body_params['b'], body_params['d'])

def body_ppf(p, body_params):
    return gg_ppf(p, body_params['a'], body_params['b'], body_params['d'])

def body_pdf_at(x0, body_params):
    return gg_pdf_at_point(x0, body_params['a'], body_params['b'],
                           body_params['d'])


def body_logderiv_at(x0, body_params):
    """Derivative d/dx[log f_body(x)] — identical (body always GG here)."""
    if x0 <= 0:
        return np.nan
    a = body_params['a']; b = body_params['b']; d = body_params['d']
    if b <= 0 or d <= 0:
        return np.nan
    return ((a - 1.0) - d * (x0 / b) ** d) / x0


def solve_body_params(body_kind, xi_R, sigma_R, phi, theta_H, d=None):
    # Always GG in this pipeline (both GPD-GG-GPD and Par-GG-Par)
    return solve_body_params_gg(xi_R, sigma_R, phi, theta_H, d)


# ═══════════════════════════════════════════════════════════════════
# FIT BODY+RIGHT
# ═══════════════════════════════════════════════════════════════════
def fit_body_right(data_sorted, case, theta_pct=95.0,
                   phi_target=None, theta_low_exclude=None,
                   fast_mode=False):
    body_kind = case['body']
    right_kind = case['right']

    n_total_all = len(data_sorted)
    if n_total_all < 50:
        return None
    theta_H = float(np.percentile(data_sorted, theta_pct))
    phi = phi_target if phi_target is not None else (100.0 - theta_pct) / 100.0

    if theta_low_exclude is not None and theta_low_exclude > 0:
        mask_fit = data_sorted > theta_low_exclude
        data_fit = data_sorted[mask_fit]
        if len(data_fit) < 50:
            data_fit = data_sorted
            theta_low_exclude = None
        n_tail_count = int(np.sum(data_fit > theta_H))
        if n_tail_count < 20 or n_tail_count >= len(data_fit):
            data_fit = data_sorted
            phi_eff = phi
        else:
            phi_eff = n_tail_count / len(data_fit)
    else:
        data_fit = data_sorted
        phi_eff = phi

    n = len(data_fit)
    if n < 50:
        return None
    emp_cdf = (np.arange(1, n + 1) - 0.5) / n

    tail_data = data_fit[data_fit > theta_H]
    if len(tail_data) < 20:
        return None
    mean_log_ratio = np.mean(np.log(tail_data / theta_H))
    xi_init = np.clip(mean_log_ratio if mean_log_ratio > 0 else 0.2, 0.05, 0.8)
    mean_excess = np.mean(tail_data - theta_H)
    sigma_init = mean_excess * (1.0 - xi_init) if xi_init < 1 else mean_excess * 0.5
    sigma_init = np.clip(sigma_init, theta_H * 0.05, theta_H * 5.0)

    def composite_cdf_loc(x, body_params, xi_R, sigma_R, F_high):
        x = np.asarray(x, dtype=float)
        cdf = np.zeros_like(x)
        body_mask = (x > 0) & (x <= theta_H)
        tail_mask = x > theta_H
        if body_mask.any():
            cdf[body_mask] = (1.0 - phi_eff) / F_high * body_cdf(
                x[body_mask], body_params)
        if tail_mask.any():
            cdf[tail_mask] = (1.0 - phi_eff) + phi_eff * gpd_cdf(
                x[tail_mask], theta_H, xi_R, sigma_R)
        return cdf

    def build_params(xi_R, sigma_R, d_val=None):
        if right_kind == 'pareto_right':
            sigma_R = xi_R * theta_H
            if sigma_R <= 0:
                return None, None
        if sigma_R <= 0 or xi_R <= -1.0 or xi_R > 5.0:
            return None, None
        body_p = solve_body_params(body_kind, xi_R, sigma_R, phi_eff, theta_H,
                                   d=d_val)
        if body_p is None:
            return None, None
        return body_p, sigma_R

    def obj_gg(params):
        xi_R, sigma_R, d_val = params
        if d_val < D_MIN_HARD or d_val > 10:
            return 1e10
        body_p, sigma_R_use = build_params(xi_R, sigma_R, d_val)
        if body_p is None:
            return 1e10
        theo = composite_cdf_loc(data_fit, body_p, xi_R, sigma_R_use,
                                 body_p['F_body_high'])
        theo = np.clip(theo, 1e-12, 1 - 1e-12)
        emp_safe = np.clip(emp_cdf, 1e-12, 1 - 1e-12)
        w = 1.0 / (1.0 - emp_safe) ** 1.5
        rmse = np.sqrt(np.mean(w * (emp_cdf - theo) ** 2))
        if d_val < D_MIN_SOFT:
            rmse += D_PENALTY_LAMBDA * (D_MIN_SOFT - d_val) ** 2
        return rmse

    best = None
    best_val = np.inf
    if fast_mode:
        xi_grid = [xi_init, xi_init * 0.7] + XI_MULTISTART_FIXED_BOOT
        sigma_factors = SIGMA_MULTISTART_FACTORS_BOOT
        d_grid = D_MULTISTART_BOOT
    else:
        xi_grid = ([xi_init * f for f in XI_MULTISTART_FACTORS]
                   + XI_MULTISTART_FIXED)
        sigma_factors = SIGMA_MULTISTART_FACTORS
        d_grid = D_MULTISTART

    for xi_try in xi_grid:
        for s_mult in sigma_factors:
            for d_try in d_grid:
                if right_kind == 'pareto_right':
                    s_try = xi_try * theta_H
                else:
                    s_try = sigma_init * s_mult
                try:
                    res = minimize(obj_gg, [xi_try, s_try, d_try],
                                   method='Nelder-Mead',
                                   options={'xatol': 1e-8, 'fatol': 1e-10,
                                            'maxiter': 3000})
                    if res.success and res.fun < best_val:
                        best_val = res.fun
                        best = res.x
                except Exception:
                    continue

    if best is None:
        return None

    xi_f, sigma_f, d_f = best
    if right_kind == 'pareto_right':
        sigma_f = xi_f * theta_H

    body_p_f, sigma_use = build_params(xi_f, sigma_f, d_f)
    if body_p_f is None:
        return None

    theo_cdf = composite_cdf_loc(data_fit, body_p_f, xi_f, sigma_use,
                                 body_p_f['F_body_high'])
    ks_stat, ks_p = kstest(emp_cdf, theo_cdf)

    mask = emp_cdf > 0.95
    if mask.sum() > 20:
        e = emp_cdf[mask]
        u = (e - e.min()) / (1 - e.min())
        u = np.clip(u, 1e-12, 1 - 1e-12)
        ks_top5_stat, ks_top5_p = kstest(u, 'uniform')
    else:
        ks_top5_stat, ks_top5_p = np.nan, np.nan

    alpha = 1.0 / xi_f if xi_f > 1e-6 else np.inf
    d_boundary = bool(abs(d_f - D_MIN_HARD) < 1e-3)

    out = {
        'theta_H': theta_H,
        'theta_pct': theta_pct,
        'xi_R': xi_f,
        'sigma_R': float(sigma_use),
        'phi': phi_eff,
        'alpha': alpha,
        'wrmse': float(best_val),
        'KS_global_stat': ks_stat, 'KS_global_p': ks_p,
        'KS_top5_stat': ks_top5_stat, 'KS_top5_p': ks_top5_p,
        'n_tail': int((data_fit > theta_H).sum()),
        'n_total': n_total_all, 'n_fit': n,
        'theta_low_used_in_fit': (float(theta_low_exclude)
                                  if theta_low_exclude is not None else 0.0),
        'd_at_hard_boundary': d_boundary,
        'body_params': body_p_f,
    }
    for k, v in body_p_f.items():
        out[f'body_{k}'] = v
    return out


# ═══════════════════════════════════════════════════════════════════
# 3 REGION MODEL
# ═══════════════════════════════════════════════════════════════════
def build_cp3(fit, left_params, obs_all, theta_L):
    if fit is None or left_params is None:
        return None
    theta_H = fit['theta_H']
    phi = fit['phi']
    body_p = fit['body_params']
    x_min = left_params['x_min_pl']

    n_total = len(obs_all)
    n_L = int(np.sum(obs_all <= theta_L))
    n_T = int(np.sum(obs_all > theta_H))
    n_B = n_total - n_L - n_T
    if n_B <= 0 or n_L <= 0 or n_T <= 0:
        return None
    if theta_L <= x_min or theta_L >= theta_H:
        return None

    F_body_low = float(body_cdf(np.array([theta_L]), body_p)[0])
    F_body_high = float(body_p['F_body_high'])
    if F_body_high - F_body_low <= 0:
        return None

    f_L_theta = pdf_L_at(theta_L, left_params, x_min, theta_L)
    f_body_theta = body_pdf_at(theta_L, body_p) / (F_body_high - F_body_low)
    if f_L_theta <= 0 or f_body_theta <= 0:
        return None

    ratio = f_body_theta / f_L_theta
    w_B = (1.0 - phi) / (1.0 + ratio)
    w_L = w_B * ratio
    w_T = phi
    if w_L <= 0 or w_B <= 0:
        return None

    return {
        'case_key': None,
        'theta_L': float(theta_L), 'theta_H': float(theta_H),
        'x_min': float(x_min),
        'xi_R': fit['xi_R'], 'sigma_R': fit['sigma_R'],
        'phi': phi,
        'body_params': body_p, 'left_params': left_params,
        'F_body_low': F_body_low, 'F_body_high': F_body_high,
        'w_L': float(w_L), 'w_B': float(w_B), 'w_T': float(w_T),
        'n_L': n_L, 'n_B': n_B, 'n_T': n_T,
        'pdf_jump_log': 0.0,
    }


def composite3_pdf(x, cp3):
    x = np.asarray(x, dtype=float)
    pdf = np.zeros_like(x)
    theta_L = cp3['theta_L']; theta_H = cp3['theta_H']; x_min = cp3['x_min']

    mask_L = (x >= x_min) & (x <= theta_L)
    if mask_L.any():
        f_L = pdf_L(x[mask_L], cp3['left_params'], x_min, theta_L)
        pdf[mask_L] = cp3['w_L'] * f_L

    mask_B = (x > theta_L) & (x <= theta_H)
    if mask_B.any():
        denom = cp3['F_body_high'] - cp3['F_body_low']
        if denom > 0:
            f_b = body_pdf(x[mask_B], cp3['body_params'])
            pdf[mask_B] = cp3['w_B'] * f_b / denom

    mask_T = x > theta_H
    if mask_T.any():
        pdf[mask_T] = cp3['w_T'] * gpd_pdf(x[mask_T], theta_H,
                                           cp3['xi_R'], cp3['sigma_R'])
    return pdf


def composite3_cdf(x, cp3):
    x = np.asarray(x, dtype=float)
    cdf = np.zeros_like(x)
    theta_L = cp3['theta_L']; theta_H = cp3['theta_H']; x_min = cp3['x_min']
    w_L, w_B, w_T = cp3['w_L'], cp3['w_B'], cp3['w_T']

    mask_L = (x >= x_min) & (x <= theta_L)
    if mask_L.any():
        F_L_cond = cdf_L(x[mask_L], cp3['left_params'], x_min, theta_L)
        cdf[mask_L] = w_L * F_L_cond

    mask_B = (x > theta_L) & (x <= theta_H)
    if mask_B.any():
        denom = cp3['F_body_high'] - cp3['F_body_low']
        F_B_cond = (body_cdf(x[mask_B], cp3['body_params']) - cp3['F_body_low']) / denom
        cdf[mask_B] = w_L + w_B * F_B_cond

    mask_T = x > theta_H
    if mask_T.any():
        F_T_cond = gpd_cdf(x[mask_T], theta_H, cp3['xi_R'], cp3['sigma_R'])
        cdf[mask_T] = w_L + w_B + w_T * F_T_cond

    cdf[x < x_min] = 0.0
    return cdf


def composite3_ppf(p, cp3):
    p = np.clip(np.asarray(p, dtype=float), 1e-15, 1.0 - 1e-15)
    q = np.zeros_like(p)
    theta_L = cp3['theta_L']; theta_H = cp3['theta_H']; x_min = cp3['x_min']
    w_L, w_B, w_T = cp3['w_L'], cp3['w_B'], cp3['w_T']

    mask_L = p <= w_L
    if mask_L.any():
        for idx, p_i in zip(np.where(mask_L)[0], p[mask_L]):
            p_cond = p_i / w_L
            def f(xv):
                return cdf_L(np.array([xv]), cp3['left_params'],
                             x_min, theta_L)[0] - p_cond
            try:
                xv = brentq(f, x_min, theta_L, xtol=1e-10, maxiter=200)
            except Exception:
                xv = x_min + p_cond * (theta_L - x_min)
            q[idx] = xv

    mask_B = (p > w_L) & (p <= w_L + w_B)
    if mask_B.any():
        p_cond = (p[mask_B] - w_L) / w_B
        denom = cp3['F_body_high'] - cp3['F_body_low']
        target_F = p_cond * denom + cp3['F_body_low']
        target_F = np.clip(target_F, 1e-15, 1.0 - 1e-15)
        q[mask_B] = body_ppf(target_F, cp3['body_params'])

    mask_T = p > w_L + w_B
    if mask_T.any():
        p_cond = (p[mask_T] - w_L - w_B) / w_T
        p_cond = np.clip(p_cond, 1e-15, 1.0 - 1e-15)
        q[mask_T] = gpd_ppf(p_cond, theta_H, cp3['xi_R'], cp3['sigma_R'])
    return q


# ═══════════════════════════════════════════════════════════════════
# KS/AD AND SWEEP
# ═══════════════════════════════════════════════════════════════════
def ks_test_3r(obs_all, cp3):
    try:
        stat, p = kstest(obs_all, lambda x: composite3_cdf(x, cp3))
    except Exception:
        data_sorted = np.sort(obs_all)
        emp = (np.arange(1, len(data_sorted) + 1) - 0.5) / len(data_sorted)
        theo = np.clip(composite3_cdf(data_sorted, cp3), 1e-12, 1.0 - 1e-12)
        stat = float(np.max(np.abs(emp - theo)))
        p = np.nan
    return float(stat), float(p)


def ad_stat_3r(obs_all, cp3):
    try:
        data_sorted = np.sort(np.asarray(obs_all, dtype=float))
        n = len(data_sorted)
        if n < 5:
            return np.nan
        U = composite3_cdf(data_sorted, cp3)
        U = np.clip(U, 1e-12, 1.0 - 1e-12)
        if not np.all(np.isfinite(U)):
            return np.nan
        i = np.arange(1, n + 1, dtype=float)
        S = np.sum((2.0 * i - 1.0) * (np.log(U) + np.log(1.0 - U[::-1])))
        A2 = -n - S / n
        return float(A2)
    except Exception:
        return np.nan


def ad_pvalue_marsaglia(A2, n):
    if not np.isfinite(A2) or A2 < 0 or n < 1:
        return np.nan

    def adinf(z):
        if z < 0.2:
            return (1.0 - np.exp(-1.2337141 / z)
                    / np.sqrt(z)) * (2.00012 + (0.247105
                    - (0.0649821 - (0.0347962
                    - (0.0116720 - 0.00168691 * z) * z) * z) * z) * z)
        return np.exp(
            -np.exp(1.0776 - (2.30695 - (0.43424 - (0.082433
                    - (0.008056 - 0.0003146 * z) * z) * z) * z) * z))

    def errfix(n_, x):
        if x < 0.8:
            c = 0.01265 + 0.1757 / n_
            if x < c:
                t = x / c
                t = np.sqrt(t) * (1.0 - t) * (49.0 * t - 102.0)
                return t * (0.0037 / (n_ ** 3) + 0.00078 / (n_ ** 2)
                            + 0.00006 / n_)
            t = (x - c) / (0.8 - c)
            t = -0.00022633 + (6.54034 - (14.6538 - (14.458 - (8.259
                    - 1.91864 * t) * t) * t) * t) * t
            return t * (0.04213 / n_ + 0.01365 / (n_ ** 2))
        t = (x - 0.8) / 0.2
        return (-130.2137 + (745.2337 - (1705.091 - (1950.646
                - (1116.360 - 255.7844 * t) * t) * t) * t) * t) / n_

    try:
        p_inf = adinf(float(A2))
        p = p_inf + errfix(float(n), p_inf)
        p = 1.0 - p
        return float(np.clip(p, 0.0, 1.0))
    except Exception:
        return np.nan


def ad_test_3r(obs_all, cp3):
    n = len(obs_all)
    A2 = ad_stat_3r(obs_all, cp3)
    p = ad_pvalue_marsaglia(A2, n) if np.isfinite(A2) else np.nan
    return (float(A2) if np.isfinite(A2) else np.nan,
            float(p) if np.isfinite(p) else np.nan)


def ks_body_cond(obs_all, cp3):
    theta_L = cp3['theta_L']; theta_H = cp3['theta_H']
    mask = (obs_all > theta_L) & (obs_all <= theta_H)
    n_B = int(mask.sum())
    if n_B < 50:
        return np.nan
    data_B = np.sort(obs_all[mask])
    F_low = cp3['F_body_low']; F_high = cp3['F_body_high']
    denom = F_high - F_low
    if denom <= 0:
        return np.nan
    F_B = (body_cdf(data_B, cp3['body_params']) - F_low) / denom
    F_B = np.clip(F_B, 12, 1.0 - 1e-12)
    emp = (np.arange(1, n_B + 1) - 0.5) / n_B
    return float(np.max(np.abs(emp - F_B)))


def ks_left_cond(obs_all, cp3):
    theta_L = cp3['theta_L']; x_min = cp3['x_min']
    mask = (obs_all >= x_min) & (obs_all <= theta_L)
    n_L = int(mask.sum())
    if n_L < 20:
        return np.nan
    data_L = np.sort(obs_all[mask])
    F_L = cdf_L(data_L, cp3['left_params'], x_min, theta_L)
    F_L = np.clip(F_L, 1e-12, 1.0 - 1e-12)
    emp = (np.arange(1, n_L + 1) - 0.5) / n_L
    return float(np.max(np.abs(emp - F_L)))


def sweep_theta_low(fit, obs_sorted_all, case, pct_grid=None):
    """Sweep θ_L with same composite score as the original."""
    if pct_grid is None:
        pct_grid = THETA_LOW_PCT_GRID
    n_total = len(obs_sorted_all)
    if fit is None or n_total < 50:
        return None, None, np.nan, []

    min_n_L = min_n_L_dynamic(n_total)

    best_cp3 = None
    best_left = None
    best_pct = np.nan
    best_score = np.inf
    sweep_log = []

    for pct in pct_grid:
        n_disc = int(n_total * (pct / 100.0))
        if n_disc < min_n_L:
            sweep_log.append((pct, np.nan, np.nan, np.nan, np.nan, np.nan))
            continue
        obs_disc_i = obs_sorted_all[:n_disc]
        theta_L_i = float(obs_sorted_all[n_disc])

        try:
            left_i = fit_left(case['left'], obs_disc_i, theta_L_i,
                              body_params=fit['body_params'])
        except Exception:
            left_i = None
        if left_i is None:
            sweep_log.append((pct, np.nan, np.nan, np.nan, np.nan, np.nan))
            continue

        cp3_i = build_cp3(fit, left_i, obs_sorted_all, theta_L_i)
        if cp3_i is None:
            sweep_log.append((pct, np.nan, np.nan, np.nan, np.nan, np.nan))
            continue
        cp3_i['case_key'] = case['key']

        try:
            ks_stat_i, ks_p_i = ks_test_3r(obs_sorted_all, cp3_i)
        except Exception:
            sweep_log.append((pct, np.nan, np.nan, np.nan, np.nan, np.nan))
            continue

        kb = ks_body_cond(obs_sorted_all, cp3_i)
        kl = ks_left_cond(obs_sorted_all, cp3_i)

        score = ks_stat_i
        if np.isfinite(kb):
            score += LAMBDA_BODY * kb
        if np.isfinite(kl):
            score += LAMBDA_LEFT * kl

        sweep_log.append((pct, ks_stat_i, ks_p_i, kb, kl, score))
        if np.isfinite(score) and score < best_score:
            best_score = score
            best_cp3 = cp3_i
            best_left = left_i
            best_pct = pct

    return best_cp3, best_left, best_pct, sweep_log


# ═══════════════════════════════════════════════════════════════════
# VRETURNS
# ═══════════════════════════════════════════════════════════════════
def extract_abs_vr(prices_array):
    pv = np.asarray(prices_array, dtype=float)
    lp = np.log(pv)
    trends = []
    idx = 0
    while idx < len(pv) - 1:
        j = idx + 1
        while j < len(pv) and pv[j] >= pv[j-1]:
            j += 1
        if j - 1 > idx:
            trends.append((idx, j-1))
            idx = j - 1
            continue
        k = idx + 1
        while k < len(pv) and pv[k] <= pv[k-1]:
            k += 1
        if k - 1 > idx:
            trends.append((idx, k-1))
            idx = k - 1
            continue
        idx += 1
    pos, neg = [], []
    for s, e in trends:
        for ii in range(1, e - s + 1):
            vr = (lp[s + ii] - lp[s]) / ii
            if vr > 0:
                pos.append(vr)
            elif vr < 0:
                neg.append(abs(vr))
    return np.array(pos), np.array(neg)


# ═══════════════════════════════════════════════════════════════════
# COMPLETE FIT (pipeline)
# ═══════════════════════════════════════════════════════════════════
def _run_single_fit(obs_sorted, case, theta_pct, n_refit_iters=MAX_REFIT_ITERS):
    fit = fit_body_right(obs_sorted, case, theta_pct=theta_pct)
    if fit is None:
        return None
    cp3, left_p, pct_opt, sweep_log = sweep_theta_low(fit, obs_sorted, case)
    if cp3 is None:
        return None

    history = [(0, fit['wrmse'],
                ks_test_3r(obs_sorted, cp3)[0],
                fit['body_params'].get('d', np.nan))]

    for it in range(1, n_refit_iters + 1):
        theta_L_opt = cp3['theta_L']
        fit_new = fit_body_right(obs_sorted, case,
                                 theta_pct=theta_pct,
                                 theta_low_exclude=theta_L_opt)
        if fit_new is None:
            break
        cp3_new, left_new, pct_new, sweep_new = sweep_theta_low(
            fit_new, obs_sorted, case)
        if cp3_new is None:
            break

        ks_old = ks_test_3r(obs_sorted, cp3)[0]
        ks_new = ks_test_3r(obs_sorted, cp3_new)[0]
        step1_b = fit.get('d_at_hard_boundary', False)
        step2_b = fit_new.get('d_at_hard_boundary', False)

        if step1_b and not step2_b:
            accept = True
        elif not step2_b and ks_new <= ks_old * REFIT_KS3R_TOL:
            accept = True
        else:
            accept = False

        history.append((it, fit_new['wrmse'], ks_new,
                        fit_new['body_params'].get('d', np.nan)))

        if accept:
            fit = fit_new
            cp3 = cp3_new
            left_p = left_new
            pct_opt = pct_new
            sweep_log = sweep_new
            if abs(ks_new - ks_old) < 1e-5:
                break
        else:
            break

    return {
        'fit': fit, 'cp3': cp3, 'left_p': left_p,
        'pct_opt': pct_opt, 'sweep_log': sweep_log,
        'history': history,
        'theta_pct': theta_pct,
    }


# ═══════════════════════════════════════════════════════════════════
# BOOTSTRAP LILLIEFORS
# ═══════════════════════════════════════════════════════════════════
def _fit_and_ks_for_bootstrap(sample_sorted, case, theta_pct_fixed):
    """Fast refit in bootstrap."""
    try:
        fit = fit_body_right(sample_sorted, case,
                             theta_pct=theta_pct_fixed,
                             fast_mode=True)
        if fit is None:
            return np.nan, np.nan, None
        cp3, left_p, pct_opt, _ = sweep_theta_low(fit, sample_sorted, case)
        if cp3 is None:
            return np.nan, np.nan, None

        if MAX_REFIT_ITERS_BOOT > 0:
            theta_L_opt = cp3['theta_L']
            fit2 = fit_body_right(sample_sorted, case,
                                  theta_pct=theta_pct_fixed,
                                  theta_low_exclude=theta_L_opt,
                                  fast_mode=True)
            if fit2 is not None:
                cp3_2, left_2, pct_2, _ = sweep_theta_low(
                    fit2, sample_sorted, case)
                if cp3_2 is not None:
                    step1_b = fit.get('d_at_hard_boundary', False)
                    step2_b = fit2.get('d_at_hard_boundary', False)
                    ks_old = ks_test_3r(sample_sorted, cp3)[0]
                    ks_new = ks_test_3r(sample_sorted, cp3_2)[0]
                    if (step1_b and not step2_b) or (
                            not step2_b and ks_new <= ks_old * REFIT_KS3R_TOL):
                        cp3 = cp3_2

        ks_stat, _ = ks_test_3r(sample_sorted, cp3)
        ad_stat, _ = ad_test_3r(sample_sorted, cp3)

        p = {
            'xi_R': float(cp3['xi_R']),
            'alpha': float(1.0/cp3['xi_R'] if cp3['xi_R'] > 1e-6 else np.nan),
            'sigma_R': float(cp3['sigma_R']),
            'a': float(cp3['body_params'].get('a', np.nan)),
            'b': float(cp3['body_params'].get('b', np.nan)),
            'd': float(cp3['body_params'].get('d', np.nan)),
            'theta_L': float(cp3['theta_L']),
            'xi_L': float(cp3['left_params'].get('xi_L', np.nan)),
            'sigma_L': float(cp3['left_params'].get('sigma_L', np.nan)),
            'alpha_pl': float(cp3['left_params'].get('alpha_pl', np.nan)),
            'w_L': float(cp3['w_L']),
            'w_B': float(cp3['w_B']),
            'w_T': float(cp3['w_T']),
        }
        return float(ks_stat), float(ad_stat), p
    except Exception:
        return np.nan, np.nan, None


# ═══════════════════════════════════════════════════════════════════
# WORKERS PARALELIZADOS
# ═══════════════════════════════════════════════════════════════════
def fit_single_theta_pct_worker(args):
    asset, sign, obs_sorted, theta_pct, case_key = args
    case = CASE_BY_KEY[case_key]
    try:
        result = _run_single_fit(obs_sorted, case, theta_pct,
                                 n_refit_iters=MAX_REFIT_ITERS)
        if result is None:
            return {
                'asset': asset, 'sign': sign, 'theta_pct': theta_pct,
                'case_key': case_key, 'valid': False, 'ks_stat': np.inf,
            }

        fit = result['fit']; cp3 = result['cp3']
        if cp3 is None or result['left_p'] is None:
            return {
                'asset': asset, 'sign': sign, 'theta_pct': theta_pct,
                'case_key': case_key, 'valid': False, 'ks_stat': np.inf,
            }

        ks3_stat, ks3_p = ks_test_3r(obs_sorted, cp3)
        ad3_stat, ad3_p = ad_test_3r(obs_sorted, cp3)
        kb = ks_body_cond(obs_sorted, cp3)
        kl = ks_left_cond(obs_sorted, cp3)

        return {
            'asset': asset, 'sign': sign, 'theta_pct': theta_pct,
            'case_key': case_key, 'valid': True,
            'ks_stat': float(ks3_stat),
            'obs_sorted': obs_sorted,
            'fit': fit, 'cp3': cp3, 'left_p': result['left_p'],
            'pct_opt': result['pct_opt'],
            'sweep_log': result['sweep_log'],
            'history': result['history'],
            'ks3_stat': ks3_stat, 'ks3_p': ks3_p,
            'ad3_stat': ad3_stat, 'ad3_p': ad3_p,
            'kb': kb, 'kl': kl,
        }
    except Exception as e:
        import traceback
        print(f"  [FIT ERROR] {asset} {sign} {case_key} theta_pct={theta_pct}: {e}")
        traceback.print_exc()
        return {
            'asset': asset, 'sign': sign, 'theta_pct': theta_pct,
            'case_key': case_key, 'valid': False, 'ks_stat': np.inf,
        }


def bootstrap_task_worker(args):
    asset, sign, case_key, boot_idx, cp3_orig, theta_pct_fixed, n_sim, seed = args
    case = CASE_BY_KEY[case_key]
    try:
        rng = np.random.default_rng(seed)
        u = rng.uniform(0, 1, n_sim)
        sample_b = composite3_ppf(u, cp3_orig)
        sample_b = np.sort(sample_b[np.isfinite(sample_b) & (sample_b > 0)])
        if len(sample_b) < 50:
            return (asset, sign, case_key, boot_idx, np.nan, np.nan, None)
        D_ks, D_ad, P = _fit_and_ks_for_bootstrap(sample_b, case, theta_pct_fixed)
        return (asset, sign, case_key, boot_idx, D_ks, D_ad, P)
    except Exception:
        return (asset, sign, case_key, boot_idx, np.nan, np.nan, None)


def _fmt_entry(p, s, kb, kl, sc):
    def _f(x):
        return f"{x:.4f}" if np.isfinite(x) else 'NA'
    return f"{p:.1f}:{_f(s)}:{_f(kb)}:{_f(kl)}:{_f(sc)}"


def generate_record_only(fit_result):
    asset = fit_result['asset']
    sign = fit_result['sign']
    case_key = fit_result['case_key']
    try:
        obs_sorted = fit_result['obs_sorted']
        fit = fit_result['fit']
        cp3 = fit_result['cp3']
        left_p = fit_result['left_p']
        pct_opt = fit_result['pct_opt']
        sweep_log = fit_result['sweep_log']
        history = fit_result['history']
        theta_h_sweep = fit_result.get('theta_h_sweep', [])
        ks3_stat = fit_result['ks3_stat']
        ks3_p = fit_result['ks3_p']
        ad3_stat = fit_result.get('ad3_stat', np.nan)
        ad3_p = fit_result.get('ad3_p', np.nan)
        kb = fit_result['kb']
        kl = fit_result['kl']

        ks_boot_arr = fit_result['ks_boot_arr']
        ad_boot_arr = fit_result.get('ad_boot_arr',
                                     np.full_like(ks_boot_arr, np.nan))
        boot_params_list = fit_result['boot_params_list']

        ks_boot_valid = ks_boot_arr[np.isfinite(ks_boot_arr)]
        if len(ks_boot_valid) > 0:
            ks3_p_lilliefors = float(np.mean(ks_boot_valid >= ks3_stat))
            n_boot_used = len(ks_boot_valid)
        else:
            ks3_p_lilliefors = np.nan
            n_boot_used = 0

        ad_boot_valid = ad_boot_arr[np.isfinite(ad_boot_arr)]
        if len(ad_boot_valid) > 0 and np.isfinite(ad3_stat):
            ad3_p_lilliefors = float(np.mean(ad_boot_valid >= ad3_stat))
            n_boot_used_ad = len(ad_boot_valid)
        else:
            ad3_p_lilliefors = np.nan
            n_boot_used_ad = 0

        if len(ks_boot_valid) > 0:
            ks_boot_mean = float(np.mean(ks_boot_valid))
            ks_boot_median = float(np.median(ks_boot_valid))
            ks_boot_q05 = float(np.quantile(ks_boot_valid, 0.05))
            ks_boot_q95 = float(np.quantile(ks_boot_valid, 0.95))
            ks_boot_max = float(np.max(ks_boot_valid))
        else:
            ks_boot_mean = ks_boot_median = ks_boot_q05 = ks_boot_q95 = ks_boot_max = np.nan

        if len(ad_boot_valid) > 0:
            ad_boot_mean = float(np.mean(ad_boot_valid))
            ad_boot_median = float(np.median(ad_boot_valid))
            ad_boot_q05 = float(np.quantile(ad_boot_valid, 0.05))
            ad_boot_q95 = float(np.quantile(ad_boot_valid, 0.95))
            ad_boot_max = float(np.max(ad_boot_valid))
        else:
            ad_boot_mean = ad_boot_median = ad_boot_q05 = ad_boot_q95 = ad_boot_max = np.nan

        p_boot_df = pd.DataFrame([p for p in boot_params_list if p is not None])
        se_dict = {}
        if not p_boot_df.empty:
            for col in p_boot_df.columns:
                se_dict[f'{col}_se'] = float(p_boot_df[col].std(ddof=1))

        record = {
            'Model': case_key,
            'Asset': asset, 'Sign': sign,
            'theta_H': fit['theta_H'], 'theta_pct': fit['theta_pct'],
            'xi_R': fit['xi_R'], 'sigma_R': fit['sigma_R'], 'phi': fit['phi'],
            'alpha': fit['alpha'], 'wrmse': fit['wrmse'],
            'KS_global_stat': fit['KS_global_stat'],
            'KS_global_p': fit['KS_global_p'],
            'KS_top5_stat': fit['KS_top5_stat'],
            'KS_top5_p': fit['KS_top5_p'],
            'n_tail': fit['n_tail'], 'n_total': fit['n_total'],
            'n_fit': fit['n_fit'],
            'd_at_hard_boundary': fit['d_at_hard_boundary'],
            **{f'body_{k}': v for k, v in fit['body_params'].items()},
            'theta_L_3r': cp3['theta_L'],
            'theta_low_pct_opt': pct_opt,
            'x_min': cp3['x_min'],
            'w_L_3r': cp3['w_L'], 'w_B_3r': cp3['w_B'], 'w_T_3r': cp3['w_T'],
            'n_L_3r': cp3['n_L'], 'n_B_3r': cp3['n_B'], 'n_T_3r': cp3['n_T'],
            'KS_stat_3r': ks3_stat,
            'KS_p_3r_asymptotic': ks3_p,
            'KS_p_3r_lilliefors': ks3_p_lilliefors,
            'n_boot_lilliefors': n_boot_used,
            'KS_boot_mean': ks_boot_mean,
            'KS_boot_median': ks_boot_median,
            'KS_boot_q05': ks_boot_q05,
            'KS_boot_q95': ks_boot_q95,
            'KS_boot_max': ks_boot_max,
            'AD_stat_3r': ad3_stat,
            'AD_p_3r_asymptotic': ad3_p,
            'AD_p_3r_lilliefors': ad3_p_lilliefors,
            'n_boot_lilliefors_ad': n_boot_used_ad,
            'AD_boot_mean': ad_boot_mean,
            'AD_boot_median': ad_boot_median,
            'AD_boot_q05': ad_boot_q05,
            'AD_boot_q95': ad_boot_q95,
            'AD_boot_max': ad_boot_max,
            'KS_body_cond_3r': kb, 'KS_left_cond_3r': kl,
            'KS_p_3r': ks3_p,
            **se_dict,
            **{f'left_{k}': v for k, v in left_p.items()},
            'sweep_log': ';'.join(_fmt_entry(p, s, kb2, kl2, sc)
                                  for p, s, _, kb2, kl2, sc in sweep_log),
            'theta_h_sweep_log': ';'.join(
                f"{pct:.2f}:{ks:.4f}" if np.isfinite(ks) else f"{pct:.2f}:NA"
                for pct, ks, _ in theta_h_sweep),
            'n_refit_iters_used': len(history) - 1,
            'd_history': ';'.join(f"{h[0]}:{h[3]:.3f}" for h in history),
            'ks_boot_all': ';'.join(f"{d:.5f}" for d in ks_boot_valid),
            'ad_boot_all': ';'.join(f"{d:.5f}" for d in ad_boot_valid),
        }
        return record
    except Exception as e:
        import traceback
        print(f"  [RECORD ERROR] {asset} {sign} {case_key}: {e}")
        traceback.print_exc()
        return None


# ═══════════════════════════════════════════════════════════════════
# COMPARATIVE PLOTS
# ═══════════════════════════════════════════════════════════════════
def add_legend(ax, loc='best'):
    ax.legend(loc=loc, fontsize=11.5, frameon=True, framealpha=0.85,
              edgecolor='0.6', labelspacing=0.4, handlelength=1.5,
              borderpad=0.5, handletextpad=0.5)

def _sane_ylim_log(ax, counts_min=0.5, floor_decades=4):
    try:
        y0, y1 = ax.get_ylim()
        if y1 > 0 and np.isfinite(y1):
            floor = max(counts_min, y1 / (10 ** floor_decades))
            if y0 < floor:
                ax.set_ylim(floor, y1)
    except Exception:
        pass


def _compare_ccdf_on_ax(ax, obs_sorted, fit_by_model, sign):
    n = len(obs_sorted)
    y_emp = (n - np.arange(1, n + 1) + 0.5) / (n + 1)

    ax.step(obs_sorted, y_emp, where='post', color=C_EMPIRICAL, lw=0.7,
            alpha=0.55, zorder=1, label=None)
    def _subsample(indices, target=600):
        if len(indices) <= target:
            return indices
        step = max(1, len(indices) // target)
        return indices[::step]
    idx_all = np.arange(n)
    s_all = _subsample(idx_all)
    ax.plot(obs_sorted[s_all], y_emp[s_all], '.', color=C_EMPIRICAL,
            ms=2.2, alpha=0.9, zorder=2, label=f'Empirical (n={n})')

    xf_full = np.linspace(obs_sorted.min(), obs_sorted.max(), 2000)
    for case_key in ['exp_gg_gpd', 'pareto_gg_pareto']:
        fit_res = fit_by_model.get(case_key)
        if fit_res is None:
            continue
        case = CASE_BY_KEY[case_key]
        cp3 = fit_res['cp3']
        theta_L = cp3['theta_L']; theta_H = cp3['theta_H']; x_min = cp3['x_min']
        ls = case['linestyle']; lbl_short = case['label_short']

        xf = xf_full[(xf_full >= x_min)]
        if len(xf) < 10:
            continue
        ccdf = np.clip(1.0 - composite3_cdf(xf, cp3), 1e-300, 1.0)
        seg_L = (xf >= x_min) & (xf <= theta_L)
        seg_B = (xf > theta_L) & (xf <= theta_H)
        seg_T = (xf > theta_H)
        def _masked(mask):
            return np.where(mask, ccdf, np.nan)

        ax.plot(xf, _masked(seg_L), linestyle=ls, color=C_REGION_L, lw=2,
                alpha=0.9, zorder=3, label=None)
        ax.plot(xf, _masked(seg_B), linestyle=ls, color=C_REGION_B, lw=2,
                alpha=0.9, zorder=3, label=None)
        ax.plot(xf, _masked(seg_T), linestyle=ls, color=C_REGION_T, lw=2,
                alpha=0.9, zorder=3, label=None)
        ax.plot([], [], linestyle=ls, color='k', label=lbl_short)
        ax.axvline(x=theta_L, color=C_REGION_L, ls=ls, lw=0.8, alpha=0.45)
        ax.axvline(x=theta_H, color=C_REGION_T, ls=ls, lw=0.8, alpha=0.5)

    ax.set_yscale('log')
    xl = 'VReturns' if sign == 'positive' else '|VReturns|'
    ax.set_xlabel(xl); ax.set_ylabel('CCDF (log)')
    ax.grid(True, which='both', ls=':', alpha=0.3)
    add_legend(ax, 'upper right')
    _sane_ylim_log(ax, counts_min=0.2/n, floor_decades=6)


def _compare_hist_on_ax(ax, obs_sorted, fit_by_model):
    obs_all = obs_sorted
    n_all = len(obs_all)
    log_min = np.log10(max(obs_all.min(), 1e-8))
    log_max = np.log10(obs_all.max())
    be = np.logspace(log_min, log_max, 70)
    counts, _ = np.histogram(obs_all, bins=be)
    bc = np.sqrt(be[:-1] * be[1:])
    bw = np.diff(be)
    m = counts > 0

    if m.any():
        ax.scatter(bc[m], counts[m], color=C_EMPIRICAL, s=16,
                   edgecolors='k', linewidths=0.3, zorder=3,
                   label=f'Empirical (n={n_all})')

    for case_key in ['exp_gg_gpd', 'pareto_gg_pareto']:
        fit_res = fit_by_model.get(case_key)
        if fit_res is None:
            continue
        case = CASE_BY_KEY[case_key]
        cp3 = fit_res['cp3']
        theta_L = cp3['theta_L']; theta_H = cp3['theta_H']; x_min = cp3['x_min']
        ls = case['linestyle']; lbl_short = case['label_short']

        x_L = np.logspace(np.log10(max(x_min, obs_all.min())),
                          np.log10(theta_L), 400)
        pdf_L_val = composite3_pdf(x_L, cp3)
        bw_L = np.interp(x_L, bc, bw)
        counts_L = pdf_L_val * n_all * bw_L
        counts_L = np.where((pdf_L_val > 0) & (counts_L >= 0.2),
                            counts_L, np.nan)
        ax.plot(x_L, counts_L, linestyle=ls, color=C_REGION_L, lw=2,
                alpha=0.9, label=None)

        x_B = np.logspace(np.log10(theta_L), np.log10(theta_H), 400)
        pdf_B = composite3_pdf(x_B, cp3)
        bw_B = np.interp(x_B, bc, bw)
        counts_B = pdf_B * n_all * bw_B
        counts_B = np.where((pdf_B > 0) & (counts_B >= 0.2), counts_B, np.nan)
        ax.plot(x_B, counts_B, linestyle=ls, color=C_REGION_B, lw=2,
                alpha=0.9, label=None)

        x_T = np.logspace(np.log10(theta_H), np.log10(obs_all.max()), 400)
        pdf_T = composite3_pdf(x_T, cp3)
        bw_T = np.interp(x_T, bc, bw)
        counts_T = pdf_T * n_all * bw_T
        counts_T = np.where((pdf_T > 0) & (counts_T >= 0.2), counts_T, np.nan)
        ax.plot(x_T, counts_T, linestyle=ls, color=C_REGION_T, lw=2,
                alpha=0.9, label=None)

        ax.plot([], [], linestyle=ls, color='k', label=lbl_short)
        ax.axvline(x=theta_L, color=C_REGION_L, ls=ls, lw=0.8, alpha=0.45)
        ax.axvline(x=theta_H, color=C_REGION_T, ls=ls, lw=0.8, alpha=0.5)

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('|VReturn| (log)'); ax.set_ylabel('Counts (log)')
    ax.grid(True, which='both', ls=':', alpha=0.3)
    add_legend(ax, 'upper left')
    _sane_ylim_log(ax)


def _compare_qq_on_ax(ax, obs_sorted, fit_by_model):
    n = len(obs_sorted)
    p = (np.arange(1, n + 1) - 0.5) / n
    tq_all = []
    for case_key in ['exp_gg_gpd', 'pareto_gg_pareto']:
        fit_res = fit_by_model.get(case_key)
        if fit_res is None:
            continue
        tq_all.append(composite3_ppf(p, fit_res['cp3']))
    if not tq_all:
        return
    tq_concat = np.concatenate(tq_all)
    lo = min(np.nanmin(tq_concat), obs_sorted.min())
    hi = max(np.nanmax(tq_concat), obs_sorted.max())

    ax.plot([lo, hi], [lo, hi], '-', color=C_EMPIRICAL, lw=1.1,
            alpha=0.85, zorder=1, label='Perfect fit (y=x)')

    for case_key in ['exp_gg_gpd', 'pareto_gg_pareto']:
        fit_res = fit_by_model.get(case_key)
        if fit_res is None:
            continue
        case = CASE_BY_KEY[case_key]
        cp3 = fit_res['cp3']
        theta_L = cp3['theta_L']; theta_H = cp3['theta_H']
        tq = composite3_ppf(p, cp3)
        marker = case['marker']
        lbl_short = case['label_short']

        mL = obs_sorted <= theta_L
        mB = (obs_sorted > theta_L) & (obs_sorted <= theta_H)
        mT = obs_sorted > theta_H

        if marker == 'o':
            marker_kwargs = dict(s=10, alpha=0.6, edgecolors='none')
        else:
            marker_kwargs = dict(s=14, alpha=0.75, linewidths=1.0)

        ax.scatter(tq[mL], obs_sorted[mL], marker=marker,
                   color=C_REGION_L, zorder=3, label=None, **marker_kwargs)
        ax.scatter(tq[mB], obs_sorted[mB], marker=marker,
                   color=C_REGION_B, zorder=3, label=None, **marker_kwargs)
        ax.scatter(tq[mT], obs_sorted[mT], marker=marker,
                   color=C_REGION_T, zorder=3, label=None, **marker_kwargs)
        ax.scatter([], [], marker=marker, color='k', label=lbl_short,
                   **marker_kwargs)

    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Empirical Quantiles')
    if obs_sorted.max() / max(obs_sorted.min(), 1e-12) > 100:
        ax.set_xscale('log'); ax.set_yscale('log')
    ax.grid(True, ls=':', alpha=0.3)
    add_legend(ax, 'upper left')


def generate_comparative_plots(asset, fits_pos, fits_neg, out_dir):
    for plot_type in ['hist', 'ccdf', 'qq']:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for i, (sign_dir, fits_by_model) in enumerate(
                [('positive', fits_pos), ('negative', fits_neg)]):
            ax = axes[i]
            if not fits_by_model:
                ax.set_axis_off()
                continue

            any_fit = next(iter(fits_by_model.values()))
            obs_sorted = any_fit['obs_sorted']

            if plot_type == 'hist':
                _compare_hist_on_ax(ax, obs_sorted, fits_by_model)
                ax.set_title(f'{asset} {sign_dir.capitalize()} VReturns')
            elif plot_type == 'ccdf':
                _compare_ccdf_on_ax(ax, obs_sorted, fits_by_model, sign_dir)
                ax.set_title(f'{asset} {sign_dir.capitalize()} VReturns (CCDF)')
            elif plot_type == 'qq':
                _compare_qq_on_ax(ax, obs_sorted, fits_by_model)
                ax.set_title(f'{asset} {sign_dir.capitalize()} VReturns (QQ-plot)')

        fig.tight_layout()
        out_path = os.path.join(out_dir, f'plot_{plot_type}_{asset}.png')
        fig.savefig(out_path, dpi=140, bbox_inches='tight')
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    print("=" * 78)
    print(f"Exp-GΓ-GPD vs Par-GΓ-Par — finer grids — "
          f"{NUM_WORKERS} workers")
    print(f"  Models: {[c['key'] for c in CASES]}")
    print(f"  Assets: {ASSETS_ALLOWED}")
    print(f"  θ_H grid ({len(THETA_H_PCT_GRID)} pts): {THETA_H_PCT_GRID}")
    print(f"  θ_L grid ({len(THETA_LOW_PCT_GRID)} pts): {THETA_LOW_PCT_GRID}")
    print(f"  Multi-start d (main, {len(D_MULTISTART)}): {D_MULTISTART}")
    print(f"  Multi-start d (bootstrap): {D_MULTISTART_BOOT}")
    print(f"  Max refit iters (main): {MAX_REFIT_ITERS}, (bootstrap): {MAX_REFIT_ITERS_BOOT}")
    print(f"  Lilliefors bootstrap: {DO_LILLIEFORS} (B={N_BOOTSTRAP})")
    print(f"  tqdm: {HAS_TQDM}  Output: {OUT_DIR}/")
    print("=" * 78)

    # Date ranges according to Table I
    TABLE1_RANGES = {
        'DJIA':   ('1992-01-02', '2023-12-29'),
        'DAX':    ('1992-01-02', '2023-12-29'),
        'IPC':    ('1992-01-02', '2023-12-29'),
        'Nikkei': ('1992-01-06', '2023-12-29'),
    }

    data = {}
    for p in sorted(glob.glob(os.path.join(DATA_DIR, '*_close.pkl'))):
        name = os.path.basename(p).replace('_close.pkl', '')
        if name not in ASSETS_ALLOWED:
            continue
        try:
            with open(p, 'rb') as f:
                ts = pickle.load(f).dropna()
            if name in TABLE1_RANGES:
                start_dt, end_dt = TABLE1_RANGES[name]
                ts = ts.loc[start_dt:end_dt]
            data[name] = ts
            print(f"  Loaded {name}: {len(data[name])} obs "
                  f"(Period: {ts.index[0].date()} to {ts.index[-1].date()})")
        except Exception as e:
            print(f"  Warning {name}: {e}")
    if not data:
        print(f"Sin datos. Se esperaba: {ASSETS_ALLOWED}"); return

    fit_tasks = []
    for name, prices in data.items():
        pos, neg = extract_abs_vr(prices.values)
        if len(pos) >= 200:
            fit_tasks.append((name, 'positive', pos))
        if len(neg) >= 200:
            fit_tasks.append((name, 'negative', neg))

    # ─── Fase 1 ────────────────────────────────────────────────
    n_series = len(fit_tasks)
    n_per_serie = len(THETA_H_PCT_GRID)
    n_models = len(CASES)
    n_total_tasks_f1 = n_series * n_per_serie * n_models
    print(f"\n─── Fase 1/3: Ajuste paralelo "
          f"({n_series} series × {n_per_serie} θ_H × {n_models} modelos "
          f"= {n_total_tasks_f1} tareas) ───")

    fit_subtasks = []
    obs_sorted_cache = {}
    for (asset, sign, obs) in fit_tasks:
        obs_s = np.sort(obs)
        obs_sorted_cache[(asset, sign)] = obs_s
        for theta_pct in THETA_H_PCT_GRID:
            for case in CASES:
                fit_subtasks.append(
                    (asset, sign, obs_s, theta_pct, case['key']))

    keys_all = set()
    for (asset, sign, _obs) in fit_tasks:
        for case in CASES:
            keys_all.add((asset, sign, case['key']))
    subresults_by_serie = {k: [] for k in keys_all}

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(fit_single_theta_pct_worker, t): t
                   for t in fit_subtasks}
        iterator = as_completed(futures)
        if HAS_TQDM:
            iterator = tqdm(iterator, total=len(futures),
                            desc="  Fitting", unit="fit")
        for future in iterator:
            res = future.result()
            if res is not None:
                key = (res['asset'], res['sign'], res['case_key'])
                subresults_by_serie[key].append(res)

    # Phase 1b: choose best theta_pct per (series, model)
    fit_results = {}
    for key, subresults in subresults_by_serie.items():
        valid = [r for r in subresults if r.get('valid', False)]
        if not valid:
            continue
        best = min(valid, key=lambda r: r['ks_stat'])
        theta_h_sweep = [
            (r['theta_pct'], r['ks_stat'], np.nan) for r in subresults
        ]
        best['theta_h_sweep'] = theta_h_sweep
        fit_results[key] = best

    for case in CASES:
        ck = case['key']
        n_ok = sum(1 for k in fit_results.keys() if k[2] == ck)
        print(f"  ✓ {case['label_short']}: {n_ok}/{n_series} series ajustadas")

    if not fit_results:
        print("No valid fits found."); return

    # ─── Phase 2 — Parallel Bootstrap Lilliefors (B=1000) ─────────────────
    total_boot = N_BOOTSTRAP * len(fit_results)
    print(f"\n─── Phase 2/3: Parallel Bootstrap Lilliefors "
          f"({len(fit_results)} fits × B={N_BOOTSTRAP} = "
          f"{total_boot} tareas) ───")

    boot_tasks = []
    for (asset, sign, case_key), res in fit_results.items():
        n_sim = min(len(res['obs_sorted']), BOOTSTRAP_MAX_N)
        base_seed = hash((asset, sign, case_key)) & 0xFFFFFF
        for b in range(N_BOOTSTRAP):
            boot_tasks.append((
                asset, sign, case_key, b,
                res['cp3'], res['fit']['theta_pct'],
                n_sim, base_seed + b,
            ))

    boot_results_ks = {k: [np.nan] * N_BOOTSTRAP for k in fit_results.keys()}
    boot_results_ad = {k: [np.nan] * N_BOOTSTRAP for k in fit_results.keys()}
    boot_results_ps = {k: [None] * N_BOOTSTRAP for k in fit_results.keys()}

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(bootstrap_task_worker, t) for t in boot_tasks]
        iterator = as_completed(futures)
        if HAS_TQDM:
            iterator = tqdm(iterator, total=len(boot_tasks),
                            desc="  Bootstrap", unit="rep")
        for future in iterator:
            asset, sign, case_key, boot_idx, D_ks, D_ad, P = future.result()
            key = (asset, sign, case_key)
            if key in boot_results_ks:
                boot_results_ks[key][boot_idx] = D_ks
                boot_results_ad[key][boot_idx] = D_ad
                boot_results_ps[key][boot_idx] = P

    for key, arr in boot_results_ks.items():
        n_valid = int(np.sum(np.isfinite(arr)))
        print(f"  {key[0]} {key[1]} [{key[2]}]: {n_valid}/{N_BOOTSTRAP} "
              f"bootstraps válidos")

    # ─── Fase 3 — Records + Plots ──────────────────────────────
    print(f"\n─── Fase 3/3: Generación de records CSV y plots ───")

    records = []
    items = list(fit_results.items())
    iterator = tqdm(items, desc="  Records", unit="serie") if HAS_TQDM else items
    for key, fit_res in iterator:
        fit_res['ks_boot_arr'] = np.array(boot_results_ks[key])
        fit_res['ad_boot_arr'] = np.array(boot_results_ad[key])
        fit_res['boot_params_list'] = boot_results_ps[key]
        rec = generate_record_only(fit_res)
        if rec is not None:
            records.append(rec)

    if not records:
        print("Sin registros."); return

    print("\n  Generando plots comparativos (CCDF, histograma, QQ)...")
    fits_by_asset_sign = {}
    for (asset, sign, case_key), fit_res in fit_results.items():
        fits_by_asset_sign.setdefault((asset, sign), {})[case_key] = fit_res

    assets_found = sorted(set(k[0] for k in fit_results.keys()))
    for asset in assets_found:
        fits_pos = fits_by_asset_sign.get((asset, 'positive'), {})
        fits_neg = fits_by_asset_sign.get((asset, 'negative'), {})
        generate_comparative_plots(asset, fits_pos, fits_neg, OUT_DIR)
        print(f"    ✓ {asset}: plot_ccdf, plot_hist, plot_qq")

    # CSV principal
    df = pd.DataFrame(records)
    cols_order = ['Model', 'Asset', 'Sign'] + [
        c for c in df.columns if c not in ('Model', 'Asset', 'Sign')]
    df = df[cols_order]
    df = df.sort_values(['Model', 'Asset', 'Sign']).reset_index(drop=True)
    out_csv = os.path.join(OUT_DIR, 'metrics.csv')
    df.to_csv(out_csv, index=False, encoding='utf-8')
    print(f"\n✓ {len(records)} records → {out_csv}")

    # bootstrap_details.csv
    boot_rows = []
    for rec in records:
        ks_str = rec.get('ks_boot_all', '')
        ad_str = rec.get('ad_boot_all', '')
        ks_vals = ks_str.split(';') if ks_str else []
        ad_vals = ad_str.split(';') if ad_str else []
        n_max = max(len(ks_vals), len(ad_vals))
        for b_idx in range(n_max):
            try:
                ks_val = float(ks_vals[b_idx]) if b_idx < len(ks_vals) else np.nan
            except Exception:
                ks_val = np.nan
            try:
                ad_val = float(ad_vals[b_idx]) if b_idx < len(ad_vals) else np.nan
            except Exception:
                ad_val = np.nan
            boot_rows.append({
                'Model': rec['Model'],
                'Asset': rec['Asset'],
                'Sign': rec['Sign'],
                'bootstrap_idx': b_idx,
                'D_boot_ks': ks_val,
                'KS_stat_obs': rec['KS_stat_3r'],
                'exceeds_observed_ks': (ks_val >= rec['KS_stat_3r']
                                        if np.isfinite(ks_val) else np.nan),
                'D_boot_ad': ad_val,
                'AD_stat_obs': rec.get('AD_stat_3r', np.nan),
                'exceeds_observed_ad': (ad_val >= rec.get('AD_stat_3r', np.nan)
                                        if (np.isfinite(ad_val) and
                                            np.isfinite(rec.get('AD_stat_3r', np.nan)))
                                        else np.nan),
            })
    if boot_rows:
        df_boot = pd.DataFrame(boot_rows)
        df_boot = df_boot.sort_values(
            ['Model', 'Asset', 'Sign', 'bootstrap_idx']).reset_index(drop=True)
        boot_csv = os.path.join(OUT_DIR, 'bootstrap_details.csv')
        df_boot.to_csv(boot_csv, index=False, encoding='utf-8')
        print(f"✓ Bootstrap details → {boot_csv}  ({len(boot_rows)} rows)")

    # Resumen
    def _pass_rate_for(df_sub, p_col, alpha):
        s = df_sub[p_col].dropna()
        if len(s) == 0:
            return "0/0 (NA)"
        return (f"{int((s > alpha).sum())}/{len(s)} "
                f"({100*(s > alpha).mean():.1f}%)")

    lines = []
    lines.append("=" * 78)
    lines.append("RESUMEN Exp-GΓ-GPD vs Par-GΓ-Par (grillas finas)")
    lines.append("=" * 78)
    lines.append(f"Total records: {len(df)}\n")

    for case in CASES:
        ck = case['key']
        df_sub = df[df['Model'] == ck]
        if df_sub.empty:
            continue
        lines.append("─" * 78)
        lines.append(f"Modelo: {case['label']}  [key={ck}]  (n={len(df_sub)})")
        lines.append("─" * 78)
        lines.append("N° series que pasan el KS test:")
        lines.append(f"                                 α=0.01         α=0.05         α=0.10")
        lines.append(f"  KS_p_3r_asymptotic (clásico):  "
                     f"{_pass_rate_for(df_sub, 'KS_p_3r_asymptotic', 0.01):>14s} "
                     f"{_pass_rate_for(df_sub, 'KS_p_3r_asymptotic', 0.05):>14s} "
                     f"{_pass_rate_for(df_sub, 'KS_p_3r_asymptotic', 0.10):>14s}")
        lines.append(f"  KS_p_3r_lilliefors (bootstrap):"
                     f"{_pass_rate_for(df_sub, 'KS_p_3r_lilliefors', 0.01):>14s} "
                     f"{_pass_rate_for(df_sub, 'KS_p_3r_lilliefors', 0.05):>14s} "
                     f"{_pass_rate_for(df_sub, 'KS_p_3r_lilliefors', 0.10):>14s}")
        lines.append("")
        lines.append("N° series que pasan el AD test:")
        lines.append(f"                                 α=0.01         α=0.05         α=0.10")
        lines.append(f"  AD_p_3r_asymptotic (Marsaglia):"
                     f"{_pass_rate_for(df_sub, 'AD_p_3r_asymptotic', 0.01):>14s} "
                     f"{_pass_rate_for(df_sub, 'AD_p_3r_asymptotic', 0.05):>14s} "
                     f"{_pass_rate_for(df_sub, 'AD_p_3r_asymptotic', 0.10):>14s}")
        lines.append(f"  AD_p_3r_lilliefors (bootstrap):"
                     f"{_pass_rate_for(df_sub, 'AD_p_3r_lilliefors', 0.01):>14s} "
                     f"{_pass_rate_for(df_sub, 'AD_p_3r_lilliefors', 0.05):>14s} "
                     f"{_pass_rate_for(df_sub, 'AD_p_3r_lilliefors', 0.10):>14s}")
        lines.append("")
        lines.append(
            f"  {'Asset':<8s} {'Sign':<10s} "
            f"{'KS_stat':>8s} {'KS_pA':>8s} {'KS_pL':>8s}  "
            f"{'AD_stat':>9s} {'AD_pA':>8s} {'AD_pL':>8s}  "
            f"{'θ_H%':>6s} {'θ_L%':>6s}")
        for _, r in df_sub.sort_values(['Asset', 'Sign']).iterrows():
            lines.append(
                f"  {r['Asset']:<8s} {r['Sign']:<10s} "
                f"{r['KS_stat_3r']:>8.4f} "
                f"{r['KS_p_3r_asymptotic']:>8.4f} "
                f"{r['KS_p_3r_lilliefors']:>8.4f}  "
                f"{r.get('AD_stat_3r', float('nan')):>9.4f} "
                f"{r.get('AD_p_3r_asymptotic', float('nan')):>8.4f} "
                f"{r.get('AD_p_3r_lilliefors', float('nan')):>8.4f}  "
                f"{r['theta_pct']:>6.2f} "
                f"{r['theta_low_pct_opt']:>6.2f}")
        lines.append("")

    summary = "\n".join(lines)
    with open(os.path.join(OUT_DIR, 'summary_pass_rate.txt'),
              'w', encoding='utf-8') as f:
        f.write(summary)
    print("\n" + summary)
    print(f"\n✓ Terminado. Todos los outputs en {OUT_DIR}/")


if __name__ == '__main__':
    main()
