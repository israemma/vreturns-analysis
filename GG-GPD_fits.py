"""
Hybrid fit of GENERALIZED GAMMA (body) + GPD (tail) with metrics,
parallel parametric bootstrap, fine threshold search, standard errors
and plots.

Data:
  - Close prices are loaded from the `financial_data/` folder
    (files `{INDEX}_close.pkl`, each a pandas.Series with
    DatetimeIndex).
  - Each series is filtered to the date range of Table 1:
        DJIA:   1992-01-02 -> 2023-12-29
        DAX:    1992-01-02 -> 2023-12-29
        IPC:    1992-01-02 -> 2023-12-29
        Nikkei: 1992-01-06 -> 2023-12-29
  - VReturns (Def. 5) are computed from the filtered prices:
        VR_m^{k,i} = (log p_{m+i} - log p_m) / i,
    for i = 1, ..., k within each uninterrupted trend (uptrend or
    downtrend) of duration k. Positive VReturns come from uptrends;
    negative VReturns (in absolute value) come from downtrends.

Model:
  - Body (x <= u):  Generalized Gamma GG(a, c, s), MLE.
       PDF: f(x) = c/(s*Gamma(a)) * (x/s)^(a*c-1) * exp(-(x/s)^c)
       scipy: stats.gengamma with parameters (a, c, scale=s), loc=0.
  - Tail (x  > u):  GPD(xi, sigma), MLE over excesses y = x - u.
  - Threshold u: fine search over quantiles, selecting the one with
    the highest min(p-value) across {KS, CvM_body, CvM_tail, AD_tail}.

Standard errors (SE): observed Hessian of the log-likelihood by
finite differences, evaluated at the MLE.

Parallel parametric bootstrap via ProcessPoolExecutor.

Usage:
    python3 Gamma_Generalizada-GPD_fit_fino.py 10           # B=10, 30 workers
    python3 Gamma_Generalizada-GPD_fit_fino.py 50 16        # B=50, 16 workers
    python3 Gamma_Generalizada-GPD_fit_fino.py 10 30 -v     # verbose
"""

import os, sys, time, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from concurrent.futures import ProcessPoolExecutor

plt.rcParams.update({
    'figure.dpi': 110, 'savefig.dpi': 140, 'font.size': 10,
    'axes.grid': True, 'grid.alpha': 0.3,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------
# Data configuration
# ---------------------------------------------------------------
# Folder with the close-price pickle files. Each file is a
# pandas.Series with a DatetimeIndex.
DATA_DIR_NAME = 'financial_data'

# Date ranges from Table 1. Applied to the price series before
# extracting VReturns.
TABLE1_RANGES = {
    'DJIA':   ('1992-01-02', '2023-12-29'),
    'DAX':    ('1992-01-02', '2023-12-29'),
    'IPC':    ('1992-01-02', '2023-12-29'),
    'Nikkei': ('1992-01-06', '2023-12-29'),
}


# ---------------------------------------------------------------
# Price loading and VReturns computation
# ---------------------------------------------------------------
def load_close_prices(name, data_dir):
    """Load `{name}_close.pkl` and apply the Table 1 date filter."""
    path = os.path.join(data_dir, f'{name}_close.pkl')
    with open(path, 'rb') as f:
        ts = pickle.load(f)
    ts = ts.dropna()
    if name in TABLE1_RANGES:
        start_dt, end_dt = TABLE1_RANGES[name]
        ts = ts.loc[start_dt:end_dt]
    return ts


def extract_abs_vr(prices_array):
    """Extract VReturns from a price series.

    Walks the series identifying maximal uninterrupted trends
    (uptrends with p_j >= p_{j-1}; downtrends with p_k <= p_{k-1}).
    For each trend of duration k starting at index s, generates
    VReturns according to Def. 5:
        VR_s^{k,i} = (log p_{s+i} - log p_s) / i,   i = 1, ..., k
    Returns two arrays: positive VReturns and |negative VReturns|.
    """
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


# ---------------------------------------------------------------
# Model: Generalized Gamma (body) + GPD (tail)
# ---------------------------------------------------------------
def fit_gengamma(x):
    """MLE of the Generalized Gamma with loc=0.

    Tries several initial points (a0, c0) to avoid bad local
    optima; the GG likelihood is notoriously hard to optimize.
    """
    best = None
    inits = [(1.0, 1.0), (0.5, 2.0), (2.0, 0.5), (1.0, 2.0), (0.7, 1.5)]
    for a0, c0 in inits:
        try:
            a, c, _, s = stats.gengamma.fit(x, a0, c0, floc=0)
            if not np.isfinite([a, c, s]).all(): continue
            if a <= 0 or c <= 0 or s <= 0: continue
            ll = np.sum(stats.gengamma.logpdf(x, a, c, scale=s))
            if not np.isfinite(ll): continue
            if (best is None) or (ll > best[0]):
                best = (ll, a, c, s)
        except Exception:
            continue
    if best is None:
        raise RuntimeError("GG fit failed for all initial points")
    return best[1], best[2], best[3]   # a, c, s

def fit_gpd(excesses):
    xi, _, sigma = stats.genpareto.fit(excesses, floc=0)
    return xi, sigma

def hybrid_cdf(x, u, a, c, s, xi, sigma, p_tail):
    """Hybrid GG+GPD CDF."""
    x = np.atleast_1d(x).astype(float)
    F_u_gg = stats.gengamma.cdf(u, a, c, scale=s)
    F = np.empty_like(x)
    body = x <= u
    F[body] = stats.gengamma.cdf(x[body], a, c, scale=s) * (1 - p_tail) / F_u_gg
    F[~body] = (1 - p_tail) + p_tail * stats.genpareto.cdf(x[~body] - u, xi, scale=sigma)
    return F

def hybrid_ppf(p, u, a, c, s, xi, sigma, p_tail):
    p = np.atleast_1d(p).astype(float)
    out = np.empty_like(p)
    F_u = 1 - p_tail
    body = p <= F_u
    F_u_gg = stats.gengamma.cdf(u, a, c, scale=s)
    p_body_eq = np.clip(p[body] * F_u_gg / F_u, 1e-12, 1 - 1e-12)
    out[body] = stats.gengamma.ppf(p_body_eq, a, c, scale=s)
    p_exc = np.clip((p[~body] - F_u) / p_tail, 1e-12, 1 - 1e-12)
    out[~body] = u + stats.genpareto.ppf(p_exc, xi, scale=sigma)
    return out

def hybrid_sample(n, params, rng):
    u = rng.uniform(0, 1, n)
    return hybrid_ppf(u, **params)


def fit_at_quantile(x_sorted, q):
    n = len(x_sorted)
    u = np.quantile(x_sorted, q)
    body = x_sorted[x_sorted <= u]
    excess = x_sorted[x_sorted > u] - u
    if len(body) < 50 or len(excess) < 30:
        return None
    try:
        a, c, s = fit_gengamma(body)
        xi, sigma = fit_gpd(excess)
    except Exception:
        return None
    if not np.isfinite([a, c, s, xi, sigma]).all() or sigma <= 0:
        return None
    p_tail = len(excess) / n
    return dict(u=u, a=a, c=c, s=s, xi=xi, sigma=sigma, p_tail=p_tail)


# ---------------------------------------------------------------
# Standard errors via observed Hessian
# ---------------------------------------------------------------
def _numeric_hessian(loglik, theta, eps_rel=1e-4):
    k = len(theta)
    H = np.zeros((k, k))
    eps = np.array([max(abs(t), 1e-3) * eps_rel for t in theta])
    f0 = loglik(theta)
    if not np.isfinite(f0):
        return None
    for i in range(k):
        tp = theta.copy(); tp[i] += eps[i]
        tm = theta.copy(); tm[i] -= eps[i]
        H[i, i] = (loglik(tp) - 2*f0 + loglik(tm)) / eps[i]**2
    for i in range(k):
        for j in range(i+1, k):
            tpp = theta.copy(); tpp[i] += eps[i]; tpp[j] += eps[j]
            tpm = theta.copy(); tpm[i] += eps[i]; tpm[j] -= eps[j]
            tmp = theta.copy(); tmp[i] -= eps[i]; tmp[j] += eps[j]
            tmm = theta.copy(); tmm[i] -= eps[i]; tmm[j] -= eps[j]
            H[i, j] = (loglik(tpp) - loglik(tpm) - loglik(tmp) + loglik(tmm)) / (4*eps[i]*eps[j])
            H[j, i] = H[i, j]
    return H

def standard_errors(x_sorted, params):
    """SE for (a, c, s, xi, sigma)."""
    u = params['u']
    body = x_sorted[x_sorted <= u]
    excess = x_sorted[x_sorted > u] - u
    se = {'se_a': np.nan, 'se_c': np.nan, 'se_s': np.nan,
          'se_xi': np.nan, 'se_sigma': np.nan}

    # GG
    def ll_gg(theta):
        a, c, s = theta
        if a <= 0 or c <= 0 or s <= 0: return -np.inf
        val = stats.gengamma.logpdf(body, a, c, scale=s)
        if not np.all(np.isfinite(val)): return -np.inf
        return np.sum(val)
    try:
        H = _numeric_hessian(
            ll_gg,
            np.array([params['a'], params['c'], params['s']]),
        )
        if H is not None:
            cov = np.linalg.inv(-H)
            if np.all(np.diag(cov) > 0):
                se['se_a'] = np.sqrt(cov[0, 0])
                se['se_c'] = np.sqrt(cov[1, 1])
                se['se_s'] = np.sqrt(cov[2, 2])
    except Exception:
        pass

    # GPD
    def ll_gpd(theta):
        xi, sg = theta
        if sg <= 0: return -np.inf
        val = stats.genpareto.logpdf(excess, xi, scale=sg)
        if not np.all(np.isfinite(val)): return -np.inf
        return np.sum(val)
    try:
        H = _numeric_hessian(
            ll_gpd,
            np.array([params['xi'], params['sigma']]),
        )
        if H is not None:
            cov = np.linalg.inv(-H)
            if np.all(np.diag(cov) > 0):
                se['se_xi'] = np.sqrt(cov[0, 0])
                se['se_sigma'] = np.sqrt(cov[1, 1])
    except Exception:
        pass
    return se


# ---------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------
def metric_W1(x_sorted, params):
    n = len(x_sorted)
    p_grid = (np.arange(1, n + 1) - 0.5) / n
    q_mod = hybrid_ppf(p_grid, **params)
    return np.mean(np.abs(x_sorted - q_mod))

def metric_KS(x_sorted, params):
    n = len(x_sorted)
    F_emp_up = np.arange(1, n + 1) / n
    F_emp_lo = np.arange(0, n) / n
    F_mod = hybrid_cdf(x_sorted, **params)
    return max(np.max(F_emp_up - F_mod), np.max(F_mod - F_emp_lo))

def metric_CvM_subset(x_sub, F_sub):
    m = len(x_sub)
    if m == 0: return np.nan
    i = np.arange(1, m + 1)
    return 1/(12*m) + np.sum((F_sub - (2*i - 1) / (2*m))**2)

def metric_AD_subset(x_sub, F_sub):
    m = len(x_sub)
    if m == 0: return np.nan
    F = np.clip(F_sub, 1e-15, 1 - 1e-15)
    i = np.arange(1, m + 1)
    S = np.sum((2*i - 1) / m * (np.log(F) + np.log(1 - F[::-1])))
    return -m - S

def all_metrics(x_sorted, params):
    u = params['u']
    F_mod = hybrid_cdf(x_sorted, **params)
    body_mask = x_sorted <= u
    tail_mask = ~body_mask
    return {
        'W1':       metric_W1(x_sorted, params),
        'KS':       metric_KS(x_sorted, params),
        'CvM_body': metric_CvM_subset(x_sorted[body_mask], F_mod[body_mask]),
        'CvM_tail': metric_CvM_subset(x_sorted[tail_mask], F_mod[tail_mask]),
        'AD_tail':  metric_AD_subset(x_sorted[tail_mask], F_mod[tail_mask]),
    }


# ---------------------------------------------------------------
# Single bootstrap replica (module-level, picklable)
# ---------------------------------------------------------------
def _bootstrap_replica(args):
    """Return (metrics_dict, params_dict) or None on failure.
    The returned params_dict comes from the refit on each replica
    and is used to estimate bootstrap SEs of the parameters.
    """
    n, params, seed, refit_q_grid = args
    rng = np.random.default_rng(seed)
    try:
        sample = hybrid_sample(n, params, rng)
        sample = sample[np.isfinite(sample) & (sample > 0)]
        if len(sample) < 200: return None
        sample = np.sort(sample)
        ranks = np.arange(1, len(sample) + 1)
        S_emp = 1 - ranks / (len(sample) + 1)
        log_S_emp = np.log(S_emp)
        best = None
        for q in refit_q_grid:
            pb = fit_at_quantile(sample, q)
            if pb is None: continue
            S_mod = np.clip(1 - hybrid_cdf(sample, **pb), 1e-15, 1)
            mse = np.mean((np.log(S_mod) - log_S_emp) ** 2)
            if (best is None) or (mse < best[0]):
                best = (mse, pb)
        if best is None: return None
        return (all_metrics(sample, best[1]), best[1])
    except Exception:
        return None


def _eval_candidate(args):
    """Step 1: for a given (x, q) fit and return (score, q, params, m)."""
    x_sorted, q = args
    params = fit_at_quantile(x_sorted, q)
    if params is None: return None
    m = all_metrics(x_sorted, params)
    score = m['KS'] + m['CvM_body'] + m['CvM_tail'] + 0.01 * m['AD_tail']
    return (score, q, params, m)


def bootstrap_pvalues_parallel(x_sorted, params, B, seed, executor,
                                refit_q_grid=None):
    if refit_q_grid is None:
        # Smaller grid than the main one: each replica must stay fast.
        refit_q_grid = np.arange(0.90, 0.99, 0.02)   # 5 quantiles
    n = len(x_sorted)
    obs = all_metrics(x_sorted, params)
    tasks = [(n, params, seed + b, refit_q_grid) for b in range(B)]
    results = list(executor.map(_bootstrap_replica, tasks))
    boot = {k: [] for k in obs}
    # Parameters refit on each replica (for bootstrap SEs)
    boot_params = {'a': [], 'c': [], 's': [],
                   'xi': [], 'sigma': [], 'u': []}
    n_ok = 0
    for r in results:
        if r is None: continue
        n_ok += 1
        m_r, p_r = r
        for k in boot:
            if np.isfinite(m_r[k]): boot[k].append(m_r[k])
        for k in boot_params:
            if k in p_r and np.isfinite(p_r[k]):
                boot_params[k].append(p_r[k])
    pvals = {}
    for k, o in obs.items():
        arr = np.array(boot[k])
        pvals[k] = (1 + np.sum(arr >= o)) / (1 + len(arr)) if len(arr) > 0 else np.nan
    return obs, pvals, n_ok, boot_params


# ---------------------------------------------------------------
# Fine threshold search
# ---------------------------------------------------------------
def find_best_threshold_parallel(x, B, seed, executor,
                                  q_grid_fine=None, top_k=5, verbose=False):
    x = np.sort(np.asarray(x, float))
    x = x[x > 0]
    n = len(x)
    if q_grid_fine is None:
        # Bounded range around the typical optimum (q* ~ 0.965-0.970).
        q_grid_fine = np.arange(0.955, 0.975 + 1e-9, 0.0025)   # 9 candidates

    # Step 1 (parallel): cheap score without bootstrap
    tasks = [(x, q) for q in q_grid_fine]
    results = list(executor.map(_eval_candidate, tasks))
    cands = [r for r in results if r is not None]
    if not cands: return None
    cands.sort(key=lambda t: t[0])
    top = cands[:top_k]
    if verbose:
        print(f"  Top-{top_k} candidates (step 1):")
        for sc, q, p, m in top:
            print(f"    q={q:.4f} u={p['u']:.5f} "
                  f"GG(a={p['a']:.2f},c={p['c']:.2f},s={p['s']:.4f}) "
                  f"GPD(xi={p['xi']:+.2f},sig={p['sigma']:.4f}) "
                  f"KS={m['KS']:.4f} CvMb={m['CvM_body']:.4f} "
                  f"CvMt={m['CvM_tail']:.4f}")

    # Step 2: bootstrap only on top_k
    best = None
    for i, (sc, q, params, _) in enumerate(top):
        obs, pv, n_ok, boot_params = bootstrap_pvalues_parallel(
            x, params, B=B, seed=seed + 10000 * i, executor=executor,
        )
        pmin = min(pv['KS'], pv['CvM_body'], pv['CvM_tail'], pv['AD_tail'])
        # Bootstrap SE: standard deviation of refits across B replicas
        se_boot = {}
        for k, arr in boot_params.items():
            arr = np.asarray(arr, float)
            se_boot[k] = float(np.std(arr, ddof=1)) if len(arr) >= 2 else np.nan
        if verbose:
            print(f"    q={q:.4f}  pmin={pmin:.3f}  "
                  f"pKS={pv['KS']:.3f} pCvMb={pv['CvM_body']:.3f} "
                  f"pCvMt={pv['CvM_tail']:.3f} pADt={pv['AD_tail']:.3f} "
                  f"ok={n_ok}/{B}")
        rec = dict(q=q, params=params, obs=obs, pvals=pv,
                   n_ok=n_ok, pmin=pmin, se_boot=se_boot)
        if (best is None) or (pmin > best['pmin']):
            best = rec
    return dict(n=n, x=x, best=best)


# ---------------------------------------------------------------
# Consolidated plots
# Three 4x2 figures: log-linear CCDF, log-log CCDF, QQ-plot.
# Body (x <= u) in blue (Generalized Gamma).
# Tail (x  > u) in red (Generalized Pareto).
# Empirical data in black as scattered points (no lines).
# ---------------------------------------------------------------

_PLOT_STYLE_SET = False

def _set_plot_style():
    """Apply seaborn-whitegrid style with serif fonts. Idempotent."""
    global _PLOT_STYLE_SET
    if _PLOT_STYLE_SET:
        return
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        plt.style.use('seaborn-whitegrid')
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
    _PLOT_STYLE_SET = True


# Colors
_C_BODY      = '#1f77b4'   # blue -> Generalized Gamma
_C_TAIL      = '#d62728'   # red  -> Generalized Pareto
_C_EMPIRICAL = '#000000'   # black

# Row order in the consolidated plots
_PLOT_ORDER = ['DJIA', 'DAX', 'IPC', 'Nikkei']


def _add_legend(ax, loc='best'):
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


def _subsample(n, target=600):
    if n <= target:
        return np.arange(n)
    step = max(1, n // target)
    return np.arange(0, n, step)


def _ccdf_panel(ax, obs_sorted, params, sign, scale='loglinear'):
    """Empirical CCDF + hybrid model curve in two colors."""
    n = len(obs_sorted)
    y_emp = (n - np.arange(1, n + 1) + 0.5) / (n + 1)

    # Empirical: scattered points only (no connecting line)
    s_all = _subsample(n)
    ax.plot(obs_sorted[s_all], y_emp[s_all], '.', color=_C_EMPIRICAL,
            ms=2.2, alpha=0.9, zorder=2, label=f'Empirical (n={n})')

    # Model: body in blue (x<=u), tail in red (x>u)
    u = params['u']
    x_min, x_max = obs_sorted.min(), obs_sorted.max() * 1.02
    xf = np.linspace(x_min, x_max, 2000)
    ccdf = np.clip(1.0 - hybrid_cdf(xf, **params), 1e-300, 1.0)
    seg_body = xf <= u
    seg_tail = xf > u

    def _masked(mask):
        return np.where(mask, ccdf, np.nan)

    ax.plot(xf, _masked(seg_body), '-', color=_C_BODY, lw=2.0,
            alpha=0.9, zorder=3, label='Generalized Gamma')
    ax.plot(xf, _masked(seg_tail), '-', color=_C_TAIL, lw=2.0,
            alpha=0.9, zorder=3, label='Generalized Pareto')
    ax.axvline(x=u, color='gray', ls='--', lw=0.8, alpha=0.55,
               label=f'u = {u:.4f}')

    if scale == 'loglinear':
        ax.set_yscale('log')
        xl_base = ('VReturns' if sign == 'positive'
                   else r'$|$VReturns$^{-}|$')
        ax.set_xlabel(xl_base)
        ax.set_ylabel('CCDF (log)')
    elif scale == 'loglog':
        ax.set_xscale('log')
        ax.set_yscale('log')
        xl_base = ('VReturns (log)' if sign == 'positive'
                   else r'$|$VReturns$^{-}|$ (log)')
        ax.set_xlabel(xl_base)
        ax.set_ylabel('CCDF (log)')

    ax.grid(True, which='both', ls=':', alpha=0.3)
    _sane_ylim_log(ax, counts_min=0.2/n, floor_decades=6)


def _qq_panel(ax, obs_sorted, params, sign):
    """Q-Q plot: model quantiles (x) vs empirical (y) in two colors."""
    n = len(obs_sorted)
    p = (np.arange(1, n + 1) - 0.5) / n
    q_mod = hybrid_ppf(p, **params)

    u = params['u']
    mB = obs_sorted <= u
    mT = obs_sorted > u

    lo = min(np.nanmin(q_mod), obs_sorted.min())
    hi = max(np.nanmax(q_mod), obs_sorted.max())

    # Identity line
    ax.plot([lo, hi], [lo, hi], '-', color=_C_EMPIRICAL, lw=1.1,
            alpha=0.85, zorder=1, label='Perfect fit (y=x)')

    ax.scatter(q_mod[mB], obs_sorted[mB], marker='o', s=10,
               color=_C_BODY, alpha=0.6, edgecolors='none', zorder=3,
               label='Generalized Gamma')
    ax.scatter(q_mod[mT], obs_sorted[mT], marker='o', s=14,
               color=_C_TAIL, alpha=0.8, edgecolors='none', zorder=3,
               label='Generalized Pareto')

    # Log scale when the range spans more than two decades
    if obs_sorted.max() / max(obs_sorted.min(), 1e-12) > 100:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Model quantiles (log)')
        ax.set_ylabel('Empirical quantiles (log)')
    else:
        ax.set_xlabel('Model quantiles')
        ax.set_ylabel('Empirical quantiles')
    ax.grid(True, ls=':', alpha=0.3)


def _make_consolidated_plot(all_fits, kind, outdir, indices_order):
    """One 4x2 figure: rows = indices, columns = [positive, negative].
    `kind` in {'ccdf_loglinear', 'ccdf_loglog', 'qq'}.
    """
    _set_plot_style()
    nrows, ncols = len(indices_order), 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.0 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    for i, name in enumerate(indices_order):
        if name not in all_fits:
            for j in (0, 1):
                axes[i, j].set_axis_off()
            continue
        for j, side in enumerate(['positive', 'negative']):
            ax = axes[i, j]
            info = all_fits[name].get(side)
            if info is None:
                ax.set_axis_off()
                continue
            x = info['x_sorted']
            params = info['params']

            if kind == 'ccdf_loglinear':
                _ccdf_panel(ax, x, params, side, scale='loglinear')
            elif kind == 'ccdf_loglog':
                _ccdf_panel(ax, x, params, side, scale='loglog')
            elif kind == 'qq':
                _qq_panel(ax, x, params, side)

            sign_dir = 'Positive' if side == 'positive' else 'Negative'
            if kind in ('ccdf_loglinear', 'ccdf_loglog'):
                ax.set_title(f'{name} {sign_dir} VReturns (CCDF)')
            elif kind == 'qq':
                ax.set_title(f'{name} {sign_dir} VReturns (QQ-plot)')

            # Legend only on the top row
            if i == 0:
                if kind == 'ccdf_loglog':
                    _add_legend(ax, 'lower left')
                elif kind == 'ccdf_loglinear':
                    _add_legend(ax, 'upper right')
                else:  # qq
                    _add_legend(ax, 'upper left')

    fig.tight_layout()
    fname_map = {
        'ccdf_loglinear': 'ccdf_loglinear.png',
        'ccdf_loglog':    'ccdf_loglog.png',
        'qq':             'qq_plots.png',
    }
    out = os.path.join(outdir, fname_map[kind])
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return out


def make_all_consolidated_plots(all_fits, outdir, indices):
    """Generate the three consolidated figures (log-lin CCDF, log-log, QQ)."""
    order = [n for n in _PLOT_ORDER if n in all_fits]
    for n in indices:
        if n not in order and n in all_fits:
            order.append(n)
    outs = []
    for kind in ('ccdf_loglinear', 'ccdf_loglog', 'qq'):
        outs.append(_make_consolidated_plot(all_fits, kind, outdir, order))
    return outs


# ---------------------------------------------------------------
# Driver
# ---------------------------------------------------------------
def run_all(B=10, n_workers=30, indices=('IPC', 'Nikkei', 'DAX', 'DJIA'),
            data_dir=None, seed=42, verbose=False, make_plots=True,
            q_grid_fine=None, outdir_name='GG_body_GPD_tail_VReturns_fit',
            executor=None, csv_name=None, q_overrides=None):
    """If `executor` is passed, it is reused (no pool is created/closed here).
    `q_grid_fine` is forwarded to find_best_threshold_parallel.
    `outdir_name` controls the output subfolder for plots and CSV.
    `q_overrides`: optional dict {(index, side): q} to force a specific
    quantile for certain cases (overrides `q_grid_fine`).
    """
    if data_dir is None:
        data_dir = os.path.join(SCRIPT_DIR, DATA_DIR_NAME)
    if q_overrides is None:
        q_overrides = {}
    rows = []
    all_fits = {name: {} for name in indices}

    owns_executor = executor is None
    if owns_executor:
        executor = ProcessPoolExecutor(max_workers=n_workers)
    try:
        ex = executor
        for idx_i, name in enumerate(indices):
            # Load close prices and apply the Table 1 date filter.
            try:
                prices = load_close_prices(name, data_dir)
            except Exception as e:
                print(f"[WARN] Could not load {name}: {e}", flush=True)
                continue
            if verbose:
                print(f"\n>>> {name}: {len(prices)} prices "
                      f"({prices.index[0].date()} -> "
                      f"{prices.index[-1].date()})", flush=True)
            # Compute VReturns from the filtered prices.
            # `pos` are positive VReturns; `neg` are |VR-|, both > 0.
            pos, neg = extract_abs_vr(prices.values)
            for side, x in [('positive', pos), ('negative', neg)]:
                t0 = time.time()
                label = f"{name:<6s} {side}"
                if verbose: print(f"\n>>> {label}  (n_vr={len(x)})")
                case_q_grid = q_grid_fine
                if (name, side) in q_overrides:
                    q_ov = q_overrides[(name, side)]
                    case_q_grid = np.array([q_ov])
                    if verbose:
                        print(f"    [override] using q={q_ov:.4f}")
                res = find_best_threshold_parallel(
                    x, B=B,
                    seed=seed + 1000*idx_i + (0 if side == 'positive' else 1),
                    executor=ex, verbose=verbose,
                    q_grid_fine=case_q_grid,
                )
                b = res['best']
                p, obs, pv = b['params'], b['obs'], b['pvals']
                dt = time.time() - t0
                line = (f"{name:<7s} {side:<8s} | n={res['n']:>5d} "
                        f"| W1={obs['W1']:.5f} "
                        f"| AD_tail p={pv['AD_tail']:.3f} "
                        f"| CvM_tail p={pv['CvM_tail']:.3f} "
                        f"| CvM_body p={pv['CvM_body']:.3f} "
                        f"| KS p={pv['KS']:.3f} "
                        f"| GG(a={p['a']:.2f},c={p['c']:.2f},s={p['s']:.4f}) "
                        f"GPD(xi={p['xi']:+.2f},sig={p['sigma']:.4f}) "
                        f"u={p['u']:.4f} q={b['q']:.4f} "
                        f"| ok={b['n_ok']}/{B} | {dt:.1f}s")
                print(line, flush=True)
                se = standard_errors(res['x'], p)
                seb = b.get('se_boot', {})
                x_arr = res['x']
                n_body = int(np.sum(x_arr <= p['u']))
                n_tail = int(np.sum(x_arr >  p['u']))
                # alpha = 1/xi (Pareto-type tail index). Hessian SE via
                # delta-method; bootstrap SE also via delta-method.
                xi_, se_xi_ = p['xi'], se['se_xi']
                if xi_ > 0 and np.isfinite(se_xi_):
                    alpha_val = 1.0 / xi_
                    se_alpha  = se_xi_ / (xi_ ** 2)
                else:
                    alpha_val = np.nan
                    se_alpha  = np.nan
                se_boot_xi = seb.get('xi', np.nan)
                if xi_ > 0 and np.isfinite(se_boot_xi):
                    se_boot_alpha = se_boot_xi / (xi_ ** 2)
                else:
                    se_boot_alpha = np.nan
                rows.append(dict(
                    index=name, side=side,
                    # sample sizes per part
                    n=res['n'], n_body=n_body, n_tail=n_tail,
                    # threshold
                    q_star=b['q'], u=p['u'], p_tail=p['p_tail'],
                    # GG (body) with Hessian SE and bootstrap SE
                    gg_a=p['a'], se_gg_a=se['se_a'], se_boot_gg_a=seb.get('a', np.nan),
                    gg_c=p['c'], se_gg_c=se['se_c'], se_boot_gg_c=seb.get('c', np.nan),
                    gg_s=p['s'], se_gg_s=se['se_s'], se_boot_gg_s=seb.get('s', np.nan),
                    # GPD (tail) with Hessian SE and bootstrap SE
                    gpd_xi=p['xi'],       se_gpd_xi=se['se_xi'],
                    se_boot_gpd_xi=seb.get('xi', np.nan),
                    gpd_sigma=p['sigma'], se_gpd_sigma=se['se_sigma'],
                    se_boot_gpd_sigma=seb.get('sigma', np.nan),
                    # Tail index alpha = 1/xi with both SEs (delta-method)
                    alpha_tail=alpha_val, se_alpha_tail=se_alpha,
                    se_boot_alpha_tail=se_boot_alpha,
                    # observed metrics
                    W1=obs['W1'], KS=obs['KS'],
                    CvM_body=obs['CvM_body'], CvM_tail=obs['CvM_tail'],
                    AD_tail=obs['AD_tail'],
                    # bootstrap p-values
                    p_KS=pv['KS'], p_CvM_body=pv['CvM_body'],
                    p_CvM_tail=pv['CvM_tail'], p_AD_tail=pv['AD_tail'],
                    pmin=b['pmin'],
                    # meta
                    ok=b['n_ok'], B=B, seconds=dt,
                ))
                all_fits[name][side] = dict(
                    x_sorted=res['x'], params=p, q=b['q'],
                )
    finally:
        if owns_executor:
            executor.shutdown()

    df_out = pd.DataFrame(rows)

    if make_plots:
        outdir = os.path.join(SCRIPT_DIR, outdir_name)
        os.makedirs(outdir, exist_ok=True)
        print(f"\nGenerating consolidated figures in {outdir} ...")
        outs = make_all_consolidated_plots(all_fits, outdir, indices)
        for out in outs:
            print(f"  -> {out}")
        name_csv = csv_name or f'metrics_gengamma_gpd_B{B}.csv'
        out_csv = os.path.join(outdir, name_csv)
        df_out.to_csv(out_csv, index=False)
        print(f"  -> {out_csv}")

    return df_out


if __name__ == "__main__":
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    verbose = '-v' in sys.argv

    # Base quantile: 0.92 for all cases.
    # For cases where q=0.92 rejects the GPD fit (p < 0.05 on CvM_tail
    # or AD_tail with B=1000), use q=0.90 which includes more data in
    # the tail and improves the fit.
    quantiles = [0.92]
    q_overrides = {
        ('Nikkei', 'positive'): 0.90,
        ('DJIA',   'negative'): 0.90,
    }

    print(f"Parametric bootstrap B={B} | workers={n_workers} | "
          f"base quantiles {quantiles}", flush=True)
    print(f"Data: {DATA_DIR_NAME}/  (close prices, Table 1 dates)",
          flush=True)
    print(f"Overrides: {q_overrides}", flush=True)
    print("="*78, flush=True)
    t0 = time.time()

    # Single pool reused across all runs.
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for q_fix in quantiles:
            q_grid = np.array([q_fix])
            outdir_name = (f'GG_body_GPD_tail_VReturns_fit_'
                           f'q{int(round(q_fix*100)):02d}')
            print(f"\n{'#'*78}\n# Base quantile q = {q_fix:.2f}  ->  "
                  f"{outdir_name}/\n{'#'*78}", flush=True)
            t1 = time.time()
            run_all(B=B, n_workers=n_workers, verbose=verbose,
                    q_grid_fine=q_grid, outdir_name=outdir_name,
                    executor=ex, q_overrides=q_overrides)
            print(f"\n[q={q_fix:.2f}] time: {time.time()-t1:.1f}s",
                  flush=True)

    print(f"\nTotal time: {time.time() - t0:.1f}s")