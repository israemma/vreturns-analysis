"""
Hybrid model comparison: GG-GPD, Gamma-GPD, Log-Normal-GPD.

Fits and goodness-of-fit tests are performed on ALL VReturns extracted
from the price series (Definition 5: VR_s^{k,i} = (log p_{s+i} - log p_s)/i
for i = 1, ..., k within each maximal trend of duration k).

Methodology:
  Fits      : MLE on all VReturns.
  Tests     : parametric i.i.d. bootstrap with B replicas, each refit.
  Metrics   : W^2 (Cramer-von Mises), A^2 (Anderson-Darling), KS,
              computed on the conditional CDF renormalized to [0,1]
              within each subset (body and tail). The resulting W^2
              and A^2 are on the classical Stephens scale, with a 5%
              critical value W^2_0.05 ~ 0.461.
  SEs       : bootstrap standard errors of the model parameters,
              obtained as the sample standard deviation of the refits
              across the B replicas.
  Tail idx  : Pareto tail index alpha = 1/xi (defined when xi > 0),
              with SE by the delta method SE(alpha) = SE(xi) / xi^2.

Outputs in compare_3models_allVR_B{B}/:
  compare_3models_allVR_B{B}.csv : metrics, p-values and bootstrap SEs.
  ccdf_loglinear.png             : empirical CCDF + GG body + GPD tail.
  ccdf_loglog.png                : same, log-log axes.
  qq_plots.png                   : Q-Q plot for GG, body and tail.
  qq_3models.png                 : Q-Q overlay for the three models.

Usage:
    python3 compare_3models_allVR.py 1000       # B=1000, 30 workers
    python3 compare_3models_allVR.py 1000 8     # B=1000, 8 workers
    python3 compare_3models_allVR.py 1000 30 -v # verbose
"""

import os, sys, time, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from concurrent.futures import ProcessPoolExecutor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR_NAME = 'financial_data'

# Date ranges of the price series (Table 1 of the paper).
TABLE1_RANGES = {
    'DJIA':   ('1992-01-02', '2023-12-29'),
    'DAX':    ('1992-01-02', '2023-12-29'),
    'IPC':    ('1992-01-02', '2023-12-29'),
    'Nikkei': ('1992-01-06', '2023-12-29'),
}


# ---------------------------------------------------------------
# Data loading and VReturns extraction
# ---------------------------------------------------------------
def load_close_prices(name, data_dir):
    """Load `{name}_close.pkl` and filter to the Table 1 date range."""
    path = os.path.join(data_dir, f'{name}_close.pkl')
    with open(path, 'rb') as f:
        ts = pickle.load(f)
    ts = ts.dropna()
    if name in TABLE1_RANGES:
        s, e = TABLE1_RANGES[name]
        ts = ts.loc[s:e]
    return ts


def extract_abs_vr(prices_array):
    """Extract all VReturns from a price series.

    For each maximal trend of duration k starting at index s, generate
    VR_s^{k,i} = (log p_{s+i} - log p_s) / i for i = 1, ..., k.
    Returns two arrays: positive VReturns and absolute values of
    negative VReturns.
    """
    pv = np.asarray(prices_array, dtype=float).ravel()
    lp = np.log(pv)
    trends = []
    idx = 0
    while idx < len(pv) - 1:
        j = idx + 1
        while j < len(pv) and pv[j] >= pv[j-1]:
            j += 1
        if j - 1 > idx:
            trends.append((idx, j-1)); idx = j-1; continue
        k = idx + 1
        while k < len(pv) and pv[k] <= pv[k-1]:
            k += 1
        if k - 1 > idx:
            trends.append((idx, k-1)); idx = k-1; continue
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
# Body distributions and shared GPD tail
# ---------------------------------------------------------------
class GGBody:
    name = 'GG'
    params_keys = ['a', 'c', 's']

    @staticmethod
    def fit(x):
        """MLE of the Generalized Gamma with multiple starting points."""
        best = None
        for a0, c0 in [(1.0, 1.0), (0.5, 2.0), (2.0, 0.5),
                       (1.0, 2.0), (0.7, 1.5)]:
            try:
                a, c, _, s = stats.gengamma.fit(x, a0, c0, floc=0)
                if not np.isfinite([a, c, s]).all(): continue
                if a <= 0 or c <= 0 or s <= 0: continue
                ll = np.sum(stats.gengamma.logpdf(x, a, c, scale=s))
                if not np.isfinite(ll): continue
                if best is None or ll > best[0]:
                    best = (ll, a, c, s)
            except Exception:
                continue
        if best is None:
            raise RuntimeError("GG fit failed for all initial points")
        return {'a': float(best[1]), 'c': float(best[2]), 's': float(best[3])}

    @staticmethod
    def cdf(x, p): return stats.gengamma.cdf(x, p['a'], p['c'], scale=p['s'])
    @staticmethod
    def ppf(q, p): return stats.gengamma.ppf(q, p['a'], p['c'], scale=p['s'])


class GammaBody:
    name = 'Gamma'
    params_keys = ['a', 's']

    @staticmethod
    def fit(x):
        """MLE of the Gamma distribution with loc=0."""
        a, _, s = stats.gamma.fit(x, floc=0)
        if not np.isfinite([a, s]).all() or a <= 0 or s <= 0:
            raise RuntimeError("Gamma fit failed")
        return {'a': float(a), 's': float(s)}

    @staticmethod
    def cdf(x, p): return stats.gamma.cdf(x, p['a'], scale=p['s'])
    @staticmethod
    def ppf(q, p): return stats.gamma.ppf(q, p['a'], scale=p['s'])


class LNBody:
    name = 'LN'
    params_keys = ['mu', 'sigma']

    @staticmethod
    def fit(x):
        """MLE of the Log-Normal distribution with loc=0."""
        s, _, sc = stats.lognorm.fit(x, floc=0)
        if not np.isfinite([s, sc]).all() or s <= 0 or sc <= 0:
            raise RuntimeError("LN fit failed")
        return {'mu': float(np.log(sc)), 'sigma': float(s)}

    @staticmethod
    def cdf(x, p): return stats.lognorm.cdf(x, p['sigma'], scale=np.exp(p['mu']))
    @staticmethod
    def ppf(q, p): return stats.lognorm.ppf(q, p['sigma'], scale=np.exp(p['mu']))


BODIES = {'GG': GGBody, 'Gamma': GammaBody, 'LN': LNBody}


# ---------------------------------------------------------------
# Hybrid model: body distribution + GPD tail
# ---------------------------------------------------------------
def hybrid_cdf(x, params, Body):
    """CDF of the hybrid body + GPD model."""
    x = np.atleast_1d(x).astype(float)
    u, p_tail = params['u'], params['p_tail']
    F_u_body = Body.cdf(u, params['body'])
    F = np.empty_like(x)
    body = x <= u
    F[body] = Body.cdf(x[body], params['body']) * (1 - p_tail) / F_u_body
    F[~body] = (1 - p_tail) + p_tail * stats.genpareto.cdf(
        x[~body] - u, params['xi'], scale=params['sigma_gpd'])
    return F


def hybrid_ppf(q, params, Body):
    """Quantile function of the hybrid body + GPD model."""
    q = np.atleast_1d(q).astype(float)
    out = np.empty_like(q)
    u, p_tail = params['u'], params['p_tail']
    F_u = 1 - p_tail
    body = q <= F_u
    F_u_body = Body.cdf(u, params['body'])
    p_body_eq = np.clip(q[body] * F_u_body / F_u, 1e-12, 1 - 1e-12)
    out[body] = Body.ppf(p_body_eq, params['body'])
    p_exc = np.clip((q[~body] - F_u) / p_tail, 1e-12, 1 - 1e-12)
    out[~body] = u + stats.genpareto.ppf(
        p_exc, params['xi'], scale=params['sigma_gpd'])
    return out


def hybrid_sample(n, params, Body, rng):
    """Draw a sample of size n from the hybrid model via inverse CDF."""
    u = rng.uniform(0, 1, n)
    return hybrid_ppf(u, params, Body)


def fit_at_quantile(x_sorted, q, Body):
    """Fit body + GPD tail with threshold at the q-th quantile."""
    n = len(x_sorted)
    u = float(np.quantile(x_sorted, q))
    body_x = x_sorted[x_sorted <= u]
    excess = x_sorted[x_sorted > u] - u
    if len(body_x) < 30 or len(excess) < 15:
        return None
    try:
        body_p = Body.fit(body_x)
        xi, _, sigma_gpd = stats.genpareto.fit(excess, floc=0)
    except Exception:
        return None
    if not np.isfinite([xi, sigma_gpd]).all() or sigma_gpd <= 0:
        return None
    return dict(u=u, body=body_p, xi=float(xi), sigma_gpd=float(sigma_gpd),
                p_tail=len(excess) / n)


# ---------------------------------------------------------------
# Goodness-of-fit statistics on the renormalized CDF
# ---------------------------------------------------------------
def metric_KS_subset(x_sub, F_sub):
    """Kolmogorov-Smirnov statistic on a subset."""
    m = len(x_sub)
    if m == 0: return np.nan
    F_emp_up = np.arange(1, m + 1) / m
    F_emp_lo = np.arange(0, m) / m
    return max(np.max(F_emp_up - F_sub), np.max(F_sub - F_emp_lo))


def metric_CvM_subset(x_sub, F_sub):
    """Cramer-von Mises statistic W^2 on a subset."""
    m = len(x_sub)
    if m == 0: return np.nan
    i = np.arange(1, m + 1)
    return 1.0/(12*m) + np.sum((F_sub - (2*i - 1)/(2*m))**2)


def metric_AD_subset(x_sub, F_sub):
    """Anderson-Darling statistic A^2 on a subset."""
    m = len(x_sub)
    if m == 0: return np.nan
    F = np.clip(F_sub, 1e-15, 1 - 1e-15)
    i = np.arange(1, m + 1)
    S = np.sum((2*i - 1)/m * (np.log(F) + np.log(1 - F[::-1])))
    return -m - S


def all_metrics(x_sorted, params, Body):
    """Compute KS, W^2 and A^2 separately for body and tail.

    The hybrid CDF is renormalized to [0,1] within each subset, so the
    resulting W^2 and A^2 are on the classical Stephens scale.
    """
    u, p_tail = params['u'], params['p_tail']
    F_mod = hybrid_cdf(x_sorted, params, Body)
    body_mask = x_sorted <= u
    tail_mask = ~body_mask
    F_u = 1.0 - p_tail
    F_body_norm = np.clip(F_mod[body_mask] / F_u, 0.0, 1.0)
    F_tail_norm = np.clip((F_mod[tail_mask] - F_u) / p_tail, 0.0, 1.0)
    return {
        'KS_body':  metric_KS_subset(x_sorted[body_mask], F_body_norm),
        'KS_tail':  metric_KS_subset(x_sorted[tail_mask], F_tail_norm),
        'CvM_body': metric_CvM_subset(x_sorted[body_mask], F_body_norm),
        'CvM_tail': metric_CvM_subset(x_sorted[tail_mask], F_tail_norm),
        'AD_body':  metric_AD_subset(x_sorted[body_mask], F_body_norm),
        'AD_tail':  metric_AD_subset(x_sorted[tail_mask], F_tail_norm),
    }


# ---------------------------------------------------------------
# Parametric bootstrap with refit per replica
# ---------------------------------------------------------------
def _replica(args):
    """One bootstrap replica: sample, refit, return metrics and params."""
    n, params, body_name, q, seed = args
    Body = BODIES[body_name]
    rng = np.random.default_rng(seed)
    try:
        s = hybrid_sample(n, params, Body, rng)
        s = s[np.isfinite(s) & (s > 0)]
        if len(s) < 100:
            return None
        s = np.sort(s)
        pb = fit_at_quantile(s, q, Body)
        if pb is None:
            return None
        return all_metrics(s, pb, Body), pb
    except Exception:
        return None


def bootstrap_pvalues(x_sorted, params, Body, q, B, seed, executor):
    """Run B parametric bootstrap replicas and compute p-values and SEs.

    Returns the observed metrics, the bootstrap p-values, the number of
    successful replicas, and a dictionary of bootstrap standard errors
    for the model parameters (body keys and tail keys xi, sigma_gpd, u).
    """
    obs = all_metrics(x_sorted, params, Body)
    n = len(x_sorted)
    tasks = [(n, params, Body.name, q, seed + b) for b in range(B)]
    results = list(executor.map(_replica, tasks))
    boot = {k: [] for k in obs}
    boot_pars = {k: [] for k in Body.params_keys}
    boot_pars['xi'] = []
    boot_pars['sigma_gpd'] = []
    boot_pars['u'] = []
    for r in results:
        if r is None: continue
        m_r, p_r = r
        for k in obs:
            if np.isfinite(m_r[k]):
                boot[k].append(m_r[k])
        for k in Body.params_keys:
            v = p_r['body'].get(k, np.nan)
            if np.isfinite(v):
                boot_pars[k].append(v)
        for k in ('xi', 'sigma_gpd', 'u'):
            v = p_r.get(k, np.nan)
            if np.isfinite(v):
                boot_pars[k].append(v)
    pvals = {}
    for k, o in obs.items():
        arr = np.array(boot[k])
        pvals[k] = (1 + np.sum(arr >= o)) / (1 + len(arr)) if len(arr) else np.nan
    n_ok = min((len(v) for v in boot.values()), default=0)
    se_boot = {}
    for k, arr in boot_pars.items():
        arr = np.asarray(arr, dtype=float)
        se_boot[k] = float(np.std(arr, ddof=1)) if len(arr) >= 2 else np.nan
    return obs, pvals, n_ok, se_boot


# ---------------------------------------------------------------
# Plot style and colors
# ---------------------------------------------------------------
_PLOT_STYLE_SET = False

def _set_plot_style():
    """Apply seaborn whitegrid style with serif fonts."""
    global _PLOT_STYLE_SET
    if _PLOT_STYLE_SET: return
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18,
        'xtick.labelsize': 14, 'ytick.labelsize': 14,
        'legend.fontsize': 13, 'figure.titlesize': 20,
        'lines.linewidth': 1.5, 'lines.markersize': 5,
    })
    _PLOT_STYLE_SET = True


_C_BODY      = '#1f77b4'   # body curve color
_C_TAIL      = '#d62728'   # tail curve color
_C_EMPIRICAL = '#000000'   # empirical scatter color

_PLOT_ORDER = ['DJIA', 'DAX', 'IPC', 'Nikkei']

_COLORS_3 = {'GG': '#1f77b4', 'Gamma': '#2ca02c', 'LN': '#d62728'}
_LABELS_3 = {'GG': 'Generalized Gamma + GPD',
             'Gamma': 'Gamma + GPD',
             'LN': 'Log-Normal + GPD'}


def _add_legend(ax, loc='best'):
    ax.legend(loc=loc, fontsize=11.5, frameon=True, framealpha=0.85,
              edgecolor='0.6', labelspacing=0.4, handlelength=1.5,
              borderpad=0.5, handletextpad=0.5)


def _sane_ylim_log(ax, counts_min=0.5, floor_decades=4):
    """Avoid log axes that extend many decades below the data range."""
    try:
        y0, y1 = ax.get_ylim()
        if y1 > 0 and np.isfinite(y1):
            floor = max(counts_min, y1 / (10 ** floor_decades))
            if y0 < floor:
                ax.set_ylim(floor, y1)
    except Exception:
        pass


def _subsample(n, target=600):
    """Return indices for plotting up to `target` evenly-spaced points."""
    if n <= target: return np.arange(n)
    step = max(1, n // target)
    return np.arange(0, n, step)


# ---------------------------------------------------------------
# CCDF and QQ panels
# ---------------------------------------------------------------
def _ccdf_panel(ax, obs_sorted, params, Body, sign, scale='loglinear'):
    """Draw the empirical CCDF as scatter and the hybrid CCDF as two
    colored curves (body and tail) on a single axis."""
    n = len(obs_sorted)
    y_emp = (n - np.arange(1, n + 1) + 0.5) / (n + 1)

    s_all = _subsample(n)
    ax.plot(obs_sorted[s_all], y_emp[s_all], '.', color=_C_EMPIRICAL,
            ms=2.2, alpha=0.9, zorder=2, label=f'Empirical (n={n})')

    u = params['u']
    x_min, x_max = obs_sorted.min(), obs_sorted.max() * 1.02
    xf = np.linspace(x_min, x_max, 2000)
    ccdf = np.clip(1.0 - hybrid_cdf(xf, params, Body), 1e-300, 1.0)
    seg_body = xf <= u
    seg_tail = xf > u

    def _masked(mask): return np.where(mask, ccdf, np.nan)

    ax.plot(xf, _masked(seg_body), '-', color=_C_BODY, lw=2.0,
            alpha=0.9, zorder=3, label='Generalized Gamma')
    ax.plot(xf, _masked(seg_tail), '-', color=_C_TAIL, lw=2.0,
            alpha=0.9, zorder=3, label='Generalized Pareto')
    ax.axvline(x=u, color='gray', ls='--', lw=0.8, alpha=0.55,
               label=f'u = {u:.4f}')

    if scale == 'loglinear':
        ax.set_yscale('log')
        xl_base = 'VReturns' if sign == 'positive' else r'$|$VReturns$^{-}|$'
        ax.set_xlabel(xl_base)
        ax.set_ylabel('CCDF (log)')
    elif scale == 'loglog':
        ax.set_xscale('log'); ax.set_yscale('log')
        xl_base = ('VReturns (log)' if sign == 'positive'
                   else r'$|$VReturns$^{-}|$ (log)')
        ax.set_xlabel(xl_base)
        ax.set_ylabel('CCDF (log)')

    ax.grid(True, which='both', ls=':', alpha=0.3)
    _sane_ylim_log(ax, counts_min=0.2/n, floor_decades=6)


def _qq_panel(ax, obs_sorted, params, Body, sign):
    """Q-Q plot of model quantiles vs empirical quantiles with body
    and tail in different colors."""
    n = len(obs_sorted)
    p = (np.arange(1, n + 1) - 0.5) / n
    q_mod = hybrid_ppf(p, params, Body)

    u = params['u']
    mB = obs_sorted <= u
    mT = obs_sorted > u

    lo = min(np.nanmin(q_mod), obs_sorted.min())
    hi = max(np.nanmax(q_mod), obs_sorted.max())

    ax.plot([lo, hi], [lo, hi], '-', color=_C_EMPIRICAL, lw=1.1,
            alpha=0.85, zorder=1, label='Perfect fit (y=x)')

    ax.scatter(q_mod[mB], obs_sorted[mB], marker='o', s=10,
               color=_C_BODY, alpha=0.6, edgecolors='none', zorder=3,
               label='Generalized Gamma')
    ax.scatter(q_mod[mT], obs_sorted[mT], marker='o', s=14,
               color=_C_TAIL, alpha=0.8, edgecolors='none', zorder=3,
               label='Generalized Pareto')

    if obs_sorted.max() / max(obs_sorted.min(), 1e-12) > 100:
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('Model quantiles (log)')
        ax.set_ylabel('Empirical quantiles (log)')
    else:
        ax.set_xlabel('Model quantiles')
        ax.set_ylabel('Empirical quantiles')
    ax.grid(True, ls=':', alpha=0.3)


def _make_grid_GG(all_data, kind, outdir, indices_order):
    """Build a 4x2 grid of panels for the GG hybrid model only."""
    _set_plot_style()
    nrows = len(indices_order)
    fig, axes = plt.subplots(nrows, 2, figsize=(14, 4.0 * nrows))
    if nrows == 1: axes = axes.reshape(1, -1)

    for i, name in enumerate(indices_order):
        if name not in all_data:
            for j in (0, 1): axes[i, j].set_axis_off()
            continue
        for j, side in enumerate(['positive', 'negative']):
            ax = axes[i, j]
            entry = all_data[name].get(side)
            if entry is None or 'GG' not in entry['fits']:
                ax.set_axis_off(); continue
            x_all = entry['x_all']
            params = entry['fits']['GG']['params']
            Body = BODIES['GG']

            if kind == 'ccdf_loglinear':
                _ccdf_panel(ax, x_all, params, Body, side, scale='loglinear')
            elif kind == 'ccdf_loglog':
                _ccdf_panel(ax, x_all, params, Body, side, scale='loglog')
            elif kind == 'qq':
                _qq_panel(ax, x_all, params, Body, side)

            sign_dir = 'Positive' if side == 'positive' else 'Negative'
            if kind in ('ccdf_loglinear', 'ccdf_loglog'):
                ax.set_title(f'{name} {sign_dir} VReturns (CCDF)')
            elif kind == 'qq':
                ax.set_title(f'{name} {sign_dir} VReturns (QQ-plot)')

            if i == 0:
                if kind == 'ccdf_loglog':
                    _add_legend(ax, 'lower left')
                elif kind == 'ccdf_loglinear':
                    _add_legend(ax, 'upper right')
                else:
                    _add_legend(ax, 'upper left')

    fig.tight_layout()
    fname_map = {'ccdf_loglinear': 'ccdf_loglinear.png',
                 'ccdf_loglog':    'ccdf_loglog.png',
                 'qq':             'qq_plots.png'}
    out = os.path.join(outdir, fname_map[kind])
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return out


def _make_grid_3models(all_data, outdir, indices_order):
    """Build a 4x2 Q-Q grid overlaying the three body models."""
    _set_plot_style()
    nrows = len(indices_order)
    fig, axes = plt.subplots(nrows, 2, figsize=(14, 4.0 * nrows))
    if nrows == 1: axes = axes.reshape(1, -1)

    for i, name in enumerate(indices_order):
        if name not in all_data:
            for j in (0, 1): axes[i, j].set_axis_off()
            continue
        for j, side in enumerate(['positive', 'negative']):
            ax = axes[i, j]
            entry = all_data[name].get(side)
            if entry is None: ax.set_axis_off(); continue
            x_all = entry['x_all']
            n = len(x_all)
            p = (np.arange(1, n + 1) - 0.5) / n
            lo = float(x_all.min()); hi = float(x_all.max())
            ax.plot([lo, hi], [lo, hi], '-', color='k', lw=1.0, alpha=0.7,
                    label='Perfect fit (y=x)')
            for body_name, info in entry['fits'].items():
                Body = BODIES[body_name]
                q_mod = hybrid_ppf(p, info['params'], Body)
                ax.scatter(q_mod, x_all, s=6, alpha=0.5,
                           color=_COLORS_3[body_name], edgecolors='none',
                           label=_LABELS_3[body_name])
            if (hi / max(lo, 1e-12)) > 100:
                ax.set_xscale('log'); ax.set_yscale('log')
                ax.set_xlabel('Model quantiles (log)')
                ax.set_ylabel('Empirical quantiles (log)')
            else:
                ax.set_xlabel('Model quantiles')
                ax.set_ylabel('Empirical quantiles')
            sign_dir = 'Positive' if side == 'positive' else 'Negative'
            ax.set_title(f'{name} {sign_dir} (n = {n})')
            ax.grid(True, ls=':', alpha=0.4)
            if i == 0 and j == 0:
                ax.legend(loc='upper left', frameon=True, framealpha=0.9,
                          fontsize=9)
    fig.tight_layout()
    out = os.path.join(outdir, 'qq_3models.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return out


# ---------------------------------------------------------------
# Driver
# ---------------------------------------------------------------
def run_all(B=1000, n_workers=30, q_default=0.92, verbose=False):
    """Fit the three hybrid models for each (index, side) case, run the
    parametric bootstrap, save the CSV with metrics, p-values and SEs,
    and generate the consolidated plots."""
    data_dir = os.path.join(SCRIPT_DIR, DATA_DIR_NAME)
    indices = ['IPC', 'Nikkei', 'DAX', 'DJIA']
    sides = ['positive', 'negative']
    # Per-case quantile overrides for the body/tail threshold. Each
    # index uses the same quantile on both the positive and negative
    # sides. IPC uses the default q=0.92.
    q_overrides = {('DJIA',   'positive'): 0.90,
                   ('DJIA',   'negative'): 0.90,
                   ('Nikkei', 'positive'): 0.90,
                   ('Nikkei', 'negative'): 0.90,
                   ('DAX',    'positive'): 0.93,
                   ('DAX',    'negative'): 0.93}

    outdir = os.path.join(SCRIPT_DIR, f'compare_3models_allVR_B{B}')
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, f'compare_3models_allVR_B{B}.csv')

    # Resume support: skip cells already present in the CSV.
    done = set()
    rows = []
    if os.path.exists(csv_path):
        df_done = pd.read_csv(csv_path)
        df_done = df_done[df_done['B'] == B]
        done = set((r['index'], r['side'], r['model'])
                   for _, r in df_done.iterrows())
        rows = df_done.to_dict('records')
        if done:
            print(f"Resuming: {len(done)} cells already done with B={B}.",
                  flush=True)

    all_data = {idx: {} for idx in indices}

    print(f"B={B} | workers={n_workers} | q_default={q_default}", flush=True)
    print(f"q_overrides: {q_overrides}", flush=True)
    print(f"Data: ALL VReturns (Def. 5)  |  Models: GG-GPD, Gamma-GPD, LN-GPD",
          flush=True)
    print(f"Output: {outdir}/", flush=True)
    print("=" * 78, flush=True)
    t_total = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for idx in indices:
            try:
                prices = load_close_prices(idx, data_dir)
            except Exception as e:
                print(f"[WARN] Could not load {idx}: {e}", flush=True)
                continue
            prices_arr = np.asarray(prices.values, dtype=float).ravel()
            pos, neg = extract_abs_vr(prices_arr)
            for side in sides:
                x = pos if side == 'positive' else neg
                x = np.sort(x[x > 0])
                n = len(x)
                q = q_overrides.get((idx, side), q_default)
                fits_for_plots = {}
                for body_name in ['GG', 'Gamma', 'LN']:
                    Body = BODIES[body_name]
                    pb = fit_at_quantile(x, q, Body)
                    if pb is not None:
                        fits_for_plots[body_name] = {'params': pb}
                    if (idx, side, body_name) in done:
                        continue
                    if pb is None:
                        print(f"  {idx:<6s} {side:<8s} {body_name:<6s} | "
                              f"FIT FAILED", flush=True)
                        continue
                    t0 = time.time()
                    obs, pvals, n_ok, se_boot = bootstrap_pvalues(
                        x, pb, Body, q, B=B,
                        seed=hash((idx, side, body_name)) & 0xffffffff,
                        executor=ex,
                    )
                    dt = time.time() - t0
                    body_str = ", ".join(
                        f"{k}={pb['body'][k]:+.4f}" for k in Body.params_keys)
                    se_body_str = ", ".join(
                        f"se_{k}={se_boot.get(k, np.nan):.4f}"
                        for k in Body.params_keys)
                    # Pareto tail index alpha = 1/xi. Defined only for
                    # xi > 0 (heavy/Pareto-type tail). SE via the
                    # delta method: SE(alpha) = SE(xi) / xi^2.
                    xi_hat = pb['xi']
                    se_xi  = se_boot.get('xi', np.nan)
                    if xi_hat > 0 and np.isfinite(xi_hat):
                        alpha_tail = 1.0 / xi_hat
                        se_alpha   = (se_xi / (xi_hat ** 2)
                                      if np.isfinite(se_xi) else np.nan)
                        alpha_str  = f"alpha=1/xi={alpha_tail:.2f} ({se_alpha:.3f})"
                    else:
                        alpha_tail = np.nan
                        se_alpha   = np.nan
                        alpha_str  = "alpha=N/A (xi<=0)"
                    print(f"  {idx:<6s} {side:<8s} {body_name:<6s} | "
                          f"n={n} q={q:.2f} | "
                          f"W2b={obs['CvM_body']:.4f} (p={pvals['CvM_body']:.3f}) | "
                          f"W2t={obs['CvM_tail']:.4f} (p={pvals['CvM_tail']:.3f}) | "
                          f"KSb={obs['KS_body']:.4f} (p={pvals['KS_body']:.3f}) | "
                          f"KSt={obs['KS_tail']:.4f} (p={pvals['KS_tail']:.3f}) | "
                          f"ADb={obs['AD_body']:.3f} (p={pvals['AD_body']:.3f}) | "
                          f"ADt={obs['AD_tail']:.3f} (p={pvals['AD_tail']:.3f}) | "
                          f"{body_str} ({se_body_str}) "
                          f"xi={xi_hat:+.3f} ({se_xi:.4f}) "
                          f"sig={pb['sigma_gpd']:.4f} "
                          f"({se_boot.get('sigma_gpd', np.nan):.4f}) | "
                          f"{alpha_str} | "
                          f"ok={n_ok}/{B} | {dt:.1f}s", flush=True)
                    # Build the CSV row. Body SE columns are named
                    # se_boot_<key> where <key> depends on the model.
                    row = dict(
                        index=idx, side=side, model=body_name,
                        n=n, q=q, u=pb['u'], p_tail=pb['p_tail'],
                        body_params=str(pb['body']),
                        xi=pb['xi'], sigma_gpd=pb['sigma_gpd'],
                        alpha_tail=alpha_tail, se_alpha_tail=se_alpha,
                        **{f'se_boot_{k}': se_boot.get(k, np.nan)
                           for k in Body.params_keys},
                        se_boot_xi=se_boot.get('xi', np.nan),
                        se_boot_sigma_gpd=se_boot.get('sigma_gpd', np.nan),
                        se_boot_u=se_boot.get('u', np.nan),
                        KS_body=obs['KS_body'], KS_tail=obs['KS_tail'],
                        CvM_body=obs['CvM_body'], CvM_tail=obs['CvM_tail'],
                        AD_body=obs['AD_body'],   AD_tail=obs['AD_tail'],
                        p_KS_body=pvals['KS_body'], p_KS_tail=pvals['KS_tail'],
                        p_CvM_body=pvals['CvM_body'], p_CvM_tail=pvals['CvM_tail'],
                        p_AD_body=pvals['AD_body'], p_AD_tail=pvals['AD_tail'],
                        n_ok=n_ok, B=B, seconds=dt,
                    )
                    rows.append(row)
                    pd.DataFrame(rows).to_csv(csv_path, index=False)
                all_data[idx][side] = {'x_all': x, 'fits': fits_for_plots}

    # Generate the consolidated plots.
    print(f"\nGenerating consolidated figures in {outdir} ...", flush=True)
    indices_order = [n for n in _PLOT_ORDER if n in all_data and all_data[n]]
    for n in indices:
        if n not in indices_order and n in all_data and all_data[n]:
            indices_order.append(n)

    for kind in ('ccdf_loglinear', 'ccdf_loglog', 'qq'):
        out = _make_grid_GG(all_data, kind, outdir, indices_order)
        print(f"  -> {out}", flush=True)

    out = _make_grid_3models(all_data, outdir, indices_order)
    print(f"  -> {out}", flush=True)

    print(f"\nTotal: {time.time()-t_total:.1f}s")
    print(f"CSV:   {csv_path}")


if __name__ == "__main__":
    pos_args = [a for a in sys.argv[1:] if not a.startswith('-')]
    B            = int(pos_args[0]) if len(pos_args) > 0 else 1000
    n_workers    = int(pos_args[1]) if len(pos_args) > 1 else 30
    verbose      = '-v' in sys.argv
    run_all(B=B, n_workers=n_workers, verbose=verbose)