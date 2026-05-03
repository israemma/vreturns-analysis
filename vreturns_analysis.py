"""
VReturns Analysis Script

"""

import numpy as np
import pandas as pd
import pickle
import os
import csv
import warnings
import multiprocessing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy import stats
from scipy.special import gammainc, gammaincinv, gammaln
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from statsmodels.tsa.stattools import acf, adfuller
    from statsmodels.stats.diagnostic import acorr_ljungbox
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    from arch.unitroot import PhillipsPerron
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False

try:
    import isotree
    HAS_ISOTREE = True
except ImportError:
    HAS_ISOTREE = False

try:
    from sklearn.ensemble import IsolationForest as SklearnIF
    from sklearn.preprocessing import RobustScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# MATPLOTLIB CONFIGURATION
# ═══════════════════════════════════════════════════════════════
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 22,
    'axes.labelsize': 28, 'axes.titlesize': 30,
    'xtick.labelsize': 22, 'ytick.labelsize': 22,
    'legend.fontsize': 20, 'figure.titlesize': 34,
    'lines.linewidth': 1.5, 'lines.markersize': 5,
})

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════
INDICES = ['DJIA', 'DAX', 'IPC', 'Nikkei']
INDEX_COLORS = {'DJIA': '#1f77b4', 'DAX': '#d62728', 'IPC': '#2ca02c', 'Nikkei': '#ff7f0e'}
INDEX_MARKERS = {'DJIA': 'o', 'DAX': 's', 'IPC': '^', 'Nikkei': 'D'}
INDEX_LINESTYLES = {'DJIA': '-', 'DAX': '--', 'IPC': '-.', 'Nikkei': ':'}
INDEX_START_DATES = {'DJIA': '1992-01-02', 'DAX': '1992-01-02', 'IPC': '1992-01-02', 'Nikkei': '1992-01-06'}
END_DATE = '2023-12-29'
ASCENDING_COLOR = '#2ca02c'
DESCENDING_COLOR = '#d62728'

# Anomaly detection config
ANOM_SEED = 42
ANOM_N_ESTIMATORS = 5000
ANOM_MAX_SAMPLES = 2048
ANOM_N_THREADS = 30
ANOM_INITIAL_MONTHS = 120
ANOM_CONTAMINATION = 0.01
ANOM_CRISIS_PERIODS = [
    ('Dot-com', '2000-03-01', '2002-10-31'),
    ('GFC', '2007-10-01', '2009-03-31'),
    ('COVID', '2020-02-01', '2020-04-30'),
]

# ═══════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════
def fs_floor(figsize):
    w = figsize[0]
    if w >= 14: return 14.0
    if w >= 9:  return 13.0
    if w >= 6:  return 12.0
    return 11.0

def smart_legend(ax, fig, loc="best", ncol=1, title=None, markerscale=None,
                 fs_min=16.0, fs_max=30.0, fs_step=0.5):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_inv = ax.transAxes.inverted()
    artist_boxes = []
    for artist in ax.get_children():
        if isinstance(artist, (plt.matplotlib.legend.Legend, plt.matplotlib.spines.Spine,
                               plt.matplotlib.axis.XAxis, plt.matplotlib.axis.YAxis)):
            continue
        try:
            bb = artist.get_window_extent(renderer=renderer)
            if bb.width == 0 and bb.height == 0: continue
            p0 = ax_inv.transform((bb.x0, bb.y0))
            p1 = ax_inv.transform((bb.x1, bb.y1))
            x0, x1 = sorted([p0[0], p1[0]])
            y0, y1 = sorted([p0[1], p1[1]])
            artist_boxes.append((x0, y0, x1, y1))
        except Exception: continue
    corners = [
        ("upper right", 0.45, 0.45, 1.0, 1.0),
        ("upper left", 0.0, 0.45, 0.55, 1.0),
        ("lower right", 0.45, 0.0, 1.0, 0.55),
        ("lower left", 0.0, 0.0, 0.55, 0.55),
    ]
    if loc != "best":
        filtered = [c for c in corners if c[0] == loc]
        if filtered: corners = filtered
    def _free_area(rx0, ry0, rx1, ry1):
        region_area = max((rx1 - rx0) * (ry1 - ry0), 1e-9)
        occupied = 0.0
        for ax0, ay0, ax1, ay1 in artist_boxes:
            ix0 = max(rx0, ax0); ix1 = min(rx1, ax1)
            iy0 = max(ry0, ay0); iy1 = min(ry1, ay1)
            if ix1 > ix0 and iy1 > iy0:
                occupied += (ix1 - ix0) * (iy1 - iy0)
        return 1.0 - occupied / region_area
    def _overlap(lx0, ly0, lx1, ly1):
        total = 0.0
        for ax0, ay0, ax1, ay1 in artist_boxes:
            ix0 = max(lx0, ax0); ix1 = min(lx1, ax1)
            iy0 = max(ly0, ay0); iy1 = min(ly1, ay1)
            if ix1 > ix0 and iy1 > iy0:
                total += (ix1 - ix0) * (iy1 - iy0)
        return total
    scored = sorted(corners, key=lambda c: _free_area(c[1], c[2], c[3], c[4]), reverse=True)
    best_loc = scored[0][0]
    base_kw = dict(frameon=True, framealpha=0.92, edgecolor="0.6", fancybox=False,
                   borderpad=0.5, labelspacing=0.45, handlelength=1.8,
                   handletextpad=0.5, borderaxespad=0.4, ncol=ncol)
    if title: base_kw["title"] = title
    if markerscale: base_kw["markerscale"] = markerscale
    candidates = np.arange(fs_max, fs_min - fs_step, -fs_step)
    best_leg = None
    for fs in candidates:
        leg = ax.legend(loc=best_loc, fontsize=fs, **base_kw)
        fig.canvas.draw()
        try:
            lb = leg.get_window_extent(renderer=renderer)
            lp0 = ax_inv.transform((lb.x0, lb.y0))
            lp1 = ax_inv.transform((lb.x1, lb.y1))
            lx0, lx1 = sorted([lp0[0], lp1[0]])
            ly0, ly1 = sorted([lp0[1], lp1[1]])
        except Exception:
            leg.remove(); continue
        tol = 0.01
        if not (lx0 < -tol or lx1 > 1+tol or ly0 < -tol or ly1 > 1+tol) and _overlap(lx0, ly0, lx1, ly1) <= 1e-4:
            best_leg = leg; break
        leg.remove()
    if best_leg is None:
        best_leg = ax.legend(loc=best_loc, fontsize=fs_min, **base_kw)
    return best_leg

def add_legend(ax, figsize, loc="best", ncol=1, title=None):
    w = figsize[0]
    if w >= 14: fs = 13.5
    elif w >= 12: fs = 12.0
    elif w >= 9: fs = 11.0
    elif w >= 6: fs = 10.0
    else: fs = 9.0
    kw = dict(fontsize=fs, frameon=True, framealpha=0.92, edgecolor="0.6",
              fancybox=False, borderpad=0.5, labelspacing=0.4, handlelength=1.8,
              handletextpad=0.5, borderaxespad=0.4, ncol=ncol)
    if title: kw["title"] = title
    ax.legend(loc=loc, **kw)

# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════
def load_close_prices(data_folder='financial_data'):
    data = {}
    for name in INDICES:
        fn = os.path.join(data_folder, f'{name}_close.pkl')
        if os.path.exists(fn):
            try:
                with open(fn, 'rb') as f:
                    data[name] = pickle.load(f)
            except Exception: pass
    return data

def download_if_missing(data):
    if len(data) == len(INDICES):
        return data
    if not HAS_YFINANCE:
        print("WARNING: yfinance not available, using only cached data")
        return data
    tickers = {'DJIA': '^DJI', 'DAX': '^GDAXI', 'IPC': '^MXX', 'Nikkei': '^N225'}
    os.makedirs('financial_data', exist_ok=True)
    for name, ticker in tickers.items():
        if name in data: continue
        try:
            df = yf.download(ticker, start='1990-01-01', end='2024-01-01', progress=False)
            if not df.empty:
                series = df['Close'].dropna()
                data[name] = series
                with open(os.path.join('financial_data', f'{name}_close.pkl'), 'wb') as f:
                    pickle.dump(series, f)
        except Exception as e:
            print(f"  Warning {name}: {e}")
    return data

def filter_by_date(data):
    end_dt = pd.to_datetime(END_DATE)
    filtered = {}
    for name, series in data.items():
        start_dt = pd.to_datetime(INDEX_START_DATES.get(name, '1992-01-02'))
        mask = (series.index >= start_dt) & (series.index <= end_dt)
        filtered[name] = series[mask]
    return filtered

def get_data():
    data = load_close_prices()
    data = download_if_missing(data)
    return filter_by_date(data)

# ═══════════════════════════════════════════════════════════════
# TREND IDENTIFICATION
# ═══════════════════════════════════════════════════════════════
def identify_trends_strict(prices):
    """Strict inequalities, skip ties. Used by ACF, anomaly, histograms."""
    if len(prices) < 2: return []
    trends, current_dir, trend_start = [], None, 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]: direction = 'up'
        elif prices[i] < prices[i-1]: direction = 'down'
        else: continue
        if current_dir is None:
            current_dir, trend_start = direction, i-1
        elif current_dir != direction:
            trends.append({'start': trend_start, 'end': i-1, 'direction': current_dir})
            current_dir, trend_start = direction, i-1
    if current_dir is not None:
        trends.append({'start': trend_start, 'end': len(prices)-1, 'direction': current_dir})
    return trends

def identify_trends_nonstrict(prices):
    """Non-strict inequalities (>= <=). Used by observables, pdf_vreturns."""
    trends = []
    i = 0
    while i < len(prices) - 1:
        j = i + 1
        while j < len(prices) and prices[j] >= prices[j-1]: j += 1
        if j - 1 > i:
            trends.append({'start': i, 'end': j-1, 'direction': 'up'})
            i = j - 1; continue
        k = i + 1
        while k < len(prices) and prices[k] <= prices[k-1]: k += 1
        if k - 1 > i:
            trends.append({'start': i, 'end': k-1, 'direction': 'down'})
            i = k - 1; continue
        i += 1
    return trends

# ═══════════════════════════════════════════════════════════════
# VRETURN CALCULATIONS
# ═══════════════════════════════════════════════════════════════
def calc_all_VReturns(prices, trends):
    vrets = []
    for t in trends:
        s, e = t['start'], t['end']
        dur = e - s
        if dur < 1: continue
        ps = prices[s]
        if ps <= 0: continue
        lps = np.log(ps)
        for i in range(1, dur+1):
            idx = s + i
            if idx < len(prices) and prices[idx] > 0:
                vrets.append((np.log(prices[idx]) - lps) / i)
    return np.array(vrets)

def calc_all_TVReturns(prices, trends):
    tvrets = []
    for t in trends:
        s, e = t['start'], t['end']
        dur = e - s
        if dur < 1: continue
        ps, pe = prices[s], prices[e]
        if ps <= 0 or pe <= 0: continue
        tvrets.append((np.log(pe) - np.log(ps)) / dur)
    return np.array(tvrets)

def calc_all_TReturns(prices, trends):
    trets = []
    for t in trends:
        s, e = t['start'], t['end']
        ps, pe = prices[s], prices[e]
        if ps <= 0 or pe <= 0: continue
        trets.append(np.log(pe) - np.log(ps))
    return np.array(trets)

def compute_histogram_linear(data_sorted, n_bins=200, range_=None):
    if len(data_sorted) == 0: return np.array([]), np.array([])
    counts, edges = np.histogram(data_sorted, bins=n_bins, range=range_)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts

def compute_histogram_sinh(data_sorted, n_bins=1000, scale=50000.0):
    if len(data_sorted) == 0: return np.array([]), np.array([])
    x_min, x_max = data_sorted.min(), data_sorted.max()
    z_min = np.arcsinh(x_min * scale)
    z_max = np.arcsinh(x_max * scale)
    z_edges = np.linspace(z_min, z_max, n_bins + 1)
    bin_edges = np.sinh(z_edges) / scale
    counts, _ = np.histogram(data_sorted, bins=bin_edges)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return centers, counts

# ═══════════════════════════════════════════════════════════════
# MODULE 1: ACF STATIONARITY
# ═══════════════════════════════════════════════════════════════
def _acf_calculate_statistics(data, max_lags=50):
    if len(data) < max_lags + 1: max_lags = len(data) - 1
    acf_values, confint = acf(data, nlags=max_lags, alpha=0.05, fft=True)
    try:
        adf_result = adfuller(data, autolag='AIC')
        adf_stat, adf_p, adf_cv = adf_result[0], adf_result[1], adf_result[4]
    except Exception:
        adf_stat, adf_p, adf_cv = np.nan, np.nan, {}
    try:
        lb_result = acorr_ljungbox(data, lags=min(20, len(data)//4), return_df=True)
        lb_stat, lb_p = lb_result['lb_stat'].iloc[-1], lb_result['lb_pvalue'].iloc[-1]
    except Exception:
        lb_stat, lb_p = np.nan, np.nan
    pp_stat, pp_p = np.nan, np.nan
    if HAS_ARCH:
        try:
            pp = PhillipsPerron(data)
            pp_stat, pp_p = pp.stat, pp.pvalue
        except Exception: pass
    lower_ci = confint[:, 0] - acf_values
    upper_ci = confint[:, 1] - acf_values
    return {'acf_values': acf_values, 'lower_ci': lower_ci, 'upper_ci': upper_ci,
            'adf_statistic': adf_stat, 'adf_pvalue': adf_p, 'adf_critical_values': adf_cv,
            'pp_statistic': pp_stat, 'pp_pvalue': pp_p,
            'lb_statistic': lb_stat, 'lb_pvalue': lb_p, 'n_observations': len(data)}

def _acf_plot(vreturns_by_index, output_dir):
    fig, ax = plt.subplots(figsize=(20, 10))
    max_lags = 50
    stats_summary = {}
    for name in INDICES:
        if name not in vreturns_by_index or len(vreturns_by_index[name]) == 0: continue
        acf_stats = _acf_calculate_statistics(vreturns_by_index[name], max_lags)
        stats_summary[name] = acf_stats
        lags = np.arange(len(acf_stats['acf_values']))
        ax.plot(lags, acf_stats['acf_values'], marker='.', linestyle=INDEX_LINESTYLES[name],
                color=INDEX_COLORS[name], markersize=6, linewidth=1.5, label=name)
    ax.set_xlabel('Lag', fontsize=28); ax.set_ylabel('Autocorrelation', fontsize=28)
    ax.grid(True, alpha=0.3); ax.set_xlim(0, max_lags); ax.set_ylim(-0.1, 1.05)
    smart_legend(ax, fig, loc='upper right', fs_min=fs_floor(fig.get_size_inches()))
    plt.tight_layout()
    fp = os.path.join(output_dir, 'ACF_VReturns.png')
    plt.savefig(fp, dpi=300, bbox_inches='tight', facecolor='white'); plt.close()
    return fp, stats_summary

def _acf_export_csv(stats_summary, output_dir):
    rows = []
    for name, s in stats_summary.items():
        rows.append({'Index': name, 'Sample Size': s['n_observations'],
                     'ADF Statistic': s['adf_statistic'], 'ADF p-value': s['adf_pvalue'],
                     'PP Statistic': s.get('pp_statistic', np.nan),
                     'PP p-value': s.get('pp_pvalue', np.nan),
                     'Stationarity (ADF)': 'Stationary' if s['adf_pvalue'] < 0.05 else 'Non-stationary',
                     'Stationarity (PP)': 'Stationary' if s.get('pp_pvalue', 1) < 0.05 else 'Non-stationary'})
    df = pd.DataFrame(rows)
    fp = os.path.join(output_dir, 'UnitRoot_Tests_ADF_PP.csv')
    df.to_csv(fp, index=False)
    return fp

def run_acf_stationarity(data):
    if not HAS_STATSMODELS:
        print("  [ACF] SKIPPED - statsmodels not available"); return
    print("\n" + "="*60)
    print("  MODULE: ACF Stationarity Analysis")
    print("="*60)
    output_dir = "acf_stationarity_vreturns"
    os.makedirs(output_dir, exist_ok=True)
    vrets = {}
    for name, series in data.items():
        if series.empty: continue
        prices = series.values.flatten()
        trends = identify_trends_strict(prices)
        vrets[name] = calc_all_VReturns(prices, trends)
        print(f"  [{name}] VReturns: {len(vrets[name])}")
    plot_path, stats_summary = _acf_plot(vrets, output_dir)
    csv_path = _acf_export_csv(stats_summary, output_dir)
    print(f"  Plot: {plot_path}")
    print(f"  CSV:  {csv_path}")
    for name, s in stats_summary.items():
        status = "STATIONARY" if s['adf_pvalue'] < 0.05 else "NON-STATIONARY"
        print(f"  {name}: {status} (ADF p={s['adf_pvalue']:.6f}, N={s['n_observations']:,})")

# ═══════════════════════════════════════════════════════════════
# MODULE 2: PDF EMPIRICAL (all indices)
# ═══════════════════════════════════════════════════════════════
def _pdf_extract_vreturns(prices):
    p = np.ravel(prices.values)
    trends = []
    i = 0
    while i < len(p) - 1:
        j = i + 1
        while j < len(p) and p[j] >= p[j-1]: j += 1
        if j - 1 > i:
            trends.append({'start': i, 'end': j-1, 'direction': 'up'}); i = j-1; continue
        k = i + 1
        while k < len(p) and p[k] <= p[k-1]: k += 1
        if k - 1 > i:
            trends.append({'start': i, 'end': k-1, 'direction': 'down'}); i = k-1; continue
        i += 1
    vret = []
    for tr in trends:
        s, e = tr['start'], tr['end']
        lps = np.log(p[s])
        for ii in range(1, (e - s) + 1):
            vret.append((np.log(p[s + ii]) - lps) / ii)
    return np.array(vret)

def run_pdf_empirical(data):
    print("\n" + "="*60)
    print("  MODULE: PDF Empirical (All Indices)")
    print("="*60)
    output_dir = "pdf_vreturns_empirical_all_indices"
    os.makedirs(output_dir, exist_ok=True)
    vret_data = {}
    for name, prices in data.items():
        if prices.empty: continue
        vr = _pdf_extract_vreturns(prices)
        vret_nz = vr[vr != 0]
        vret_data[name] = np.sort(vret_nz)
        print(f"  [{name}] VReturns: {len(vret_nz)} (pos={np.sum(vret_nz>0)}, neg={np.sum(vret_nz<0)})")
    plot_order = [k for k in INDICES if k in vret_data]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
    for name in plot_order:
        ds = vret_data[name]
        c, n = compute_histogram_linear(ds, n_bins=500)
        m = n > 0
        ax1.scatter(c[m], n[m], color=INDEX_COLORS[name], marker=INDEX_MARKERS[name],
                    s=20, alpha=0.85, zorder=3, edgecolors='none', label=name)
    ax1.set_yscale('log'); ax1.set_ylim(bottom=0.5)
    ax1.set_xlabel('VReturns', fontsize=28); ax1.set_ylabel('Counts (log)', fontsize=28)
    ax1.grid(True, which='both', ls='--', alpha=0.3)
    smart_legend(ax1, fig, markerscale=2.2, fs_min=20.0, fs_max=30.0)
    for name in plot_order:
        ds = vret_data[name]
        c, n = compute_histogram_sinh(ds, n_bins=200, scale=50000.0)
        m = n > 0
        ax2.scatter(c[m], n[m], color=INDEX_COLORS[name], marker=INDEX_MARKERS[name],
                    s=20, alpha=0.85, zorder=3, edgecolors='none', label=name)
    ax2.set_yscale('log'); ax2.set_ylim(bottom=0.5)
    ax2.set_xlabel('VReturns', fontsize=28); ax2.set_ylabel('Counts (log)', fontsize=28)
    ax2.grid(True, which='both', ls='--', alpha=0.3)
    smart_legend(ax2, fig, markerscale=2.2, fs_min=20.0, fs_max=30.0)
    fig.tight_layout()
    fp = os.path.join(output_dir, 'PDF_Full_VReturns_Combined_Linear_Sinh.png')
    fig.savefig(fp, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f"  Plot saved -> {fp}")

# ═══════════════════════════════════════════════════════════════
# MODULE 3: OBSERVABLES (Kurtosis & Std by Duration)
# ═══════════════════════════════════════════════════════════════
def _obs_compute_returns(prices_series):
    prices_arr = np.nan_to_num(prices_series.values.flatten(), nan=0.0)
    daily = np.log(prices_arr[1:] / prices_arr[:-1])
    daily = daily[~np.isnan(daily)]
    trends = identify_trends_nonstrict(prices_arr)
    t_ret, tv_ret, v_ret = [], [], []
    t_dur = {d: [] for d in range(1, 11)}
    tv_dur = {d: [] for d in range(1, 11)}
    for t in trends:
        s, e = t['start'], t['end']
        dur = e - s
        ps, pe = prices_arr[s], prices_arr[e]
        if ps <= 0 or pe <= 0: continue
        tr = np.log(pe) - np.log(ps)
        tvr = tr / dur
        t_ret.append(tr); tv_ret.append(tvr)
        if 1 <= dur <= 10:
            t_dur[dur].append(tr); tv_dur[dur].append(tvr)
        for i in range(1, dur+1):
            pi = prices_arr[s+i]
            if pi > 0:
                v_ret.append((np.log(pi) - np.log(ps)) / i)
    return daily, np.array(t_ret), np.array(tv_ret), np.array(v_ret), t_dur, tv_dur

def _obs_calc_stats(data):
    if len(data) == 0: return 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    n = len(data); m = np.mean(data); s = np.std(data, ddof=1)
    se = s/np.sqrt(n) if n > 0 else np.nan
    sk = stats.skew(data, bias=False); se_sk = np.sqrt(6.0/n) if n > 0 else np.nan
    ku = stats.kurtosis(data, bias=False, fisher=True); se_ku = np.sqrt(24.0/n) if n > 0 else np.nan
    return n, m, s, se, sk, se_sk, ku, se_ku

def run_observables(data):
    print("\n" + "="*60)
    print("  MODULE: Observables (Kurtosis & Std by Duration)")
    print("="*60)
    output_dir = "observables_kurtosis_std_by_duration"
    os.makedirs(output_dir, exist_ok=True)
    table_rows = []
    plot_t_kurt = {i: [] for i in INDICES}; plot_tv_kurt = {i: [] for i in INDICES}
    plot_t_std = {i: [] for i in INDICES}; plot_tv_std = {i: [] for i in INDICES}
    for name in INDICES:
        if name not in data:
            print(f"  Missing data for {name}"); continue
        prices = data[name].dropna()
        daily, t_ret, tv_ret, v_ret, t_dur, tv_dur = _obs_compute_returns(prices)
        for mname, mdata in [("Daily Returns", daily), ("TReturns", t_ret),
                              ("VReturns", v_ret), ("TVReturns", tv_ret)]:
            n, m, s, se, sk, se_sk, ku, se_ku = _obs_calc_stats(mdata)
            table_rows.append({'Index': name, 'Metric': mname, 'n': n,
                'Mean \u00b1 SE': f"{m:.6f} \u00b1 {se:.6f}" if not np.isnan(m) else "NaN",
                'Std': s,
                'Skewness \u00b1 SE': f"{sk:.6f} \u00b1 {se_sk:.6f}" if not np.isnan(sk) else "NaN",
                'Kurtosis \u00b1 SE': f"{ku:.6f} \u00b1 {se_ku:.6f}" if not np.isnan(ku) else "NaN"})
        for d in range(1, 11):
            if len(t_dur[d]) > 3:
                _, _, ts, _, _, _, tk, _ = _obs_calc_stats(t_dur[d])
                plot_t_kurt[name].append((d, tk)); plot_t_std[name].append((d, ts))
            else:
                plot_t_kurt[name].append((d, np.nan)); plot_t_std[name].append((d, np.nan))
            if len(tv_dur[d]) > 3:
                _, _, tvs, _, _, _, tvk, _ = _obs_calc_stats(tv_dur[d])
                plot_tv_kurt[name].append((d, tvk)); plot_tv_std[name].append((d, tvs))
            else:
                plot_tv_kurt[name].append((d, np.nan)); plot_tv_std[name].append((d, np.nan))
    df_t = pd.DataFrame(table_rows)
    tp = os.path.join(output_dir, 'table_returns_observables_metrics.csv')
    df_t.to_csv(tp, index=False)
    print(f"  Table saved: {tp}")
    markers = INDEX_MARKERS; colors = INDEX_COLORS; ls = INDEX_LINESTYLES
    # Kurtosis plot
    fig3, axes3 = plt.subplots(1, 2, figsize=(20, 8))
    for idx in INDICES:
        if not plot_t_kurt[idx]: continue
        xt, yt = zip(*plot_t_kurt[idx]); xtv, ytv = zip(*plot_tv_kurt[idx])
        axes3[0].plot(xt, yt, marker=markers[idx], linestyle=ls[idx], color=colors[idx], label=idx, alpha=0.8)
        axes3[1].plot(xtv, ytv, marker=markers[idx], linestyle=ls[idx], color=colors[idx], label=idx, alpha=0.8)
    axes3[0].set_title('TReturns Kurtosis by Duration', fontsize=30)
    axes3[0].set_xlabel('Trend Duration (Days)', fontsize=28); axes3[0].set_ylabel('Excess Kurtosis', fontsize=28)
    axes3[0].axhline(0, color='gray', linestyle='--', alpha=0.5)
    smart_legend(axes3[0], fig3, fs_min=fs_floor(fig3.get_size_inches())); axes3[0].grid(True, linestyle='--', alpha=0.5)
    axes3[1].set_title('TVReturns Kurtosis by Duration', fontsize=30)
    axes3[1].set_xlabel('Trend Duration (Days)', fontsize=28); axes3[1].set_ylabel('Excess Kurtosis', fontsize=28)
    axes3[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
    smart_legend(axes3[1], fig3, fs_min=fs_floor(fig3.get_size_inches())); axes3[1].grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout(pad=3.0)
    fp3 = os.path.join(output_dir, 'Excess_Kurtosis_by_Duration.png')
    fig3.savefig(fp3, dpi=300, bbox_inches='tight'); plt.close(fig3)
    print(f"  Kurtosis plot: {fp3}")
    # Std plot
    fig4, axes4 = plt.subplots(1, 2, figsize=(20, 8))
    for idx in INDICES:
        if not plot_t_std[idx]: continue
        xt, yt = zip(*plot_t_std[idx]); xtv, ytv = zip(*plot_tv_std[idx])
        axes4[0].plot(xt, yt, marker=markers[idx], linestyle=ls[idx], color=colors[idx], label=idx, alpha=0.8)
        axes4[1].plot(xtv, ytv, marker=markers[idx], linestyle=ls[idx], color=colors[idx], label=idx, alpha=0.8)
    axes4[0].set_title('TReturns Standard Deviation by Duration', fontsize=30)
    axes4[0].set_xlabel('Trend Duration (Days)', fontsize=28); axes4[0].set_ylabel('Standard Deviation', fontsize=28)
    smart_legend(axes4[0], fig4, loc="upper left", fs_min=fs_floor(fig4.get_size_inches())); axes4[0].grid(True, linestyle='--', alpha=0.5)
    axes4[1].set_title('TVReturns Standard Deviation by Duration', fontsize=30)
    axes4[1].set_xlabel('Trend Duration (Days)', fontsize=28); axes4[1].set_ylabel('Standard Deviation', fontsize=28)
    smart_legend(axes4[1], fig4, loc="lower left", fs_min=fs_floor(fig4.get_size_inches())); axes4[1].grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout(pad=3.0)
    fp4 = os.path.join(output_dir, 'Standard_Deviation_by_Duration.png')
    fig4.savefig(fp4, dpi=300, bbox_inches='tight'); plt.close(fig4)
    print(f"  Std plot: {fp4}")

# ═══════════════════════════════════════════════════════════════
# MODULE 4: COLLAPSE BY MU
# ═══════════════════════════════════════════════════════════════
def _collapse_extract_trends(prices):
    p = prices.values
    if len(p) < 2: return p, []
    trends, cur, start = [], None, 0
    for i in range(1, len(p)):
        if p[i] > p[i-1]: d = 'up'
        elif p[i] < p[i-1]: d = 'down'
        else: continue
        if cur is None: cur, start = d, i-1
        elif cur != d:
            trends.append({'start': start, 'end': i-1, 'direction': cur})
            cur, start = d, i-1
    if cur is not None:
        trends.append({'start': start, 'end': len(p)-1, 'direction': cur})
    return p, trends

def _collapse_compute_quantities(p, trends):
    S_l, V_l, r_l, i_l, L_l = [], [], [], [], []
    for tr in trends:
        s, e = tr['start'], tr['end']
        L = e - s
        if L < 1: continue
        lps = np.log(p[s])
        for ii in range(1, L+1):
            Si = np.log(p[s+ii]) - lps
            S_l.append(Si); V_l.append(Si/ii)
            r_l.append(np.log(p[s+ii]) - np.log(p[s+ii-1]))
            i_l.append(ii); L_l.append(L)
    return (np.array(S_l), np.array(V_l), np.array(r_l),
            np.array(i_l, dtype=int), np.array(L_l, dtype=int))

def _paper_mode(vals, n_bins=80):
    vals = np.asarray(vals)
    vals = vals[vals != 0]
    if len(vals) < 5: return np.nan
    av = np.abs(vals)
    counts, edges = np.histogram(av, bins=n_bins)
    if counts.max() == 0: return np.nan
    j = int(np.argmax(counts))
    return float(0.5 * (edges[j] + edges[j+1]))

def run_collapse_by_mu(data):
    print("\n" + "="*60)
    print("  MODULE: VReturns Collapse by Mu")
    print("="*60)
    output_dir = 'vreturns_collapse_by_mu'
    os.makedirs(output_dir, exist_ok=True)
    quantities, mu_table = {}, {}
    for name in INDICES:
        if name not in data or data[name].empty: continue
        p, trends = _collapse_extract_trends(data[name])
        S, V, r, istep, L = _collapse_compute_quantities(p, trends)
        quantities[name] = {'S': S, 'V': V, 'r_inside': r, 'i': istep, 'L': L}
        print(f"  [{name}] {len(trends)} trends, {len(S)} pairs")
        r_nz = r[r != 0]
        mu_table[name] = (float(np.mean(r_nz[r_nz > 0])), float(np.mean(np.abs(r_nz[r_nz < 0]))),
                          float(np.mean(np.abs(r_nz))))
    # Build plot
    n_bins_mode = 80
    mode_table = {}
    for name in INDICES:
        if name not in quantities or name not in mu_table: continue
        V = quantities[name]['V']; V = V[V != 0]; mu = mu_table[name][2]
        mode_table[name] = {
            'mode_pos_raw': _paper_mode(V[V > 0], n_bins_mode),
            'mode_neg_raw_abs': _paper_mode(V[V < 0], n_bins_mode),
            'mode_pos_over_mu': _paper_mode(V[V > 0] / mu, n_bins_mode),
            'mode_neg_over_mu_abs': _paper_mode(V[V < 0] / mu, n_bins_mode),
        }
    fig, ax = plt.subplots(1, 1, figsize=(18, 10))
    for name in INDICES:
        if name not in quantities or name not in mu_table: continue
        V = quantities[name]['V']; V = V[V != 0]; mu = mu_table[name][2]
        c, n = compute_histogram_linear(V / mu, n_bins=300, range_=(-8, 8))
        m = n > 0
        ax.scatter(c[m], n[m], color=INDEX_COLORS[name], marker=INDEX_MARKERS[name],
                   s=18, alpha=0.8, edgecolors='none', label=name)
    ax.set_yscale('log'); ax.set_ylim(bottom=0.5)
    ax.set_xlabel(r'$V_i / \mu$', fontsize=28); ax.set_ylabel('Counts (log)', fontsize=28)
    ax.grid(True, which='both', ls='--', alpha=0.3)
    ax.legend(loc='upper right', fontsize=26, frameon=True, framealpha=0.92, edgecolor='0.6',
              fancybox=False, borderpad=0.5, labelspacing=0.4, handlelength=1.6,
              handletextpad=0.5, borderaxespad=0.4, markerscale=2.5)
    fig.tight_layout()
    fp = os.path.join(output_dir, 'EXP7_collapse_by_mu.png')
    fig.savefig(fp, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"  Plot: {fp}")
    # CSV
    csv_path = os.path.join(output_dir, 'EXP7_peaks_and_mu.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['index', 'mu_pos', 'mu_neg', 'mu_all', 'mode_pos_raw', 'mode_neg_raw',
                     'mode_pos_over_mu', 'mode_neg_over_mu'])
        for name in INDICES:
            if name not in mu_table or name not in mode_table: continue
            mp, mn, ma = mu_table[name]; md = mode_table[name]
            w.writerow([name, f'{mp:.6f}', f'{mn:.6f}', f'{ma:.6f}',
                        f'{md["mode_pos_raw"]:.6f}', f'{-md["mode_neg_raw_abs"]:.6f}',
                        f'{md["mode_pos_over_mu"]:.4f}', f'{-md["mode_neg_over_mu_abs"]:.4f}'])
        if mode_table:
            avg_p = np.nanmean([m['mode_pos_over_mu'] for m in mode_table.values()])
            avg_n = np.nanmean([m['mode_neg_over_mu_abs'] for m in mode_table.values()])
            w.writerow([]); w.writerow(['AVERAGE', '', '', '', '', '', f'{avg_p:.4f}', f'{-avg_n:.4f}'])
    print(f"  CSV:  {csv_path}")

# ═══════════════════════════════════════════════════════════════
# MODULE 5: ANOMALY DETECTION (Isolation Forest)
# ═══════════════════════════════════════════════════════════════
def _anom_calc_daily_vreturns(prices_series):
    prices = prices_series.values.astype(float)
    dates = prices_series.index
    trends = identify_trends_strict(prices)
    val_l, date_l, type_l = [], [], []
    for t in trends:
        s, e = t['start'], t['end']
        dur = e - s
        if dur < 1 or s >= len(prices) or prices[s] <= 0: continue
        lp0 = np.log(prices[s])
        tt = 'up' if prices[e] >= prices[s] else 'down'
        for i in range(1, dur+1):
            idx = s + i
            if idx < len(prices) and prices[idx] > 0:
                val_l.append(np.abs((np.log(prices[idx]) - lp0) / i))
                date_l.append(dates[idx]); type_l.append(tt)
    return pd.DataFrame({'date': np.array(date_l).flatten(), 'value': np.array(val_l).flatten(),
                         'trend_type': np.array(type_l).flatten()}).sort_values('date').reset_index(drop=True)

def _anom_calc_abs_log_returns(prices_series):
    prices = prices_series.values.astype(float).flatten()
    dates = prices_series.index
    lr = np.log(prices[1:] / prices[:-1])
    types = ['up' if r >= 0 else 'down' for r in lr]
    return pd.DataFrame({'date': dates[1:], 'value': np.abs(lr).flatten(), 'trend_type': types})

def _anom_extract_features(values):
    if len(values) < 5: return np.zeros(8, dtype=np.float32)
    return np.array([float(np.mean(values)), float(np.std(values, ddof=1)) + 1e-12,
                     float(stats.skew(values)), float(stats.kurtosis(values)),
                     float(np.percentile(values, 5)), float(np.percentile(values, 95)),
                     float(np.percentile(values, 75) - np.percentile(values, 25)),
                     float(np.max(np.abs(values)))], dtype=np.float32)

def _anom_run_iforest(deltas):
    if len(deltas) < 10:
        return np.full(len(deltas), np.nan), np.zeros(len(deltas), dtype=int)
    X = RobustScaler().fit_transform(deltas)
    if HAS_ISOTREE:
        clf = isotree.IsolationForest(n_estimators=ANOM_N_ESTIMATORS, sample_size=ANOM_MAX_SAMPLES,
                                       ndim=2, random_seed=ANOM_SEED, nthreads=ANOM_N_THREADS)
        clf.fit(X); scores = clf.predict(X)
        thr = np.percentile(scores, (1.0 - ANOM_CONTAMINATION) * 100)
        return scores, (scores >= thr).astype(int)
    else:
        clf = SklearnIF(n_estimators=ANOM_N_ESTIMATORS, max_samples=ANOM_MAX_SAMPLES,
                        contamination=ANOM_CONTAMINATION, random_state=ANOM_SEED, n_jobs=ANOM_N_THREADS)
        clf.fit(X)
        return -clf.score_samples(X), (clf.predict(X) == -1).astype(int)

def _anom_analyse(index_name, df_data):
    if df_data is None or len(df_data) < 100: return None
    t_1yr = df_data['date'].min() + pd.DateOffset(months=ANOM_INITIAL_MONTHS)
    start_idx = (df_data['date'] <= t_1yr).sum()
    if start_idx < 10: return None
    vals = df_data['value'].values.astype(np.float64)
    dates, types = df_data['date'].values, df_data['trend_type'].values
    prev = _anom_extract_features(vals[:start_idx])
    deltas_l, ed, et = [], [], []
    for i in range(start_idx, len(vals)):
        curr = _anom_extract_features(vals[:i+1])
        deltas_l.append(np.abs(curr - prev)); ed.append(dates[i]); et.append(types[i])
        prev = curr
    scores, labels = _anom_run_iforest(np.array(deltas_l, dtype=np.float32))
    return pd.DataFrame({'index': index_name, 'date': ed, 'trend_type': et,
                         'if_score': scores, 'is_anomaly': labels})

def _anom_plot_deltas(all_dfs, path, filename, label_type="VReturns", add_spikes=False):
    fig, axes = plt.subplots(len(INDICES), 1, figsize=(12, 3.5*len(INDICES)), sharex=True)
    for i, name in enumerate(INDICES):
        ax = axes[i]; df = all_dfs.get(name)
        if df is None or df.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center'); continue
        df = df.set_index('date')
        s, labs, tt = df['if_score'], df['is_anomaly'], df['trend_type']
        um, dm = tt == 'up', tt == 'down'
        if add_spikes:
            ax.vlines(s.index[um], 0, s[um], color=ASCENDING_COLOR, alpha=0.3, lw=0.5, zorder=1)
            ax.vlines(s.index[dm], 0, s[dm], color=DESCENDING_COLOR, alpha=0.3, lw=0.5, zorder=1)
        ax.scatter(s.index[um], s[um], color=ASCENDING_COLOR, s=4, alpha=0.5, zorder=2,
                   label='Uptrends' if i == 0 else "")
        ax.scatter(s.index[dm], s[dm], color=DESCENDING_COLOR, s=4, alpha=0.5, zorder=2,
                   label='Downtrends' if i == 0 else "")
        am = labs == 1
        if am.any():
            au, ad = am & um, am & dm
            if au.any(): ax.scatter(s.index[au], s[au], color=ASCENDING_COLOR, edgecolors='black', lw=0.8, s=20, zorder=5)
            if ad.any(): ax.scatter(s.index[ad], s[ad], color=DESCENDING_COLOR, edgecolors='black', lw=0.8, s=20, zorder=5)
        thr = np.percentile(s.dropna(), (1.0 - ANOM_CONTAMINATION)*100)
        ax.axhline(thr, color='darkred', ls='--', lw=1, label='Threshold' if i == 0 else "")
        ymax = s.max()*1.2 if len(s) > 0 else 1
        for cn, cs, ce in ANOM_CRISIS_PERIODS:
            ax.axvspan(pd.Timestamp(cs), pd.Timestamp(ce), alpha=0.12, color='orange', zorder=0)
            ax.text(pd.Timestamp(cs), ymax*0.85, cn, fontsize=8, color='darkorange', rotation=90)
        ax.set_ylim(0, ymax); ax.set_title(f'{name}'); ax.set_ylabel('Score'); ax.grid(True, alpha=0.3)
        if i == 0: ax.legend(loc="best", ncol=3, markerscale=3.0)
    plt.tight_layout()
    plt.savefig(os.path.join(path, filename), dpi=200, bbox_inches='tight'); plt.close()

def _anom_plot_vreturns(all_vr, path):
    fig, axes = plt.subplots(len(INDICES), 1, figsize=(14, 3.5*len(INDICES)), sharex=True)
    for i, name in enumerate(INDICES):
        ax = axes[i]; df = all_vr.get(name)
        if df is None or df.empty: continue
        df = df.set_index('date')
        um, dm = df['trend_type'] == 'up', df['trend_type'] == 'down'
        ax.vlines(df.index[um], 0, df['value'][um], color=ASCENDING_COLOR, alpha=0.4, lw=0.8, zorder=1)
        ax.vlines(df.index[dm], 0, df['value'][dm], color=DESCENDING_COLOR, alpha=0.4, lw=0.8, zorder=1)
        ax.scatter(df.index[um], df['value'][um], color=ASCENDING_COLOR, s=2, alpha=0.4, zorder=2,
                   label='Uptrend' if i == 0 else "")
        ax.scatter(df.index[dm], df['value'][dm], color=DESCENDING_COLOR, s=2, alpha=0.4, zorder=2,
                   label='Downtrend' if i == 0 else "")
        ax.set_ylim(0, df['value'].max()*1.15 if len(df) > 0 else 1)
        ax.set_title(name); ax.set_ylabel('|VReturn|'); ax.grid(True, alpha=0.3)
        if i == 0: smart_legend(ax, fig, loc="best", ncol=2, markerscale=4.0, fs_min=20.0, fs_max=30.0)
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'vreturns_cronologicos.png'), dpi=200, bbox_inches='tight'); plt.close()

def _anom_plot_acf(all_vr, path, lags=150):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12)); axes = axes.flatten()
    for i, name in enumerate(INDICES):
        ax = axes[i]; df = all_vr.get(name)
        if df is None or df.empty: continue
        values = df['value'].values
        acf_v = acf(values, nlags=lags, fft=True)
        lr = range(len(acf_v))
        ax.bar(lr, acf_v, color=INDEX_COLORS[name], width=0.7, alpha=0.7, label='ACF')
        ax.plot(list(lr)[1:], acf_v[1:], color=INDEX_COLORS[name], alpha=0.9, marker='o', ms=3, ls='-', lw=0.5)
        ci = 1.96 / np.sqrt(len(values))
        ax.axhline(ci, color='darkred', ls='--', alpha=0.6, lw=1, label='95% Conf.')
        ax.fill_between(lr, -ci, ci, color='red', alpha=0.1, zorder=0)
        ax.set_title(f'ACF of |VReturns|: {name} (Lags: {lags})', fontsize=18, pad=15)
        ax.set_xlabel('Lag', fontsize=14); ax.set_ylabel('Autocorrelation', fontsize=14)
        ax.set_ylim(-0.1, 1.05); ax.set_xlim(-2, lags+2); ax.grid(True, alpha=0.2, ls=':')
        add_legend(ax, figsize=(15, 12), loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'acf_vreturns.png'), dpi=200, bbox_inches='tight'); plt.close()

def _anom_plot_log_returns(data, path):
    fig, axes = plt.subplots(len(INDICES), 1, figsize=(14, 3.5*len(INDICES)), sharex=True)
    fig.suptitle('Daily Logarithmic Returns by Index')
    for i, name in enumerate(INDICES):
        ax = axes[i]; series = data.get(name)
        if series is None or series.empty: continue
        sv = series.values.flatten().astype(float)
        lr = pd.Series(np.log(sv[1:]/sv[:-1]), index=series.index[1:])
        ax.plot(lr.index, lr.values, color=INDEX_COLORS[name], lw=0.4, alpha=0.8, label='Log-Return')
        m = float(lr.abs().max()) if not lr.empty else 0.1
        ax.set_ylim(-m*1.1, m*1.1); ax.set_title(name); ax.set_ylabel('Log-Return'); ax.grid(True, alpha=0.3)
        add_legend(ax, figsize=(14, 3.5*len(INDICES)), loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'log_returns_diarios.png'), dpi=200); plt.close()

def run_anomaly_detection(data):
    if not HAS_SKLEARN:
        print("  [ANOMALY] SKIPPED - sklearn not available"); return
    if not HAS_STATSMODELS:
        print("  [ANOMALY] SKIPPED - statsmodels not available (needed for ACF)"); return
    print("\n" + "="*60)
    print("  MODULE: Anomaly Detection (Isolation Forest)")
    print("="*60)
    save_path = 'anomaly_detection_iforest_vreturns'
    os.makedirs(save_path, exist_ok=True)
    np.random.seed(ANOM_SEED)
    alg = "Extended IF (isotree)" if HAS_ISOTREE else "IF (sklearn)"
    print(f"  Detector: {alg}  |  Base: {ANOM_INITIAL_MONTHS}mo  |  Contam: {ANOM_CONTAMINATION}")
    all_vr = {n: _anom_calc_daily_vreturns(data[n]) for n in INDICES if n in data}
    all_ret = {n: _anom_calc_abs_log_returns(data[n]) for n in INDICES if n in data}
    results_vr, results_ret = {}, {}
    with ProcessPoolExecutor(max_workers=ANOM_N_THREADS) as ex:
        vf = {ex.submit(_anom_analyse, n, all_vr[n]): n for n in all_vr}
        rf = {ex.submit(_anom_analyse, n, all_ret[n]): n for n in all_ret}
        for f in as_completed(vf):
            n = vf[f]; results_vr[n] = f.result(); print(f"  VReturns done: {n}")
        for f in as_completed(rf):
            n = rf[f]; results_ret[n] = f.result(); print(f"  Returns done: {n}")
    valid_vr = [df for df in results_vr.values() if df is not None and not df.empty]
    if valid_vr:
        _anom_plot_deltas(results_vr, save_path, 'daily_delta_iforest_VRETURNS.png', "VReturns")
        _anom_plot_vreturns(all_vr, save_path)
        _anom_plot_acf(all_vr, save_path)
        pd.concat(valid_vr).to_csv(os.path.join(save_path, 'daily_deltas_vreturns.csv'), index=False)
    valid_ret = [df for df in results_ret.values() if df is not None and not df.empty]
    if valid_ret:
        _anom_plot_deltas(results_ret, save_path, 'daily_delta_iforest_RETURNS.png', "Returns", add_spikes=True)
        _anom_plot_log_returns(data, save_path)
        pd.concat(valid_ret).to_csv(os.path.join(save_path, 'daily_deltas_returns.csv'), index=False)
    print(f"  Results in: {save_path}/")

# ═══════════════════════════════════════════════════════════════
# MODULE 6: HISTOGRAMS BY DURATION
# ═══════════════════════════════════════════════════════════════
def _hist_calc_VRet_by_dur(prices, trends, tgt=[1,2,3,4,5,6,7,8,9]):
    vr = {d: [] for d in tgt}
    for t in trends:
        s, e = t['start'], t['end']; dur = e - s
        if dur < 1 or prices[s] <= 0: continue
        lps = np.log(prices[s])
        for i in range(1, dur+1):
            idx = s + i
            if idx < len(prices) and prices[idx] > 0 and dur in tgt:
                vr[dur].append((np.log(prices[idx]) - lps) / i)
    return {d: np.array(v) for d, v in vr.items()}

def _hist_calc_TVRet_by_dur(prices, trends, tgt=[1,2,3,4,5,6,7,8,9]):
    tv = {d: [] for d in tgt}
    for t in trends:
        s, e = t['start'], t['end']; dur = e - s
        if dur < 1 or dur not in tgt: continue
        ps, pe = prices[s], prices[e]
        if ps <= 0 or pe <= 0: continue
        tv[dur].append((np.log(pe) - np.log(ps)) / dur)
    return {d: np.array(v) for d, v in tv.items()}

def _hist_calc_TRet_by_dur(prices, trends, tgt=[1,2,3,4,5,6,7,8,9]):
    tr = {d: [] for d in tgt}
    for t in trends:
        s, e = t['start'], t['end']; dur = e - s
        if dur < 1 or dur not in tgt: continue
        ps, pe = prices[s], prices[e]
        if ps <= 0 or pe <= 0: continue
        tr[dur].append(np.log(pe) - np.log(ps))
    return {d: np.array(v) for d, v in tr.items()}

def _hist_plot_4x3(vr_di, tvr_di, tr_di, all_vr, all_tvr, all_tr, save_path):
    os.makedirs(save_path, exist_ok=True)
    dur_colors = {1:'#1f77b4', 2:'#ff7f0e', 3:'#2ca02c', 4:'#d62728',
                  5:'#9467bd', 6:'#8c564b', 7:'#e377c2', 8:'#7f7f7f', 9:'#bcbd22'}
    all_color = '#000000'
    tgt_d = [1, 3, 6, 9]
    manual_locs = {(0,0):"upper left",(0,1):"upper left",(1,1):"upper right",
                   (2,1):"upper right",(3,1):"upper left",(1,2):"upper left",(3,2):"upper left"}
    fig, axes = plt.subplots(4, 3, figsize=(24, 20))
    for i, name in enumerate(INDICES):
        for col, (by_dur, all_data, xlabel, title_sfx) in enumerate([
            (vr_di, all_vr, 'VReturns', 'VReturns'),
            (tvr_di, all_tvr, 'TVReturns', 'TVReturns'),
            (tr_di, all_tr, 'TReturns', 'TReturns')]):
            ax = axes[i, col]
            if name not in by_dur: ax.text(0.5,0.5,'No Data',ha='center',va='center',transform=ax.transAxes); continue
            bd = by_dur[name]
            all_vals = []
            for d in tgt_d:
                if len(bd[d]) > 0: all_vals.extend(bd[d])
            if name in all_data and len(all_data[name]) > 0:
                all_vals.extend(all_data[name])
            if not all_vals: ax.text(0.5,0.5,'No Data',ha='center',va='center',transform=ax.transAxes); continue
            bins = np.linspace(np.min(all_vals), np.max(all_vals), 100)
            if name in all_data and len(all_data[name]) > 0:
                ax.hist(all_data[name], bins=bins, histtype='step', color=all_color, lw=2.5, alpha=0.9,
                        label=f'All data (n={len(all_data[name])})')
            for d in tgt_d:
                if len(bd[d]) > 0:
                    ax.hist(bd[d], bins=bins, histtype='step', color=dur_colors[d], lw=2.0, alpha=0.8,
                            label=f'{d} day (n={len(bd[d])})')
            ax.set_xlabel(xlabel, fontsize=28); ax.set_ylabel('Frequency (log)', fontsize=28)
            ax.set_yscale('log'); ax.set_title(f'{name} - {title_sfx}', fontsize=30)
            loc = manual_locs.get((i, col), "best")
            smart_legend(ax, fig, loc=loc, fs_min=fs_floor(fig.get_size_inches()))
            ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fp = os.path.join(save_path, 'combined_histograms_by_duration_4x3_log_y.png')
    plt.savefig(fp, dpi=300, bbox_inches='tight'); plt.close()
    return fp

def run_histograms_by_duration(data):
    print("\n" + "="*60)
    print("  MODULE: Histograms by Duration")
    print("="*60)
    save_path = 'vreturns_histograms_by_duration'
    tgt = [1,2,3,4,5,6,7,8,9]
    vr_di, tvr_di, tr_di = {}, {}, {}
    all_vr, all_tvr, all_tr = {}, {}, {}
    for name, series in data.items():
        if series.empty: continue
        prices = series.values.flatten()
        trends = identify_trends_strict(prices)
        vr_di[name] = _hist_calc_VRet_by_dur(prices, trends, tgt)
        tvr_di[name] = _hist_calc_TVRet_by_dur(prices, trends, tgt)
        tr_di[name] = _hist_calc_TRet_by_dur(prices, trends, tgt)
        all_vr[name] = calc_all_VReturns(prices, trends)
        all_tvr[name] = calc_all_TVReturns(prices, trends)
        all_tr[name] = calc_all_TReturns(prices, trends)
        print(f"  [{name}] VR={len(all_vr[name])}, TVR={len(all_tvr[name])}, TR={len(all_tr[name])}")
    fp = _hist_plot_4x3(vr_di, tvr_di, tr_di, all_vr, all_tvr, all_tr, save_path)
    print(f"  Plot: {fp}")

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  UNIFIED VRETURNS ANALYSIS")
    print("  Running 6 analysis modules")
    print("=" * 60)

    data = get_data()
    missing = [k for k in INDICES if k not in data or data[k].empty]
    if missing:
        print(f"WARNING: Missing data for: {missing}")
    for name, series in data.items():
        if not series.empty:
            print(f"  [{name}] {series.index[0].date()} -> {series.index[-1].date()} ({len(series)} obs)")

    # 1. ACF Stationarity
    run_acf_stationarity(data)

    # 2. PDF Empirical
    run_pdf_empirical(data)

    # 3. Observables (Kurtosis & Std)
    run_observables(data)

    # 4. Collapse by Mu
    run_collapse_by_mu(data)

    # 5. Anomaly Detection
    run_anomaly_detection(data)

    # 6. Histograms by Duration
    run_histograms_by_duration(data)

    print("\n" + "=" * 60)
    print("  ALL MODULES COMPLETED")
    print("=" * 60)

if __name__ == '__main__':
    main()
