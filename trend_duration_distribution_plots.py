import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import pandas as pd
import pickle
import warnings
import yfinance as yf
from scipy.stats import geom
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

def fs_floor(figsize):
    w = figsize[0]
    if w >= 14: return 14.0
    if w >= 9:  return 13.0
    if w >= 6:  return 12.0
    return 11.0

def smart_legend(ax, fig, loc="best", ncol=1, title=None,
                 fs_min=14.0, fs_max=26.0, fs_step=0.5):
    import matplotlib.pyplot as plt
    import numpy as np
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_inv   = ax.transAxes.inverted()
    artist_boxes = []
    for artist in ax.get_children():
        if isinstance(artist, (plt.matplotlib.legend.Legend,
                               plt.matplotlib.spines.Spine,
                               plt.matplotlib.axis.XAxis,
                               plt.matplotlib.axis.YAxis)):
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
    pad = 0.03
    corners = [
        ("upper right", 0.45, 0.45, 1.0,  1.0 ),
        ("upper left",  0.0,  0.45, 0.55, 1.0 ),
        ("lower right", 0.45, 0.0,  1.0,  0.55),
        ("lower left",  0.0,  0.0,  0.55, 0.55),
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
    base_kw = dict(
        frameon=True, framealpha=0.92, edgecolor="0.6",
        fancybox=False, borderpad=0.5, labelspacing=0.45,
        handlelength=1.8, handletextpad=0.5, borderaxespad=0.4, ncol=ncol,
    )
    if title: base_kw["title"] = title
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
            leg.remove()
            continue
        tol = 0.01
        if not (lx0 < -tol or lx1 > 1 + tol or ly0 < -tol or ly1 > 1 + tol) and _overlap(lx0, ly0, lx1, ly1) <= 1e-4:
            best_leg = leg
            break
        leg.remove()
    if best_leg is None:
        best_leg = ax.legend(loc=best_loc, fontsize=fs_min, **base_kw)
    return best_leg



warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', message='invalid value encountered in log')
warnings.filterwarnings('ignore', message='invalid value encountered in sqrt')
warnings.filterwarnings('ignore', message='divide by zero encountered in divide')
warnings.filterwarnings('ignore', message='Optimal parameters not found')

# Plot style configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 22,
    'axes.labelsize': 28,
    'axes.titlesize': 30,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'figure.titlesize': 34,
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
})

ASCENDING_COLOR = '#2ca02c'  # Green
DESCENDING_COLOR = '#d62728' # Red

def download_financial_data():
    indices = {'DJIA': '^DJI', 'DAX': '^GDAXI', 'IPC': '^MXX', 'Nikkei': '^N225'}
    date_ranges = {
        'DJIA': ('1991-08-11', '2023-12-31'), 'DAX': ('1991-08-11', '2023-12-30'),
        'IPC': ('1991-08-11', '2023-12-31'), 'Nikkei': ('1991-10-11', '2023-12-30')
    }
    
    # Load cached data
    data, all_prices_raw = {}, {}
    for index_name in indices.keys():
        close_fn = os.path.join('financial_data', f"{index_name}_close.pkl")
        full_fn = os.path.join('financial_data', f"{index_name}_full.pkl")
        if os.path.exists(close_fn) and os.path.exists(full_fn):
            try:
                with open(close_fn, 'rb') as f:
                    data[index_name] = pickle.load(f)
                with open(full_fn, 'rb') as f:
                    all_prices_raw[index_name] = pickle.load(f)
            except Exception:
                pass
    
    missing_indices = [idx for idx in indices.keys() if idx not in data or idx not in all_prices_raw]
    if not missing_indices:
        pass
    else:
        for index in missing_indices:
            ticker, (start_date, end_date) = indices[index], date_ranges[index]
            try:
                df_yf = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not df_yf.empty:
                    data[index] = df_yf['Close'].dropna()
                    all_prices_raw[index] = df_yf
            except Exception:
                pass
        
        # Save downloaded data
        if not os.path.exists('financial_data'):
            os.makedirs('financial_data')
        for index_name, price_series in data.items():
            try:
                with open(os.path.join('financial_data', f"{index_name}_close.pkl"), 'wb') as f:
                    pickle.dump(price_series, f)
            except Exception:
                pass
        for index_name, price_df in all_prices_raw.items():
            try:
                with open(os.path.join('financial_data', f"{index_name}_full.pkl"), 'wb') as f:
                    pickle.dump(price_df, f)
            except Exception:
                pass
    
    # Synchronize dates
    if data:
        start_dates = [price_series.index.min() for price_series in data.values() if not price_series.empty]
        end_dates = [price_series.index.max() for price_series in data.values() if not price_series.empty]
        if start_dates and end_dates:
            common_start_date = max(start_dates)
            common_end_date = min(end_dates)
            synchronized_data = {}
            synchronized_all_prices_raw = {}
            for index_name, price_series in data.items():
                mask = (price_series.index >= common_start_date) & (price_series.index <= common_end_date)
                synchronized_data[index_name] = price_series[mask]
                if index_name in all_prices_raw:
                    price_df = all_prices_raw[index_name]
                    mask_df = (price_df.index >= common_start_date) & (price_df.index <= common_end_date)
                    synchronized_all_prices_raw[index_name] = price_df[mask_df]
            data = synchronized_data
            all_prices_raw = synchronized_all_prices_raw
    
    return data, all_prices_raw

# --- Trend Analysis Functions ---

def identify_trends(prices):
    trends = []
    i = 0
    while i < len(prices) - 1:
        j = i + 1
        while j < len(prices) and prices[j] >= prices[j-1]:
            j += 1
        if j - 1 > i:
            trends.append({'start': i, 'end': j - 1, 'direction': 1})
            i = j - 1
            continue
        k = i + 1
        while k < len(prices) and prices[k] <= prices[k-1]:
            k += 1
        if k - 1 > i:
            trends.append({'start': i, 'end': k - 1, 'direction': -1})
            i = k - 1
            continue
        i += 1
    return trends

def enrich_trend_data(prices, trends_list):
    enriched_trends = []
    for trend in trends_list:
        start_idx, end_idx = trend['start'], trend['end']
        duration = end_idx - start_idx
        if duration == 0:
            continue
        p_start, p_end = prices[start_idx], prices[end_idx]
        if p_start > 1e-9 and p_end > 1e-9:
            log_r = np.log(p_end) - np.log(p_start)
            abs_r = abs(log_r)
            velocity = log_r / duration
            abs_velocity = abs(velocity)
            trend_info = {
                'start': start_idx, 'end': end_idx,
                'duration': float(duration), 'direction': trend['direction'],
                'log_return': float(log_r), 'abs_return': float(abs_r),
                'velocity': float(velocity), 'abs_velocity': float(abs_velocity)
            }
            enriched_trends.append(trend_info)
    return enriched_trends


def precompute_trend_data(all_prices_data):
    all_trends_by_index = {}
    all_trend_sequences_by_index = {}
    for index_name, prices_series in all_prices_data.items():
        prices_series_cleaned = prices_series.dropna()
        if prices_series_cleaned.empty:
            continue
        prices_values_cleaned = prices_series_cleaned.values
        prices_cleaned = prices_values_cleaned[~np.isnan(prices_values_cleaned)]
        if len(prices_cleaned) < 2:
            continue
        trends = identify_trends(prices_cleaned)
        if not trends:
            continue
        enriched_trends = enrich_trend_data(prices_cleaned, trends)
        if not enriched_trends:
            continue
        df_trends_current_index = pd.DataFrame(enriched_trends)
        df_trends_current_index['index_name'] = index_name
        all_trends_by_index[index_name] = df_trends_current_index
    return all_trends_by_index, all_trend_sequences_by_index

# --- Auxiliary Functions ---

def calculate_rmse(observed_freq, expected_freq):
    if len(observed_freq) == 0 or len(expected_freq) == 0:
        return np.nan
    valid_indices = ~(np.isnan(observed_freq) | np.isinf(observed_freq) | np.isnan(expected_freq) | np.isinf(expected_freq))
    observed_freq = observed_freq[valid_indices]
    expected_freq = expected_freq[valid_indices]
    if len(observed_freq) == 0:
        return np.nan
    return np.sqrt(np.mean((observed_freq - expected_freq)**2))

# --- Plotting Functions ---

def _add_statistical_summary_to_plot(ax, data, geom_fit_params=None, geom_fit_rmse=None, geom_chi2=None, geom_chi2_p=None):
    if data is None or len(data) < 2:
        return
    
    handles, labels = ax.get_legend_handles_labels()
    smart_legend(ax, plt.gcf(), loc='upper right', fs_min=fs_floor(plt.gcf().get_size_inches()))


def exponential_model(x, a, b):
    """Exponential model: y = a * exp(-b * x)"""
    return a * np.exp(-b * x)

def fit_exponential_to_duration(data):
    """
    Fits y = a * exp(-b * x) to histogram counts of discrete duration data.

    Strategy:
    - Fit is done in log-space (log-linear regression) to be consistent
      with log-scale visualization and to weight tail equally.
    - Poisson weights (1/sqrt(count)) are used since counts follow Poisson.
    - Goodness-of-fit uses chi-squared (standard for count/frequency data).
    - R² is also returned but over all bins (including zeros) for reference.
    """
    try:
        # --- Build histogram with one bin per integer value ---
        bins = np.arange(data.min(), data.max() + 2) - 0.5
        counts, bin_edges = np.histogram(data, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # integer values

        # --- Fit in log-space using only bins with count > 0 ---
        # log(y) = log(a) - b*x  →  linear regression on (x, log(y))
        non_zero_mask = counts > 0
        x_fit = bin_centers[non_zero_mask]
        y_fit = counts[non_zero_mask]

        if len(x_fit) < 3:
            return None, None, np.nan, np.nan, np.nan, np.nan, None

        log_y = np.log(y_fit)

        # Poisson weights: variance of count ~ count, so weight = 1/sqrt(count)
        # In log-space: sigma_log(y) ≈ 1/sqrt(count)  (by error propagation)
        weights = 1.0 / np.sqrt(y_fit)   # Poisson uncertainty propagated to log-space

        # Weighted linear regression: log(y) = log(a) - b*x
        # Using numpy's polyfit with weights
        coeffs, cov_matrix = np.polyfit(x_fit, log_y, deg=1, w=1.0/weights, cov=True)
        b = -coeffs[0]          # slope is -b
        log_a = coeffs[1]
        a = np.exp(log_a)

        # Parameter errors from covariance matrix
        b_error = np.sqrt(cov_matrix[0, 0])
        log_a_error = np.sqrt(cov_matrix[1, 1])
        a_error = a * log_a_error   # error propagation: sigma_a = a * sigma_log(a)

        # --- Chi-squared goodness-of-fit (over ALL bins, including zeros) ---
        # This is the correct metric for count data
        y_expected_all = exponential_model(bin_centers, a, b)

        # Only include bins where expected >= 1 (standard requirement for chi-squared)
        valid_mask = y_expected_all >= 1
        if np.sum(valid_mask) < 2:
            chi2_stat, chi2_p_value = np.nan, np.nan
        else:
            obs = counts[valid_mask].astype(float)
            exp = y_expected_all[valid_mask]
            chi2_stat = np.sum((obs - exp) ** 2 / exp)
            dof = np.sum(valid_mask) - 2   # n_bins - n_parameters
            chi2_p_value = 1 - stats.chi2.cdf(chi2_stat, dof) if dof > 0 else np.nan

        # --- RMSE over non-zero bins (supplementary metric) ---
        y_expected_nz = exponential_model(x_fit, a, b)
        rmse = np.sqrt(np.mean((y_fit - y_expected_nz) ** 2))

        # --- R² over ALL bins (for reference only, not primary metric) ---
        y_expected_all_r2 = exponential_model(bin_centers, a, b)
        r2 = r2_score(counts, y_expected_all_r2)

        return (
            (a, b),
            (a_error, b_error),
            chi2_stat,
            chi2_p_value,
            rmse,
            r2,
            {'x': bin_centers, 'y': counts, 'y_fitted': y_expected_all}
        )

    except Exception as e:
        print(f"    Error in exponential fit: {e}")
        return None, None, np.nan, np.nan, np.nan, np.nan, None

def fit_geometric_to_duration(data):
    """
    Fits a geometric distribution to discrete duration data and computes goodness-of-fit metrics.
    - DOF is corrected (n-2) because p is estimated from data.
    - Tail grouping is used for chi-squared instead of renormalization.
    - Full range is used instead of truncation.
    """
    if len(data) < 2 or not np.all(data > 0):
        return None, None, None, None, None, None
    try:
        from scipy.stats import chisquare
        data = np.round(data).astype(int)
        sample_mean = np.mean(data)
        if sample_mean <= 0:
            return None, None, None, None, None, None

        # MLE / method of moments: p = 1/mean for geom starting at 1
        p_est = min(max(1.0 / sample_mean, 1e-10), 1.0)
        params = (0, p_est)

        # Observed and expected frequencies over ENTIRE real range
        max_val = int(np.max(data))
        values_range = range(1, max_val + 1)

        observed_freq = np.array([np.sum(data == val) for val in values_range], dtype=float)
        expected_freq = np.array([geom.pmf(val, p_est) * len(data) for val in values_range])

        # Group tail: bins with expected < 1 are accumulated in the last valid bin
        valid_mask = expected_freq >= 1
        if np.sum(valid_mask) >= 2:
            obs_valid = observed_freq[valid_mask]
            exp_valid = expected_freq[valid_mask]
            
            obs_tail = len(data) - np.sum(obs_valid)
            exp_tail = len(data) - np.sum(exp_valid)
            
            if exp_tail >= 1:
                obs_grouped = np.append(obs_valid, obs_tail)
                exp_grouped = np.append(exp_valid, exp_tail)
            else:
                obs_grouped = obs_valid.copy()
                exp_grouped = exp_valid.copy()
                obs_grouped[-1] += obs_tail
                exp_grouped[-1] += exp_tail

            # ddof=1 because p was estimated from the data
            chi2_stat, chi2_p_value = chisquare(obs_grouped, exp_grouped, ddof=1)
        else:
            chi2_stat, chi2_p_value = np.nan, np.nan

        # RMSE over the whole range (untruncated)
        rmse_val = calculate_rmse(observed_freq, expected_freq)

        # Bootstrap for p error
        n_bootstrap = 100
        bootstrap_ps = []
        for _ in range(n_bootstrap):
            try:
                bs = np.random.choice(data, size=len(data), replace=True)
                bs_p = min(max(1.0 / np.mean(bs), 1e-10), 1.0)
                bootstrap_ps.append(bs_p)
            except:
                continue

        p_error = np.std(bootstrap_ps) if bootstrap_ps else np.nan
        param_errors = [0, p_error]  # loc no tiene error, solo p

        return params, param_errors, chi2_stat, chi2_p_value, rmse_val, (observed_freq, expected_freq, values_range)

    except Exception as e:
        print(f"  Error in fit_geometric_to_duration: {e}")
        return None, None, None, None, None, None

def plot_trend_duration_distribution(all_trends_by_index, save_path='plots_trend_acceleration', index_name="", plot_number=1):
    """
    Generates a single plot with two subplots showing the duration distribution of
    ascending and descending trends with geometric fits.
    """
    output_folder_idx = os.path.join(save_path, index_name)
    if not os.path.exists(output_folder_idx):
        os.makedirs(output_folder_idx)
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    ax_up = axes[0]
    up_trends = all_trends_by_index.get(index_name, pd.DataFrame()).query('direction == 1')
    up_durations = up_trends['duration'].values
    ax_down = axes[1]
    down_trends = all_trends_by_index.get(index_name, pd.DataFrame()).query('direction == -1')
    down_durations = down_trends['duration'].values
    if len(up_durations) > 0:
        # Create histogram for ascending trends (discrete bins)
        bins_up = np.arange(up_durations.min(), up_durations.max() + 2) - 0.5
        counts_up, bins_up, _ = ax_up.hist(up_durations, bins=bins_up, density=False, alpha=0.7, color=ASCENDING_COLOR, edgecolor='black', label='Uptrends')
        # Fit geometric distribution to ascending durations
        geom_params_up, geom_errors_up, chi2_up, chi2_p_up, rmse_up, hist_data_up = fit_geometric_to_duration(up_durations)
        if geom_params_up is not None:
            loc, p = geom_params_up
            p_error_up = geom_errors_up[1] if geom_errors_up is not None and len(geom_errors_up) > 1 else np.nan
            label_up = f'Geometric Fit (p={p:.3f}±{p_error_up:.3f})' if pd.notna(p_error_up) else f'Geometric Fit (p={p:.3f})'
            discrete_x = np.arange(1, int(up_durations.max()) + 1)
            pmf_values = geom.pmf(discrete_x, p) * len(up_durations)  # Scale to frequency
            ax_up.bar(discrete_x, pmf_values, alpha=0.6, color='red', width=0.4, label=label_up)
            
        _add_statistical_summary_to_plot(ax_up, up_durations, geom_fit_params=geom_params_up, geom_fit_rmse=rmse_up, geom_chi2=chi2_up, geom_chi2_p=chi2_p_up)
    else:
        ax_up.text(0.5, 0.5, 'No ascending trends found', ha='center', va='center', transform=ax_up.transAxes)
    ax_up.set_title('(a) Uptrends', fontsize=30)
    ax_up.set_xlabel('Trend Duration (Days)', fontsize=28)
    ax_up.set_ylabel('Frequency (log)', fontsize=28)
    ax_up.set_yscale('log')
    smart_legend(ax_up, fig, loc='upper right', fs_min=fs_floor(fig.get_size_inches()))
    ax_up.tick_params(axis='both', which='major', labelsize=22)
    ax_up.grid(True, which="both", ls="--", alpha=0.3)
    if len(down_durations) > 0:
        bins_down = np.arange(down_durations.min(), down_durations.max() + 2) - 0.5
        counts_down, bins_down, _ = ax_down.hist(down_durations, bins=bins_down, density=False, alpha=0.7, color=DESCENDING_COLOR, edgecolor='black', label='Downtrends')
        geom_params_down, geom_errors_down, chi2_down, chi2_p_down, rmse_down, hist_data_down = fit_geometric_to_duration(down_durations)
        if geom_params_down is not None:
            loc, p = geom_params_down
            p_error_down = geom_errors_down[1] if geom_errors_down is not None and len(geom_errors_down) > 1 else np.nan
            label_down = f'Geometric Fit (p={p:.3f}±{p_error_down:.3f})' if pd.notna(p_error_down) else f'Geometric Fit (p={p:.3f})'
            discrete_x = np.arange(1, int(down_durations.max()) + 1)
            pmf_values = geom.pmf(discrete_x, p) * len(down_durations)
            ax_down.bar(discrete_x, pmf_values, alpha=0.6, color='red', width=0.4, label=label_down)
            
        _add_statistical_summary_to_plot(ax_down, down_durations, geom_fit_params=geom_params_down, geom_fit_rmse=rmse_down, geom_chi2=chi2_down, geom_chi2_p=chi2_p_down)
    else:
        ax_down.text(0.5, 0.5, 'No descending trends found', ha='center', va='center', transform=ax_down.transAxes)
    ax_down.set_title('(b) Downtrends', fontsize=30)
    ax_down.set_xlabel('Trend Duration (Days)', fontsize=28)
    ax_down.set_ylabel('Frequency (log)', fontsize=28)
    ax_down.set_yscale('log')
    smart_legend(ax_down, fig, loc='upper right', fs_min=fs_floor(fig.get_size_inches()))
    ax_down.tick_params(axis='both', which='major', labelsize=22)
    ax_down.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filename = f'plot_{plot_number:02d}_trend_duration_distribution_{index_name}.png'
    plt.savefig(f'{output_folder_idx}/{plot_filename}', dpi=300, bbox_inches='tight')
    plt.close()

def plot_trend_duration_distribution_exponential_fit(all_trends_by_index, save_path='plots_trend_acceleration', index_name="", plot_number=2):
    """
    Generates a single plot with two subplots showing the duration distribution of
    ascending and descending trends as scatter points with exponential fits.
    """
    output_folder_idx = os.path.join(save_path, index_name)
    if not os.path.exists(output_folder_idx):
        os.makedirs(output_folder_idx)
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    ax_up = axes[0]
    up_trends = all_trends_by_index.get(index_name, pd.DataFrame()).query('direction == 1')
    up_durations = up_trends['duration'].values
    
    ax_down = axes[1]
    down_trends = all_trends_by_index.get(index_name, pd.DataFrame()).query('direction == -1')
    down_durations = down_trends['duration'].values
    
    if len(up_durations) > 0:
        # Create histogram data for scatter points
        bins_up = np.arange(up_durations.min(), up_durations.max() + 2) - 0.5
        counts_up, bin_edges_up = np.histogram(up_durations, bins=bins_up)
        bin_centers_up = (bin_edges_up[:-1] + bin_edges_up[1:]) / 2
        
        # Plot as scatter points
        ax_up.scatter(bin_centers_up, counts_up, color=ASCENDING_COLOR, alpha=0.7, s=50, label='Uptrends', edgecolors='black')
        
        # Fit exponential distribution to ascending durations
        exp_params_up, exp_errors_up, chi2_up, chi2_p_up, rmse_up, r2_up, fit_data_up = fit_exponential_to_duration(up_durations)
        
        if exp_params_up is not None:
            a, b = exp_params_up
            a_error_up, b_error_up = exp_errors_up
            
            # Plot exponential fit
            x_fit = np.linspace(1, int(up_durations.max()), 200)
            y_fit = exponential_model(x_fit, a, b)
            
            label_up = (f'Exp. Fit: a={a:.1f}±{a_error_up:.1f}, b={b:.3f}±{b_error_up:.3f}')
            ax_up.plot(x_fit, y_fit, 'r-', linewidth=2, label=label_up)

            # Metrics are saved in the csv, removed from the plot
            
        else:
            print(f"  Warning: Exponential fit failed for ascending trends in {index_name}")
    else:
        ax_up.text(0.5, 0.5, 'No ascending trends found', ha='center', va='center', transform=ax_up.transAxes)
    
    ax_up.set_title('(a) Uptrends', fontsize=30)
    ax_up.set_xlabel('Trend Duration (Days)', fontsize=28)
    ax_up.set_ylabel('Frequency (log)', fontsize=28)
    ax_up.set_yscale('log')
    smart_legend(ax_up, fig, loc='upper left', fs_min=fs_floor(fig.get_size_inches()))
    ax_up.tick_params(axis='both', which='major', labelsize=22)
    ax_up.grid(True, which="both", ls="--", alpha=0.3)
    
    if len(down_durations) > 0:
        # Create histogram data for scatter points
        bins_down = np.arange(down_durations.min(), down_durations.max() + 2) - 0.5
        counts_down, bin_edges_down = np.histogram(down_durations, bins=bins_down)
        bin_centers_down = (bin_edges_down[:-1] + bin_edges_down[1:]) / 2
        
        # Plot as scatter points
        ax_down.scatter(bin_centers_down, counts_down, color=DESCENDING_COLOR, alpha=0.7, s=50, label='Downtrends', edgecolors='black')
        
        # Fit exponential distribution to descending durations
        exp_params_down, exp_errors_down, chi2_down, chi2_p_down, rmse_down, r2_down, fit_data_down = fit_exponential_to_duration(down_durations)
        
        if exp_params_down is not None:
            a, b = exp_params_down
            a_error_down, b_error_down = exp_errors_down
            
            # Plot exponential fit
            x_fit = np.linspace(1, int(down_durations.max()), 200)
            y_fit = exponential_model(x_fit, a, b)
            
            label_down = (f'Exp. Fit: a={a:.1f}±{a_error_down:.1f}, b={b:.3f}±{b_error_down:.3f}')
            ax_down.plot(x_fit, y_fit, 'r-', linewidth=2, label=label_down)

            # Metrics are saved in the csv, removed from the plot
            
    else:
        ax_down.text(0.5, 0.5, 'No descending trends found', ha='center', va='center', transform=ax_down.transAxes)
    
    ax_down.set_title('(b) Downtrends', fontsize=30)
    ax_down.set_xlabel('Trend Duration (Days)', fontsize=28)
    ax_down.set_ylabel('Frequency (log)', fontsize=28)
    ax_down.set_yscale('log')
    smart_legend(ax_down, fig, loc='upper left', fs_min=fs_floor(fig.get_size_inches()))
    ax_down.tick_params(axis='both', which='major', labelsize=22)
    ax_down.grid(True, which="both", ls="--", alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filename = f'plot_{plot_number:02d}_trend_duration_distribution_exponential_fit_{index_name}.png'
    plt.savefig(f'{output_folder_idx}/{plot_filename}', dpi=300, bbox_inches='tight')
    plt.close()





def export_geometric_fit_to_csv(all_trends_by_index, save_path='plots_trend_acceleration'):
    """
    Export geometric fit parameters and statistical metrics to a CSV file.

    Columns: Index, Direction, n, Mean, Mean_error, Median, Std, Min, Max,
             Skewness, Kurtosis, Geom_p, Geom_p_error, Geom_loc,
             Chi2_stat, Chi2_pval, RMSE, RMSE_error
    """
    rows = []
    n_bootstrap = 100

    for direction_label, direction_val in [('Up', 1), ('Down', -1)]:
        for index_name in all_trends_by_index.keys():
            df = all_trends_by_index[index_name]
            durations = df[df['direction'] == direction_val]['duration'].values

            row = {'Index': index_name, 'Direction': direction_label}

            if len(durations) < 2:
                row.update({
                    'n': len(durations),
                    'Mean': np.nan, 'Mean_error': np.nan,
                    'Median': np.nan, 'Std': np.nan,
                    'Min': np.nan, 'Max': np.nan,
                    'Skewness': np.nan, 'Kurtosis': np.nan,
                    'Geom_p': np.nan, 'Geom_p_error': np.nan, 'Geom_loc': np.nan,
                    'Chi2_stat': np.nan, 'Chi2_pval': np.nan,
                    'RMSE': np.nan, 'RMSE_error': np.nan,
                })
                rows.append(row)
                continue

            # --- Descriptive statistics ---
            sample_mean   = float(np.mean(durations))
            sample_median = float(np.median(durations))
            sample_std    = float(np.std(durations, ddof=1))
            sample_min    = int(np.min(durations))
            sample_max    = int(np.max(durations))
            sample_skew   = float(stats.skew(durations))
            sample_kurt   = float(stats.kurtosis(durations, fisher=True))

            # Bootstrap error for mean
            bs_means = [np.mean(np.random.choice(durations, size=len(durations), replace=True))
                        for _ in range(n_bootstrap)]
            mean_error = float(np.std(bs_means))

            # --- Geometric fit ---
            geom_params, geom_errors, chi2_stat, chi2_p, rmse_val, _ = fit_geometric_to_duration(durations)

            if geom_params is not None:
                loc_val, p_val = geom_params
                p_error = float(geom_errors[1]) if geom_errors is not None and len(geom_errors) > 1 else np.nan

                # Bootstrap RMSE error
                bs_rmses = []
                for _ in range(n_bootstrap):
                    try:
                        bs = np.round(np.random.choice(durations, size=len(durations), replace=True)).astype(int)
                        bs_mean = np.mean(bs)
                        if bs_mean > 0:
                            bs_p = min(max(1.0 / bs_mean, 1e-10), 1.0)
                            max_v = int(np.max(bs))
                            obs_f = np.array([np.sum(bs == v) for v in range(1, max_v + 1)], dtype=float)
                            exp_f = np.array([geom.pmf(v, bs_p) * len(bs) for v in range(1, max_v + 1)])
                            r = calculate_rmse(obs_f, exp_f)
                            if not np.isnan(r):
                                bs_rmses.append(r)
                    except:
                        continue
                rmse_error = float(np.std(bs_rmses)) if bs_rmses else np.nan
            else:
                loc_val, p_val, p_error = np.nan, np.nan, np.nan
                chi2_stat, chi2_p, rmse_val, rmse_error = np.nan, np.nan, np.nan, np.nan

            row.update({
                'n':            len(durations),
                'Mean':         round(sample_mean, 6),
                'Mean_error':   round(mean_error, 6),
                'Median':       sample_median,
                'Std':          round(sample_std, 6),
                'Min':          sample_min,
                'Max':          sample_max,
                'Skewness':     round(sample_skew, 6),
                'Kurtosis':     round(sample_kurt, 6),
                'Geom_p':       round(float(p_val), 8) if pd.notna(p_val) else np.nan,
                'Geom_p_error': round(float(p_error), 8) if pd.notna(p_error) else np.nan,
                'Geom_loc':     int(loc_val) if pd.notna(loc_val) else np.nan,
                'Chi2_stat':    round(float(chi2_stat), 6) if pd.notna(chi2_stat) else np.nan,
                'Chi2_pval':    round(float(chi2_p), 6) if pd.notna(chi2_p) else np.nan,
                'RMSE':         round(float(rmse_val), 6) if pd.notna(rmse_val) else np.nan,
                'RMSE_error':   round(float(rmse_error), 6) if pd.notna(rmse_error) else np.nan,
            })
            rows.append(row)

    columns = [
        'Index', 'Direction', 'n',
        'Mean', 'Mean_error', 'Median', 'Std', 'Min', 'Max', 'Skewness', 'Kurtosis',
        'Geom_p', 'Geom_p_error', 'Geom_loc',
        'Chi2_stat', 'Chi2_pval', 'RMSE', 'RMSE_error',
    ]
    df_out = pd.DataFrame(rows, columns=columns)
    os.makedirs(save_path, exist_ok=True)
    csv_path = os.path.join(save_path, 'geometric_fit_results.csv')
    df_out.to_csv(csv_path, index=False, float_format='%.8g')
    print(f"  Geometric fit CSV exported -> {csv_path}")
    return csv_path


def export_exponential_fit_to_csv(all_trends_by_index, save_path='plots_trend_acceleration'):
    """
    Export exponential fit parameters and statistical metrics to a CSV file.

    Columns: Index, Direction, n, Mean, Mean_error, Median, Std, Min, Max,
             Skewness, Kurtosis, Exp_a, Exp_a_error, Exp_b, Exp_b_error,
             Chi2_stat, Chi2_pval, RMSE, R2
    """
    rows = []
    n_bootstrap = 100

    for direction_label, direction_val in [('Up', 1), ('Down', -1)]:
        for index_name in all_trends_by_index.keys():
            df = all_trends_by_index[index_name]
            durations = df[df['direction'] == direction_val]['duration'].values

            row = {'Index': index_name, 'Direction': direction_label}

            if len(durations) < 2:
                row.update({
                    'n': len(durations),
                    'Mean': np.nan, 'Mean_error': np.nan,
                    'Median': np.nan, 'Std': np.nan,
                    'Min': np.nan, 'Max': np.nan,
                    'Skewness': np.nan, 'Kurtosis': np.nan,
                    'Exp_a': np.nan, 'Exp_a_error': np.nan,
                    'Exp_b': np.nan, 'Exp_b_error': np.nan,
                    'Chi2_stat': np.nan, 'Chi2_pval': np.nan,
                    'RMSE': np.nan, 'R2': np.nan,
                })
                rows.append(row)
                continue

            # --- Descriptive statistics ---
            sample_mean   = float(np.mean(durations))
            sample_median = float(np.median(durations))
            sample_std    = float(np.std(durations, ddof=1))
            sample_min    = int(np.min(durations))
            sample_max    = int(np.max(durations))
            sample_skew   = float(stats.skew(durations))
            sample_kurt   = float(stats.kurtosis(durations, fisher=True))

            # Bootstrap error for mean
            bs_means = [np.mean(np.random.choice(durations, size=len(durations), replace=True))
                        for _ in range(n_bootstrap)]
            mean_error = float(np.std(bs_means))

            # --- Exponential fit ---
            exp_params, exp_errors, chi2_stat, chi2_p, rmse_val, r2_val, _ = fit_exponential_to_duration(durations)

            if exp_params is not None:
                a_val, b_val = exp_params
                a_error, b_error = exp_errors
            else:
                a_val = b_val = a_error = b_error = np.nan
                chi2_stat = chi2_p = rmse_val = r2_val = np.nan

            row.update({
                'n':           len(durations),
                'Mean':        round(sample_mean, 6),
                'Mean_error':  round(mean_error, 6),
                'Median':      sample_median,
                'Std':         round(sample_std, 6),
                'Min':         sample_min,
                'Max':         sample_max,
                'Skewness':    round(sample_skew, 6),
                'Kurtosis':    round(sample_kurt, 6),
                'Exp_a':       round(float(a_val), 6) if pd.notna(a_val) else np.nan,
                'Exp_a_error': round(float(a_error), 6) if pd.notna(a_error) else np.nan,
                'Exp_b':       round(float(b_val), 8) if pd.notna(b_val) else np.nan,
                'Exp_b_error': round(float(b_error), 8) if pd.notna(b_error) else np.nan,
                'Chi2_stat':   round(float(chi2_stat), 6) if pd.notna(chi2_stat) else np.nan,
                'Chi2_pval':   round(float(chi2_p), 6) if pd.notna(chi2_p) else np.nan,
                'RMSE':        round(float(rmse_val), 6) if pd.notna(rmse_val) else np.nan,
                'R2':          round(float(r2_val), 8) if pd.notna(r2_val) else np.nan,
            })
            rows.append(row)

    columns = [
        'Index', 'Direction', 'n',
        'Mean', 'Mean_error', 'Median', 'Std', 'Min', 'Max', 'Skewness', 'Kurtosis',
        'Exp_a', 'Exp_a_error', 'Exp_b', 'Exp_b_error',
        'Chi2_stat', 'Chi2_pval', 'RMSE', 'R2',
    ]
    df_out = pd.DataFrame(rows, columns=columns)
    os.makedirs(save_path, exist_ok=True)
    csv_path = os.path.join(save_path, 'exponential_fit_results.csv')
    df_out.to_csv(csv_path, index=False, float_format='%.8g')
    print(f"  Exponential fit CSV exported -> {csv_path}")
    return csv_path


def generate_plots(all_trends_by_index, all_trend_sequences_by_index, save_path='plots_trend_acceleration'):
    for index_name, df_trends_current_index in all_trends_by_index.items():
        output_folder_for_index = os.path.join(save_path, index_name)
        if not os.path.exists(output_folder_for_index):
            os.makedirs(output_folder_for_index)
        plot_trend_duration_distribution(all_trends_by_index, save_path, index_name, plot_number=1)
        plot_trend_duration_distribution_exponential_fit(all_trends_by_index, save_path, index_name, plot_number=2)
    export_geometric_fit_to_csv(all_trends_by_index, save_path)
    export_exponential_fit_to_csv(all_trends_by_index, save_path)

def main():
    all_prices_data, _ = download_financial_data()
    if not all_prices_data:
        return
    all_trends_by_index, all_trend_sequences_by_index = precompute_trend_data(all_prices_data)
    if not all_trends_by_index and not all_trend_sequences_by_index:
        return
    save_directory = 'trend_duration_distribution_plots'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    generate_plots(all_trends_by_index, all_trend_sequences_by_index, save_directory)

if __name__ == "__main__":
    main()
