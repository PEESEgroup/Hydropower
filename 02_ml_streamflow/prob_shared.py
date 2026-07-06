"""
prob_shared.py - shared utility module for probabilistic forecasting
Data loading, sequence construction, metric computation, visualization

Shared by 08_lstm_prob.py / 08_tcn_prob.py / 08_itransformer_prob.py
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from numpy.lib.stride_tricks import sliding_window_view
from pathlib import Path
import json
import time
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# Random seed
# ============================================================
def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Data loading and preprocessing
# ============================================================
def load_data(dataset_file, meta_file, exclude_comids=None, min_train_hours=1200, min_val_hours=720):
    df = pd.read_parquet(dataset_file)
    with open(meta_file, 'r') as f:
        meta = json.load(f)

    feature_cols = meta['feature_cols']
    target_col = meta['target_col']

    df['datetime'] = pd.to_datetime(df['datetime'])
    if exclude_comids:
        df = df[~df['comid'].isin(exclude_comids)].reset_index(drop=True)

    df['residual'] = df[target_col] - df['flo_out']

    valid_comids = []
    for c in sorted(df['comid'].unique()):
        r = df[df['comid'] == c]
        if (len(r[r['split'] == 'train']) >= min_train_hours and
                len(r[r['split'] == 'val']) >= min_val_hours):
            valid_comids.append(c)

    return df, feature_cols, target_col, valid_comids


def precompute_sequences(df, feature_cols, target_col, valid_comids, max_seq_len):
    cache = {}
    for comid in valid_comids:
        river = df[df['comid'] == comid].sort_values('datetime').reset_index(drop=True)
        train_mask = river['split'] == 'train'

        fm = river.loc[train_mask, feature_cols].mean().values.astype(np.float32)
        fs = river.loc[train_mask, feature_cols].std().values.astype(np.float32)
        fs[fs < 1e-8] = 1.0
        # Use flow itself as the target (the Gamma distribution requires > 0)
        target_scale = float(river.loc[train_mask, target_col].mean())
        if target_scale < 1e-8:
            target_scale = 1.0
        X_all = ((river[feature_cols].values - fm) / fs).astype(np.float32)
        # Get the real flow and clip floor 0 values to prevent the Gamma NLL from blowing up
        y_raw = river[target_col].values.astype(np.float32)
        y_raw = np.maximum(y_raw, 1e-4)
        # Scale only, do not shift!
        y_all = (y_raw / target_scale).astype(np.float32)

        splits = river['split'].values

        n = len(X_all)
        if n <= max_seq_len:
            continue

        windows = sliding_window_view(X_all, max_seq_len, axis=0)
        windows = windows.transpose(0, 2, 1).copy()

        split_indices = {}
        for sn in ['train', 'test', 'val']:
            mask = np.zeros(n, dtype=bool)
            mask[max_seq_len:] = (splits[max_seq_len:] == sn)
            split_indices[sn] = np.where(mask)[0]

        cache[comid] = {
            'windows': windows, 'y_all': y_all,
            'split_indices': split_indices,
            'target_scale': target_scale,
            'flo_out': river['flo_out'].values.copy(),
            'target': river[target_col].values.copy(),
            'datetimes': river['datetime'].values.copy(),
            'is_observed': river['is_observed'].values.copy(),
        }
    return cache


def get_split_data(cache, seq_len, max_seq_len, split_name):
    idxs = cache['split_indices'][split_name]
    if len(idxs) == 0:
        return None, None, None
    offset = max_seq_len - seq_len
    window_idxs = idxs - max_seq_len
    X = cache['windows'][window_idxs, offset:, :]
    y = cache['y_all'][idxs]
    return X, y, idxs


# ============================================================
# Loss function: Gaussian NLL (PyTorch official version, numerically stable)
# ============================================================
class GammaNLLLoss(nn.Module):
    """
    Gamma distribution negative log-likelihood.
    Model outputs log_alpha, log_beta → alpha = exp(log_alpha), beta = exp(log_beta)
    Gamma mean = alpha / beta, variance = alpha / beta^2
    Naturally non-negative, well suited for flow prediction.
    """
    def forward(self, log_alpha, log_beta, target):
        # [Key fix] clamp before exp to prevent subnormal floats from slowing the GPU 10-100x
        # exp(-4.6)≈0.01, exp(9.21)≈9997; exp(-9.21)≈0.0001, exp(9.21)≈9997
        alpha = torch.exp(log_alpha.clamp(min=-4.6, max=9.21))
        beta  = torch.exp(log_beta.clamp(min=-9.21, max=9.21))
        target = target.clamp(min=1e-6)  # Gamma requires target > 0
        # NLL = -alpha*log(beta) + lgamma(alpha) - (alpha-1)*log(target) + beta*target
        nll = -alpha * torch.log(beta) + torch.lgamma(alpha) \
              - (alpha - 1) * torch.log(target) + beta * target
        return nll.mean()


# ============================================================
# Metric computation
# ============================================================
def calc_metrics(obs, sim):
    obs, sim = np.array(obs, dtype=float), np.array(sim, dtype=float)
    valid = ~(np.isnan(obs) | np.isnan(sim))
    obs, sim = obs[valid], sim[valid]
    if len(obs) < 5:
        return {'NSE': np.nan, 'KGE': np.nan, 'R2': np.nan, 'PBIAS': np.nan, 'RMSE': np.nan}
    r = np.corrcoef(obs, sim)[0, 1]
    nse = 1 - np.sum((obs-sim)**2) / np.sum((obs-np.mean(obs))**2)
    a = np.std(sim) / np.std(obs) if np.std(obs) > 0 else np.nan
    b = np.mean(sim) / np.mean(obs) if np.mean(obs) != 0 else np.nan
    kge = 1 - np.sqrt((r-1)**2 + (a-1)**2 + (b-1)**2) if not np.isnan(a) else np.nan
    pbias = 100 * np.sum(sim-obs) / np.sum(obs) if np.sum(obs) != 0 else np.nan
    return {'NSE': nse, 'KGE': kge, 'R2': r**2, 'PBIAS': pbias,
            'RMSE': np.sqrt(np.mean((obs-sim)**2))}


def calc_prob_metrics(obs, mu, sigma, levels=(0.90, 0.95)):
    obs = np.array(obs, dtype=float)
    mu = np.array(mu, dtype=float)
    sigma = np.array(sigma, dtype=float)
    valid = ~(np.isnan(obs) | np.isnan(mu) | np.isnan(sigma))
    obs, mu, sigma = obs[valid], mu[valid], sigma[valid]

    if len(obs) < 5:
        result = {}
        for lv in levels:
            pct = int(lv * 100)
            result[f'PICP_{pct}'] = np.nan
            result[f'PINAW_{pct}'] = np.nan
        result['CRPS'] = np.nan
        return result

    from scipy.stats import gamma as gamma_dist
    from scipy.special import beta as beta_func

    result = {}
    obs_range = obs.max() - obs.min()
    if obs_range < 1e-8: obs_range = 1.0

    # Recover Gamma parameters from mu and sigma
    sigma_safe = np.maximum(sigma, 1e-6)
    alpha = (mu / sigma_safe) ** 2
    beta_rate = mu / (sigma_safe ** 2)
    alpha = np.clip(alpha, 1e-2, 1e6)
    beta_rate = np.clip(beta_rate, 1e-6, 1e6)
    scale = 1.0 / beta_rate

    for lv in levels:
        pct = int(lv * 100)
        half = (1 - lv) / 2
        lower = gamma_dist.ppf(half, a=alpha, scale=scale)
        upper = gamma_dist.ppf(1 - half, a=alpha, scale=scale)
        covered = ((obs >= lower) & (obs <= upper)).mean()
        result[f'PICP_{pct}'] = covered
        avg_width = (upper - lower).mean()
        result[f'PINAW_{pct}'] = avg_width / obs_range

    # Gamma CRPS analytic formula:
    # CRPS = y*(2*F(y) - 1) - alpha/beta*(2*F_alpha+1(y) - 1) - 1/(beta*B(0.5, alpha))
    # where F is the CDF of Gamma(alpha, beta) and F_alpha+1 is the CDF of Gamma(alpha+1, beta)
    cdf_y = gamma_dist.cdf(obs, a=alpha, scale=scale)
    cdf_y_a1 = gamma_dist.cdf(obs, a=alpha + 1, scale=scale)
    crps = obs * (2 * cdf_y - 1) \
           - alpha * scale * (2 * cdf_y_a1 - 1) \
           - scale / beta_func(0.5, alpha)
    result['CRPS'] = np.mean(crps)

    return result


def calc_metrics_observed_only(obs, sim, is_observed):
    mask = np.array(is_observed, dtype=bool)
    if mask.sum() < 5:
        return {'NSE_obs': np.nan, 'KGE_obs': np.nan}
    m = calc_metrics(obs[mask], sim[mask])
    return {'NSE_obs': m['NSE'], 'KGE_obs': m['KGE']}


# ============================================================
# Training loop (generic) - returns predictions for all splits
# ============================================================
def train_probabilistic_model(model, cache, seq_len, max_seq_len, params, device, seed=42, trial=None):
    seed_everything(seed)

    # [Fix 1: strictly align naming with function]
    X_tr, y_tr, idx_tr = get_split_data(cache, seq_len, max_seq_len, 'train')
    X_va, y_va, idx_va = get_split_data(cache, seq_len, max_seq_len, 'val')    # validation set, for early stopping and tuning
    X_te, y_te, idx_te = get_split_data(cache, seq_len, max_seq_len, 'test')   # test set, for the final blind evaluation

    if X_tr is None or len(X_tr) < 20 or X_te is None or len(X_te) < 5:
        return None

    pin = (device.type == 'cuda')
    use_amp = (device.type == 'cuda')
    tr_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=params['batch_size'], shuffle=True, pin_memory=pin,
        generator=torch.Generator().manual_seed(seed),
    )
    va_loader = None
    if X_va is not None and len(X_va) > 0:
        va_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
            batch_size=params['batch_size'], shuffle=False, pin_memory=pin,
        )

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=5)
    crit = GammaNLLLoss()

    best_loss, pat_cnt, best_state = float('inf'), 0, None
    epochs = params.get('epochs', 300)
    patience = params.get('patience', 15)
    max_train_seconds = params.get('max_train_seconds', 600)  # at most 10 minutes per combo
    t_start = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        for Xb, yb in tr_loader:
            Xb = Xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                log_alpha, log_beta = model(Xb)
                loss = crit(log_alpha, log_beta, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        if va_loader:
            el = 0; ne = 0
            with torch.no_grad():
                for Xb, yb in va_loader:
                    Xb = Xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    with torch.amp.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                        la_b, lb_b = model(Xb)
                        el += crit(la_b, lb_b, yb).item()
                    ne += 1
            el /= max(ne, 1)
        else:
            el = loss.item()

        sched.step(el)
        if trial is not None:
            trial.report(el, ep)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()
        if el < best_loss:
            best_loss = el; pat_cnt = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat_cnt += 1
            if pat_cnt >= patience:
                break
        # Timeout guard: prevent anomalies such as subnormals from making a single combo run for hours
        if time.time() - t_start > max_train_seconds:
            break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    # --- Predict for all splits ---
    def predict_split(X_data, idxs):
        if X_data is None or len(X_data) == 0:
            return None
        loader = DataLoader(TensorDataset(torch.from_numpy(X_data)), batch_size=params['batch_size'], shuffle=False)
        la_list, lb_list = [], []
        with torch.no_grad():
            for (Xb,) in loader:
                with torch.amp.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                    la_b, lb_b = model(Xb.to(device))
                la_list.extend(la_b.float().cpu().numpy())
                lb_list.extend(lb_b.float().cpu().numpy())

        log_alpha = np.array(la_list)
        log_beta = np.array(lb_list)
        alpha = np.exp(np.clip(log_alpha, -5, 10))
        beta = np.exp(np.clip(log_beta, -10, 10))

        # Gamma mean and standard deviation (standardized space)
        gamma_mean = alpha / beta
        gamma_std = np.sqrt(alpha) / beta

        # De-standardize back to the real flow space
        target_scale = cache['target_scale']
        pred_flow_mu = gamma_mean * target_scale
        pred_sigma = gamma_std * target_scale

        # Ensure non-negative
        pred_flow_mu = np.maximum(pred_flow_mu, 0.0)

        # [New: directly compute rigorous Gamma-distribution confidence intervals]
        from scipy.stats import gamma as gamma_dist
        # Gamma scale parameter (Scale) in real space = target_scale / beta
        real_scale = target_scale / beta
        lower_90 = gamma_dist.ppf(0.05, a=alpha, scale=real_scale)
        upper_90 = gamma_dist.ppf(0.95, a=alpha, scale=real_scale)
        lower_95 = gamma_dist.ppf(0.025, a=alpha, scale=real_scale)
        upper_95 = gamma_dist.ppf(0.975, a=alpha, scale=real_scale)

        true_flow = cache['target'][idxs]
        flo_out = cache['flo_out'][idxs]
        dates = cache['datetimes'][idxs]
        is_obs = cache['is_observed'][idxs]

        return {
            'pred_mu': pred_flow_mu, 'pred_sigma': pred_sigma,
            'lower_90': lower_90, 'upper_90': upper_90,  # ← pass out lower and upper bounds
            'lower_95': lower_95, 'upper_95': upper_95,  # ← pass out lower and upper bounds
            'true_flow': true_flow, 'flo_out': flo_out,
            'dates': dates, 'is_observed': is_obs,
            'alpha': alpha, 'beta': beta,
        }

    splits_pred = {}
    splits_pred['train'] = predict_split(X_tr, idx_tr)
    splits_pred['val'] = predict_split(X_va, idx_va)
    splits_pred['test'] = predict_split(X_te, idx_te)

    # Test-set metrics (used only for final evaluation)
    tp = splits_pred['test']
    m_test = calc_metrics(tp['true_flow'], tp['pred_mu'])
    m_test_obs = calc_metrics_observed_only(tp['true_flow'], tp['pred_mu'], tp['is_observed'])
    m_test_prob = calc_prob_metrics(tp['true_flow'], tp['pred_mu'], tp['pred_sigma'])
    m_test_swat = calc_metrics(tp['true_flow'], tp['flo_out'])

    # Validation-set metrics ([Fix 2: return val metrics for grid-search comparison])
    vp = splits_pred['val']
    m_val = calc_metrics(vp['true_flow'], vp['pred_mu']) if vp else {}
    m_val_prob = calc_prob_metrics(vp['true_flow'], vp['pred_mu'], vp['pred_sigma']) if vp else {}

    # Training-set metrics
    trp = splits_pred['train']
    m_train = calc_metrics(trp['true_flow'], trp['pred_mu']) if trp else {}

    return {
        'metrics_point': m_test,
        'metrics_obs': m_test_obs,
        'metrics_prob': m_test_prob,
        'metrics_swat': m_test_swat,
        'metrics_val': m_val,
        'metrics_val_prob': m_val_prob,
        'metrics_train': m_train,
        'splits_pred': splits_pred,
        'best_epoch': ep - pat_cnt,
        'total_epochs': ep,              # ← add this line
        'state_dict': best_state,
    }

# ============================================================
# Save results
# ============================================================
def save_results(results_dict, all_combo_details, out_dir, model_name):
    """
    Save:
    1. Best-model metrics for each river (metrics csv)
    2. Detailed metrics of all combos for each river (grid search detail csv)
    3. Predictions for each river (predictions csv, including train/test/val)
    """
    from scipy.stats import norm
    z90 = norm.ppf(0.95)
    z95 = norm.ppf(0.975)

    final_results = []
    final_predictions = []

    for comid, r in sorted(results_dict.items()):
        if r is None:
            continue
        mp = r['metrics_point']
        mo = r['metrics_obs']
        mb = r['metrics_prob']
        ms = r['metrics_swat']
        mt = r.get('metrics_train', {})

        final_results.append({
            'comid': comid,
            'swat_NSE': ms.get('NSE', np.nan), 'swat_KGE': ms.get('KGE', np.nan),
            'train_NSE': mt.get('NSE', np.nan),
            **{k: v for k, v in mp.items()},
            **{k: v for k, v in mo.items()},
            **{k: v for k, v in mb.items()},
            **{f'best_{k}': v for k, v in r.get('best_params', {}).items()},
        })

        # Save predictions for all splits
        for split_name in ['train', 'test', 'val']:
            sp = r['splits_pred'].get(split_name)
            if sp is None:
                continue
            for i in range(len(sp['dates'])):
                final_predictions.append({
                    'datetime': sp['dates'][i],
                    'comid': comid,
                    'split': split_name,
                    'grfr_obs': sp['true_flow'][i],
                    'swat_raw': sp['flo_out'][i],
                    'pred_mu': sp['pred_mu'][i],
                    'pred_sigma': sp['pred_sigma'][i],
                    'lower_90': sp['lower_90'][i],
                    'upper_90': sp['upper_90'][i],
                    'lower_95': sp['lower_95'][i],
                    'upper_95': sp['upper_95'][i],
                    'is_observed': sp['is_observed'][i],
                })

    fr = pd.DataFrame(final_results)
    fp = pd.DataFrame(final_predictions)

    if len(fp) > 0 and 'datetime' in fp.columns:
        fp['datetime'] = pd.to_datetime(fp['datetime'])
    else:
        print(f'WARNING: predictions DataFrame is empty ({len(final_predictions)} rows)')

    fr.to_csv(out_dir / f'{model_name.lower()}_metrics.csv', index=False)
    fp.to_csv(out_dir / f'{model_name.lower()}_predictions.csv', index=False)

    # Save detailed results for all combos
    if all_combo_details:
        detail_df = pd.DataFrame(all_combo_details)
        detail_df.to_csv(out_dir / f'{model_name.lower()}_grid_detail.csv', index=False)
        print(f'Saved: {model_name.lower()}_grid_detail.csv ({len(detail_df)} rows)')

    print(f'Saved: {model_name.lower()}_metrics.csv ({len(fr)} rivers)')
    print(f'Saved: {model_name.lower()}_predictions.csv ({len(fp)} rows)')

    return fr, fp


# ============================================================
# Visualization
# ============================================================
def plot_all_rivers(fr, fp, model_name, out_dir, color='steelblue'):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    plot_comids = sorted(fr['comid'].values)
    n = len(plot_comids)
    if n == 0:
        print('No rivers to plot'); return

    # --- Figure 1: time series + confidence interval (validation period) ---
    fig, axes = plt.subplots(n, 2, figsize=(20, 4.5 * n))
    if n == 1: axes = axes.reshape(1, -1)

    for i, comid in enumerate(plot_comids):
        sub = fp[(fp['comid'] == comid) & (fp['split'] == 'val')].sort_values('datetime')
        row = fr[fr['comid'] == comid].iloc[0]

        # Left: daily-mean overview
        ax = axes[i, 0]
        sub_d = sub.set_index('datetime').resample('D').mean(numeric_only=True).reset_index()
        ax.fill_between(sub_d['datetime'].values, sub_d['lower_95'].values, sub_d['upper_95'].values,
                        alpha=0.15, color=color, label='95% PI')
        ax.fill_between(sub_d['datetime'].values, sub_d['lower_90'].values, sub_d['upper_90'].values,
                        alpha=0.25, color=color, label='90% PI')
        ax.plot(sub_d['datetime'].values, sub_d['grfr_obs'].values, 'k', lw=1.0, label='GRFR')
        ax.plot(sub_d['datetime'].values, sub_d['swat_raw'].values, 'grey', lw=0.6, alpha=0.5, label='SWAT+')
        ax.plot(sub_d['datetime'].values, sub_d['pred_mu'].values, color=color, lw=1.0, label=model_name)
        ax.set_title(f'COMID {comid} - val daily\n'
                     f'NSE={row["NSE"]:.3f} KGE={row["KGE"]:.3f} '
                     f'PICP90={row["PICP_90"]:.1%} CRPS={row["CRPS"]:.1f}', fontsize=9)
        ax.set_ylabel('Flow (m³/s)', fontsize=8)
        ax.legend(fontsize=6, loc='upper right')
        ax.grid(True, alpha=0.2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        # Right: hourly resolution for one week
        ax = axes[i, 1]
        mid = len(sub) // 2
        w = sub.iloc[mid:mid+168].copy()
        ax.fill_between(w['datetime'].values, w['lower_95'].values, w['upper_95'].values,
                        alpha=0.15, color=color)
        ax.fill_between(w['datetime'].values, w['lower_90'].values, w['upper_90'].values,
                        alpha=0.25, color=color)
        ax.plot(w['datetime'].values, w['grfr_obs'].values, 'k', lw=1.0, label='GRFR')
        ax.plot(w['datetime'].values, w['pred_mu'].values, color=color, lw=1.0, label=model_name)
        obs_mask = w['is_observed'].values == 1
        ax.scatter(w['datetime'].values[obs_mask], w['grfr_obs'].values[obs_mask],
                   c='k', s=10, zorder=5, label='3h obs')
        ax.set_title(f'COMID {comid} - val hourly (1 week)', fontsize=9)
        ax.legend(fontsize=6, loc='upper right')
        ax.grid(True, alpha=0.2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, fontsize=7)

    plt.suptitle(f'{model_name} | Median NSE={fr["NSE"].median():.3f}  '
                 f'PICP90={fr["PICP_90"].median():.1%}  CRPS={fr["CRPS"].median():.1f}',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_dir / f'{model_name.lower()}_timeseries.png', dpi=200, bbox_inches='tight')
    plt.close()

    # --- Figure 2: scatter plots (three columns: train / test / val) ---
    ncols = 3
    fig, axes = plt.subplots(n, ncols, figsize=(5 * ncols, 5 * n))
    if n == 1: axes = axes.reshape(1, -1)

    split_colors = {'train': 'grey', 'test': 'steelblue', 'val': color}
    split_names = ['train', 'test', 'val']

    for i, comid in enumerate(plot_comids):
        row = fr[fr['comid'] == comid].iloc[0]

        for j, sn in enumerate(split_names):
            ax = axes[i, j]
            sub = fp[(fp['comid'] == comid) & (fp['split'] == sn)]

            if len(sub) == 0:
                ax.set_visible(False)
                continue

            obs_v = sub['grfr_obs'].values
            pred_v = sub['pred_mu'].values
            sc = split_colors[sn]

            ax.scatter(obs_v, pred_v, c=sc, s=3, alpha=0.3, rasterized=True)
            vmin = min(obs_v.min(), pred_v.min())
            vmax = max(obs_v.max(), pred_v.max())
            ax.plot([vmin, vmax], [vmin, vmax], 'k--', lw=0.8, alpha=0.5)

            # Metrics for this split
            sm = calc_metrics(obs_v, pred_v)
            ax.set_xlabel('Obs (m³/s)', fontsize=8)
            ax.set_ylabel('Pred (m³/s)', fontsize=8)
            ax.set_title(f'COMID {comid} [{sn}]\n'
                         f'NSE={sm["NSE"]:.3f} R²={sm["R2"]:.3f}', fontsize=9)
            ax.set_aspect('equal', adjustable='datalim')
            ax.grid(True, alpha=0.2)
            ax.tick_params(labelsize=7)

    plt.suptitle(f'{model_name} - scatter (train / test / val)', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_dir / f'{model_name.lower()}_scatter.png', dpi=200, bbox_inches='tight')
    plt.close()

    # --- Figure 3: SWAT+ vs Model comparison scatter (val only) ---
    ncols_c = 4
    nrows_c = (n + ncols_c - 1) // ncols_c
    fig, axes = plt.subplots(nrows_c, ncols_c, figsize=(5*ncols_c, 5*nrows_c))
    axes_flat = axes.flatten() if nrows_c > 1 or ncols_c > 1 else [axes]

    for i, comid in enumerate(plot_comids):
        ax = axes_flat[i]
        sub = fp[(fp['comid'] == comid) & (fp['split'] == 'val')]
        row = fr[fr['comid'] == comid].iloc[0]

        obs_v = sub['grfr_obs'].values
        swat_v = sub['swat_raw'].values
        pred_v = sub['pred_mu'].values

        ax.scatter(obs_v, swat_v, c='grey', s=3, alpha=0.3, label='SWAT+', rasterized=True)
        ax.scatter(obs_v, pred_v, c=color, s=3, alpha=0.3, label=model_name, rasterized=True)

        vmin = min(obs_v.min(), swat_v.min(), pred_v.min())
        vmax = max(obs_v.max(), swat_v.max(), pred_v.max())
        ax.plot([vmin, vmax], [vmin, vmax], 'k--', lw=0.8, alpha=0.5)
        ax.set_xlabel('Obs (m³/s)', fontsize=8)
        ax.set_ylabel('Pred (m³/s)', fontsize=8)
        ax.set_title(f'COMID {comid}\nSWAT+ NSE={row["swat_NSE"]:.3f} → {model_name} NSE={row["NSE"]:.3f}', fontsize=9)
        ax.legend(fontsize=6, loc='upper left')
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(True, alpha=0.2); ax.tick_params(labelsize=7)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.suptitle(f'SWAT+ vs {model_name} - val scatter', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_dir / f'{model_name.lower()}_scatter_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()

    # --- Figure 4: parameter distribution (if fr has best_* columns) ---
    best_cols = [c for c in fr.columns if c.startswith('best_')]
    if best_cols:
        fig, axes = plt.subplots(1, len(best_cols), figsize=(4 * len(best_cols), 4))
        if len(best_cols) == 1: axes = [axes]
        for i, col in enumerate(best_cols):
            ax = axes[i]
            vals = fr[col].value_counts().sort_index()
            ax.bar(range(len(vals)), vals.values, color=color, alpha=0.8)
            ax.set_xticks(range(len(vals)))
            ax.set_xticklabels(vals.index.astype(str), rotation=45, fontsize=8)
            ax.set_xlabel(col.replace('best_', ''))
            ax.set_ylabel('# rivers')
            ax.set_title(f'{col} distribution')
            ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(out_dir / f'{model_name.lower()}_param_distribution.png', dpi=200, bbox_inches='tight')
        plt.close()

    print(f'Saved plots to {out_dir}/')
