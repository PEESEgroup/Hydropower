#!/usr/bin/env python3
"""
run_future_inference.py - ML correction inference for future-scenario SWAT+ output
============================================================

For each GCM×SSP scenario of each region:
1. Load training data → compute normalization params (fm, fs, target_scale) + channel_id→COMID mapping
2. Load trained LSTM-Transformer models (per-river)
3. Load future SWAT+ output (ml_features.parquet)
4. Inference: normalize → sliding window → forward pass → Gamma → pred_mu
5. Also run inference on historical data → historical Y_hat
6. Compute mean annual 8760-hour curve, compare future vs historical

Usage:
  python run_future_inference.py --region brazil/brazil_nordeste
  python run_future_inference.py --all --workers 4
  python run_future_inference.py --all --device cuda
"""

import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math
import json
import argparse
import logging
import time
import gc
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# Model architecture (from 09_lstm_transformer_prob.py)
# ============================================================

class LSTMBlock(nn.Module):
    def __init__(self, d_model, num_layers=1, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.lstm = nn.LSTM(
            input_size=d_model, hidden_size=d_model,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        h = self.norm(x)
        h, _ = self.lstm(h)
        h = self.dropout(h)
        return residual + h


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class LSTMTransformerProb(nn.Module):
    def __init__(self, n_features, seq_len, d_model, n_heads,
                 e_layers, lstm_layers, d_ff, dropout):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.lstm_block = LSTMBlock(d_model, num_layers=lstm_layers,
                                     dropout=dropout)
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len + 10)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                              num_layers=e_layers)
        self.attn_pool = nn.Linear(d_model, 1)
        self.fc_log_alpha = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(32, 1),
        )
        self.fc_log_beta = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(32, 1),
        )

    def forward(self, x):
        h = self.input_proj(x)
        h = self.input_norm(h)
        h = self.lstm_block(h)
        h = self.pos_enc(h)
        h = self.encoder(h)
        scores = self.attn_pool(h).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (h * weights).sum(dim=1)
        log_alpha = self.fc_log_alpha(pooled).squeeze(-1)
        log_beta = self.fc_log_beta(pooled).squeeze(-1)
        return log_alpha, log_beta


# ============================================================
# Inference utilities
# ============================================================

def run_inference(X_norm, seq_len, model, device, batch_size=4096):
    """Run model inference on normalized feature array."""
    n = len(X_norm)
    if n <= seq_len:
        return None, None

    windows = sliding_window_view(X_norm, seq_len, axis=0)
    windows = windows[:-1]  # last window predicts beyond data range
    windows = windows.transpose(0, 2, 1).copy()

    model.eval()
    la_all, lb_all = [], []
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[i:i + batch_size]).to(device)
            la, lb = model(batch)
            la_all.append(la.float().cpu().numpy())
            lb_all.append(lb.float().cpu().numpy())

    log_alpha = np.concatenate(la_all)
    log_beta = np.concatenate(lb_all)
    alpha = np.exp(np.clip(log_alpha, -5, 10))
    beta = np.exp(np.clip(log_beta, -10, 10))
    return alpha, beta


def gamma_to_flow(alpha, beta, target_scale):
    """Convert Gamma params to predicted flow and confidence intervals."""
    from scipy.stats import gamma as gamma_dist

    pred_mu = np.maximum((alpha / beta) * target_scale, 0.0)
    pred_sigma = (np.sqrt(alpha) / beta) * target_scale
    real_scale = target_scale / beta
    lower_90 = gamma_dist.ppf(0.05, a=alpha, scale=real_scale)
    upper_90 = gamma_dist.ppf(0.95, a=alpha, scale=real_scale)

    return pred_mu, pred_sigma, lower_90, upper_90


def compute_8760_curve(datetimes, values):
    """Compute mean annual 8760-hour curve."""
    df = pd.DataFrame({
        'datetime': pd.to_datetime(datetimes),
        'value': values,
    })
    df['hour_of_year'] = (df['datetime'].dt.dayofyear - 1) * 24 + df['datetime'].dt.hour
    curve = df.groupby('hour_of_year')['value'].mean()
    full_idx = pd.RangeIndex(0, 8760)
    curve = curve.reindex(full_idx).interpolate()
    return curve


# ============================================================
# Per-region inference
# ============================================================

def infer_region(region_rel, training_base, future_base, output_base,
                 device_str='cpu'):
    """
    Run inference for one region across all GCM×SSP scenarios.
    region_rel: e.g. "brazil/brazil_nordeste"
    """
    t0 = time.time()
    region_name = region_rel.split('/')[-1]
    continent = region_rel.split('/')[0]

    training_dir = Path(training_base) / region_rel
    model_dir = training_dir / 'LSTM_Transformer_prob' / 'models'
    metrics_file = (training_dir / 'LSTM_Transformer_prob'
                    / 'lstm_transformer_prob_metrics.csv')
    dataset_file = training_dir / 'datasets' / 'ml_dataset_hourly.parquet'
    meta_file = training_dir / 'datasets' / 'ml_dataset_hourly_meta.json'

    # Try both local (single region_name) and server (double region_name) paths
    future_region_dir = (Path(future_base) / continent / region_name
                         / 'Scenarios' / 'isimip3b')
    if not future_region_dir.exists():
        future_region_dir = (Path(future_base) / continent / region_name
                             / region_name / 'Scenarios' / 'isimip3b')

    output_dir = Path(output_base) / region_rel
    output_dir.mkdir(parents=True, exist_ok=True)

    log = logging.getLogger(region_name)

    # ---- Validate inputs ----
    for f, desc in [(metrics_file, 'metrics'),
                    (dataset_file, 'training parquet'),
                    (meta_file, 'meta.json')]:
        if not f.exists():
            log.error(f"Missing {desc}: {f}")
            return f"ERROR: missing {desc}"

    if not future_region_dir.exists():
        log.warning(f"No future data: {future_region_dir}")
        return f"SKIP: no future data"

    # ---- Load meta & metrics ----
    with open(meta_file) as f:
        meta = json.load(f)
    feature_cols = meta['feature_cols']
    target_col = meta['target_col']
    n_features = len(feature_cols)

    metrics_df = pd.read_csv(metrics_file)
    log.info(f"{region_rel}: {len(metrics_df)} COMIDs with trained models")

    # ---- Load training data ----
    log.info(f"Loading training data ...")
    train_df = pd.read_parquet(dataset_file)
    train_df['datetime'] = pd.to_datetime(train_df['datetime'])

    # channel_id → comid mapping
    ch_to_comid = dict(zip(train_df['channel_id'], train_df['comid']))
    comid_to_chs = {}
    for ch, comid in ch_to_comid.items():
        comid_to_chs.setdefault(int(comid), set()).add(ch)

    # ---- Precompute per-COMID normalization & load models ----
    device = torch.device(device_str)
    comid_info = {}

    for _, mrow in metrics_df.iterrows():
        comid = int(mrow['comid'])
        seq_len = int(mrow['best_seq_len'])
        d_model = int(mrow['best_d_model'])
        n_heads = int(mrow['best_n_heads'])
        e_layers = int(mrow['best_e_layers'])
        lstm_layers = int(mrow['best_lstm_layers'])
        dropout = float(mrow['best_dropout'])
        d_ff = d_model * 2

        mf = model_dir / f'lstm_transformer_prob_{comid}.pt'
        if not mf.exists():
            log.warning(f"  COMID {comid}: model file missing, skip")
            continue

        river = (train_df[train_df['comid'] == comid]
                 .sort_values('datetime').reset_index(drop=True))
        train_mask = river['split'] == 'train'
        if train_mask.sum() < 100:
            log.warning(f"  COMID {comid}: too few training rows, skip")
            continue

        fm = river.loc[train_mask, feature_cols].mean().values.astype(np.float32)
        fs = river.loc[train_mask, feature_cols].std().values.astype(np.float32)
        fs[fs < 1e-8] = 1.0
        target_scale = float(river.loc[train_mask, target_col].mean())
        if target_scale < 1e-8:
            target_scale = 1.0

        model = LSTMTransformerProb(
            n_features, seq_len, d_model, n_heads,
            e_layers, lstm_layers, d_ff, dropout,
        )
        sd = torch.load(mf, map_location=device, weights_only=True)
        model.load_state_dict(sd)
        model = model.to(device)
        model.eval()

        comid_info[comid] = {
            'model': model,
            'seq_len': seq_len,
            'fm': fm, 'fs': fs,
            'target_scale': target_scale,
            'channel_ids': comid_to_chs.get(comid, set()),
            'hist_features': river[feature_cols].values.astype(np.float32),
            'hist_datetimes': river['datetime'].values,
            'hist_flo_out': river['flo_out'].values.astype(np.float32),
            'hist_observed': river[target_col].values.astype(np.float32) if target_col in river.columns else None,
        }

    log.info(f"  Loaded {len(comid_info)} models")

    # ---- Historical inference (all data through model → Y_hat) ----
    hist_results = {}
    for comid, info in comid_info.items():
        X_hist = (info['hist_features'] - info['fm']) / info['fs']
        alpha, beta = run_inference(X_hist, info['seq_len'], info['model'],
                                    device)
        if alpha is None:
            continue
        pred_mu, _, _, _ = gamma_to_flow(alpha, beta, info['target_scale'])
        dts = info['hist_datetimes'][info['seq_len']:]
        obs_8760 = None
        if info['hist_observed'] is not None:
            obs_8760 = compute_8760_curve(
                dts, info['hist_observed'][info['seq_len']:])
        hist_results[comid] = {
            'pred_mu': pred_mu,
            'datetimes': dts,
            '8760': compute_8760_curve(dts, pred_mu),
            'flo_out_8760': compute_8760_curve(
                dts, info['hist_flo_out'][info['seq_len']:]),
            'obs_8760': obs_8760,
        }

    log.info(f"  Historical inference done for {len(hist_results)} COMIDs")

    # Free training data memory
    del train_df
    gc.collect()

    # ---- Discover scenarios ----
    scenarios = []
    for gcm_dir in sorted(future_region_dir.iterdir()):
        if not gcm_dir.is_dir():
            continue
        for ssp_dir in sorted(gcm_dir.iterdir()):
            if not ssp_dir.is_dir():
                continue
            pq = ssp_dir / 'datasets' / 'ml_features.parquet'
            if pq.exists() and pq.stat().st_size > 0:
                scenarios.append((gcm_dir.name, ssp_dir.name, pq))

    if not scenarios:
        log.warning(f"  No completed scenario parquets")
        return f"SKIP: no scenario parquets"

    log.info(f"  {len(scenarios)} scenarios found")

    # ---- Future inference per scenario ----
    summary_rows = []

    for gcm, ssp, pq_file in scenarios:
        log.info(f"  Processing {gcm}/{ssp} ...")
        fut_df = pd.read_parquet(pq_file)
        fut_df['datetime'] = pd.to_datetime(fut_df['datetime'])

        out_scenario = output_dir / gcm / ssp
        out_scenario.mkdir(parents=True, exist_ok=True)

        scenario_dfs = []

        for comid, info in comid_info.items():
            river_fut = (fut_df[fut_df['channel_id'].isin(info['channel_ids'])]
                         .sort_values('datetime').reset_index(drop=True))
            if len(river_fut) == 0:
                continue

            missing = [c for c in feature_cols if c not in river_fut.columns]
            if missing:
                log.warning(f"    COMID {comid}: missing {missing}")
                continue

            X_fut = ((river_fut[feature_cols].values.astype(np.float32)
                      - info['fm']) / info['fs'])
            alpha, beta = run_inference(X_fut, info['seq_len'],
                                        info['model'], device)
            if alpha is None:
                continue

            pred_mu, pred_sigma, lower_90, upper_90 = gamma_to_flow(
                alpha, beta, info['target_scale'])
            sl = info['seq_len']
            dts = river_fut['datetime'].values[sl:]
            flo_out = river_fut['flo_out'].values[sl:]

            out_df = pd.DataFrame({
                'datetime': dts,
                'comid': comid,
                'flo_out_swat': flo_out,
                'flo_corrected': pred_mu,
                'pred_sigma': pred_sigma,
                'lower_90': lower_90,
                'upper_90': upper_90,
            })
            scenario_dfs.append(out_df)

            # 8760 curves
            fut_8760 = compute_8760_curve(dts, pred_mu)
            fut_swat_8760 = compute_8760_curve(dts, flo_out)
            hist = hist_results.get(comid)

            row = {
                'region': region_rel,
                'comid': comid, 'gcm': gcm, 'ssp': ssp,
                'fut_ml_mean': float(np.nanmean(pred_mu)),
                'fut_swat_mean': float(np.nanmean(flo_out)),
            }
            if hist:
                row['hist_ml_mean'] = float(np.nanmean(hist['pred_mu']))
                row['ratio_fut_hist'] = (row['fut_ml_mean']
                                         / max(row['hist_ml_mean'], 1e-8))
            summary_rows.append(row)

        # Save merged scenario output
        if scenario_dfs:
            merged = pd.concat(scenario_dfs, ignore_index=True)
            merged.to_parquet(out_scenario / 'corrected_all.parquet',
                              index=False)
            log.info(f"    Saved {len(scenario_dfs)} COMIDs, "
                     f"{len(merged)} rows")

        del fut_df
        gc.collect()

    # ---- Save summary & 8760 plots ----
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(output_dir / '8760_summary.csv', index=False)
        log.info(f"  Saved 8760_summary.csv ({len(summary_rows)} entries)")

        _plot_8760(summary_rows, comid_info, hist_results, output_dir,
                   scenarios, region_name, feature_cols, device)

    # Clean up models
    for info in comid_info.values():
        del info['model']
    del comid_info
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    log.info(f"  {region_rel} done in {elapsed / 60:.1f} min")
    return f"OK: {len(summary_rows)} comid×scenario results in {elapsed / 60:.1f} min"


def _plot_8760(summary_rows, comid_info, hist_results, output_dir,
               scenarios, region_name, feature_cols, device):
    """Generate 8760-hour comparison plots per COMID."""
    plot_dir = output_dir / '8760_plots'
    plot_dir.mkdir(exist_ok=True)

    comids_with_data = sorted(set(r['comid'] for r in summary_rows))

    for comid in comids_with_data:
        entries = [r for r in summary_rows if r['comid'] == comid]
        hist = hist_results.get(comid)

        n_panels = len(entries) + 1
        fig, axes = plt.subplots(n_panels, 1,
                                 figsize=(14, 2.8 * n_panels),
                                 sharex=True)
        if n_panels == 1:
            axes = [axes]

        for i, entry in enumerate(entries):
            gcm, ssp = entry['gcm'], entry['ssp']
            ax = axes[i]

            sc_dir = output_dir / gcm / ssp
            pq = sc_dir / 'corrected_all.parquet'
            if pq.exists():
                df = pd.read_parquet(pq)
                df = df[df['comid'] == comid]
                if len(df) > 0:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    ml_8760 = compute_8760_curve(
                        df['datetime'].values, df['flo_corrected'].values)
                    swat_8760 = compute_8760_curve(
                        df['datetime'].values, df['flo_out_swat'].values)
                    x = np.arange(len(ml_8760))
                    ax.plot(x, ml_8760.values,
                            linewidth=0.8, label='ML corrected')
                    ax.plot(x, swat_8760.values,
                            linewidth=0.6, alpha=0.7, label='SWAT+ raw')

            if hist:
                hx = np.arange(len(hist['8760']))
                if hist.get('obs_8760') is not None:
                    ax.plot(hx, hist['obs_8760'].values,
                            linewidth=0.8, alpha=0.6, color='black',
                            label='Historical Observed')
                ax.plot(hx, hist['8760'].values,
                        linewidth=0.6, alpha=0.5, color='gray',
                        label='Historical ML')
                ax.plot(hx, hist['flo_out_8760'].values,
                        linewidth=0.5, alpha=0.4, color='gray',
                        linestyle='--', label='Historical SWAT+')

            ax.set_title(f"COMID {comid}: {gcm} / {ssp}", fontsize=9)
            ax.legend(fontsize=7, ncol=4)
            ax.grid(True, alpha=0.3)

        # Last panel: overlay all scenarios
        ax = axes[-1]
        for entry in entries:
            gcm, ssp = entry['gcm'], entry['ssp']
            pq = output_dir / gcm / ssp / 'corrected_all.parquet'
            if pq.exists():
                df = pd.read_parquet(pq)
                df = df[df['comid'] == comid]
                if len(df) > 0:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    c8760 = compute_8760_curve(
                        df['datetime'].values, df['flo_corrected'].values)
                    ax.plot(np.arange(len(c8760)), c8760.values,
                            linewidth=0.6, label=f"{gcm}/{ssp}")

        if hist:
            if hist.get('obs_8760') is not None:
                ax.plot(np.arange(len(hist['obs_8760'])), hist['obs_8760'].values,
                        linewidth=1.2, color='black',
                        label='Historical Observed')
            ax.plot(np.arange(len(hist['8760'])), hist['8760'].values,
                    linewidth=1.0, color='dimgray', linestyle='--',
                    label='Historical ML')

        ax.set_title(f"COMID {comid}: all scenarios", fontsize=9)
        ax.set_xlabel("Hour of Year")
        ax.legend(fontsize=6, ncol=3)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(plot_dir / f'{comid}_8760.png', dpi=150,
                    bbox_inches='tight')
        plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='ML correction for future SWAT+ scenarios')
    parser.add_argument('--region', type=str,
                        help='e.g. brazil/brazil_nordeste')
    parser.add_argument('--all', action='store_true',
                        help='Run all regions with trained models')
    parser.add_argument('--training-base', default='/datasets/swat_global',
                        help='Root of training data')
    parser.add_argument('--future-base', default='/data/swat_global',
                        help='Root of future scenario data')
    parser.add_argument('--output-dir', default='/data/swat_global/inference_output',
                        help='Where to save corrected output')
    parser.add_argument('--device', default='cpu',
                        help='cpu or cuda')
    parser.add_argument('--workers', type=int, default=1,
                        help='Parallel regions (only with --all)')
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(Path(args.output_dir) / 'inference.log'),
            logging.StreamHandler(),
        ]
    )

    if args.region:
        regions = [args.region]
    elif args.all:
        regions = []
        for cont_dir in sorted(Path(args.training_base).iterdir()):
            if not cont_dir.is_dir() or cont_dir.name.endswith('.csv'):
                continue
            for reg_dir in sorted(cont_dir.iterdir()):
                if not reg_dir.is_dir():
                    continue
                rel = f"{cont_dir.name}/{reg_dir.name}"
                mf = (reg_dir / 'LSTM_Transformer_prob'
                      / 'lstm_transformer_prob_metrics.csv')
                if mf.exists():
                    regions.append(rel)
        logging.info(f"Discovered {len(regions)} regions with trained models")
    else:
        parser.error("Specify --region or --all")

    if args.workers > 1 and len(regions) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {
                pool.submit(infer_region, r, args.training_base,
                            args.future_base, args.output_dir,
                            args.device): r
                for r in regions
            }
            for f in as_completed(futs):
                reg = futs[f]
                try:
                    res = f.result()
                    logging.info(f"[DONE] {reg}: {res}")
                except Exception as e:
                    logging.error(f"[FAIL] {reg}: {e}", exc_info=True)
    else:
        for reg in regions:
            try:
                res = infer_region(reg, args.training_base, args.future_base,
                                   args.output_dir, args.device)
                logging.info(f"[DONE] {reg}: {res}")
            except Exception as e:
                logging.error(f"[FAIL] {reg}: {e}", exc_info=True)


if __name__ == '__main__':
    main()
