"""
11_lstm_transformer_prob.py - LSTM-Transformer probabilistic forecasting
LSTM front-end temporal feature extraction + Transformer encoder global attention modeling + Gamma dual-head
Independent grid search per river
Selection criterion: lexicographic ordering (NSE>0 preferred, smallest CRPS, largest NSE)

Usage: python 11_lstm_transformer_prob.py --region /path/to/region [--workers 2] [--gpus 0,1]

Architecture: Input -> Linear proj -> LSTM blocks (local temporal) -> Transformer Encoder (global attention)
       -> Attention Pooling -> Gamma dual-head (log_alpha, log_beta)
"""

import os
# Dual-GPU parallelism: do not restrict CUDA_VISIBLE_DEVICES, keep both GPUs visible
# To use specific cards only, set on the command line: CUDA_VISIBLE_DEVICES=0,1 python ...
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

import numpy as np
import torch
import torch.nn as nn
import math
import itertools
import argparse
import time
import pickle
import shutil
import pandas as pd
import optuna
from pathlib import Path

optuna.logging.set_verbosity(optuna.logging.WARNING)

from prob_shared import (
    seed_everything, load_data, precompute_sequences,
    train_probabilistic_model, calc_metrics, calc_prob_metrics,
    save_results, plot_all_rivers,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_individual_rivers(fr, fp, model_tag, out_dir, color='#2D9B56'):
    """
    Plot each river separately and save to a subfolder.
    fp can be a DataFrame (from save_results) or a dict.
    """
    plot_dir = Path(out_dir) / f'{model_tag.lower()}_timeseries'
    plot_dir.mkdir(parents=True, exist_ok=True)

    if fp is None or len(fp) == 0:
        print('plot_individual_rivers: no prediction data, skipping plotting')
        return

    # Build comid -> metrics mapping from fr
    metrics_map = {}
    if fr is not None and len(fr) > 0:
        for _, row in fr.iterrows():
            cid = int(row['comid'])
            metrics_map[cid] = row

    # Normalize to dict: {comid: DataFrame_subset}
    if isinstance(fp, pd.DataFrame):
        river_groups = {int(cid): grp for cid, grp in fp.groupby('comid')}
    elif isinstance(fp, dict):
        river_groups = fp
    else:
        print(f'plot_individual_rivers: unsupported fp type ({type(fp)})')
        return

    def to_np(x):
        if x is None:
            return None
        if hasattr(x, 'numpy'):
            return x.numpy().flatten()
        return np.asarray(x).flatten()

    count = 0
    for comid, pdata in river_groups.items():
        if isinstance(pdata, pd.DataFrame):
            sub = pdata[pdata['split'] == 'val'].sort_values('datetime') if 'split' in pdata.columns else pdata
            if len(sub) == 0:
                continue
            dates = pd.to_datetime(sub['datetime']).values if 'datetime' in sub.columns else None
            obs   = to_np(sub.get('grfr_obs', sub.get('obs', None)))
            pred  = to_np(sub.get('pred_mu', sub.get('pred', None)))
            swat  = to_np(sub.get('swat_raw', sub.get('swat', None)))
            lower = to_np(sub.get('lower_90', sub.get('lower', None)))
            upper = to_np(sub.get('upper_90', sub.get('upper', None)))
        elif isinstance(pdata, dict):
            dates = pdata.get('dates', pdata.get('date', None))
            obs   = to_np(pdata.get('obs', pdata.get('true_flow', pdata.get('grfr_obs', None))))
            pred  = to_np(pdata.get('pred', pdata.get('pred_mu', None)))
            swat  = to_np(pdata.get('swat', pdata.get('swat_flow', pdata.get('swat_raw', None))))
            lower = to_np(pdata.get('lower', pdata.get('pred_q05', pdata.get('lower_90', None))))
            upper = to_np(pdata.get('upper', pdata.get('pred_q95', pdata.get('upper_90', None))))
        else:
            continue

        if obs is None and pred is None:
            continue

        fig, ax = plt.subplots(figsize=(12, 3.5))
        x = np.arange(len(obs)) if dates is None else dates

        if obs is not None:
            ax.plot(x, obs, color='black', linewidth=0.8, label='Observed', alpha=0.85)
        if swat is not None:
            ax.plot(x, swat, color='#1f77b4', linewidth=0.6, label='SWAT+', alpha=0.6)
        if pred is not None:
            ax.plot(x, pred, color=color, linewidth=0.8, label=model_tag)
        if lower is not None and upper is not None:
            ax.fill_between(x, lower, upper, color=color, alpha=0.18, label='90% PI')

        title = f'COMID {comid}'
        row = metrics_map.get(int(comid))
        if row is not None:
            nse = row.get('NSE', np.nan)
            crps = row.get('CRPS', np.nan)
            picp = row.get('PICP_90', np.nan)
            title += f'  |  NSE={nse:.3f}  CRPS={crps:.2f}  PICP90={picp:.1%}'

        ax.set_title(title, fontsize=10)
        ax.set_ylabel('Flow')
        ax.legend(fontsize=8, loc='upper right', ncol=4)
        ax.grid(True, alpha=0.3)
        if dates is not None and hasattr(dates[0], 'astype'):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        plt.tight_layout()
        fig.savefig(plot_dir / f'{comid}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        count += 1

    print(f'Saved {count} per-river time series plots to: {plot_dir}')

# ============================================================
# Configuration
# ============================================================
SEED = 42

PARAM_SPACE = {
    'seq_len':       [24, 48, 96],
    'd_model':       [32, 64],
    'n_heads':       [4, 8],
    'e_layers':      (1, 3),
    'lstm_layers':   (1, 2),
    'dropout':       (0.05, 0.5),
    'lr':            (5e-5, 1e-3),
}
N_TRIALS = 40

FIXED = {
    'batch_size': 8192,
    'epochs': 500,
    'patience': 20,
    'd_ff_ratio': 2,
}


# ============================================================
# LSTM residual block
# ============================================================
class LSTMBlock(nn.Module):
    """
    LSTM residual block:
      LayerNorm -> LSTM -> Linear proj -> Dropout -> Residual

    LSTM is naturally suited to short-sequence temporal modeling:
      - Small parameter count, less prone to overfitting
      - Mature and stable gating mechanism
      - cuDNN-optimized, efficient on GPU
    """
    def __init__(self, d_model, num_layers=1, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: (B, S, d_model)"""
        residual = x
        h = self.norm(x)
        h, _ = self.lstm(h)       # (B, S, d_model)
        h = self.dropout(h)
        return residual + h


# ============================================================
# Model: LSTM-Transformer probabilistic forecasting
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class LSTMTransformerProb(nn.Module):
    """
    LSTM-Transformer probabilistic forecasting model.

    Architecture:
      1. Linear projection: n_features -> d_model
      2. LSTM blocks: extract local temporal features (gated memory, cuDNN acceleration)
      3. Positional encoding + Transformer Encoder: global attention modeling
      4. Learned Attention Pooling: sequence -> vector
      5. Gamma dual-head: log_alpha, log_beta

    Design rationale:
      - LSTM excels at local temporal dependencies in short sequences, parameter-efficient
      - Transformer excels at global dependencies and feature interactions
      - The two are complementary: LSTM first encodes temporal order, Transformer then does global modeling

    Input: (B, seq_len, n_features)
    Output: log_alpha (B,), log_beta (B,)
    """
    def __init__(self, n_features, seq_len, d_model, n_heads,
                 e_layers, lstm_layers, d_ff, dropout):
        super().__init__()

        # Feature projection
        self.input_proj = nn.Linear(n_features, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        # ---- LSTM front-end ----
        self.lstm_block = LSTMBlock(d_model, num_layers=lstm_layers,
                                     dropout=dropout)

        # ---- Transformer Encoder back-end ----
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len + 10)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                              num_layers=e_layers)

        # ---- Learned Attention Pooling ----
        self.attn_pool = nn.Linear(d_model, 1)

        # ---- Gamma dual-head ----
        self.fc_log_alpha = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(32, 1),
        )
        self.fc_log_beta = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (B, S, n_features)
        h = self.input_proj(x)              # (B, S, d_model)
        h = self.input_norm(h)

        # LSTM front-end
        h = self.lstm_block(h)              # (B, S, d_model)

        # Transformer back-end
        h = self.pos_enc(h)
        h = self.encoder(h)                 # (B, S, d_model)

        # Learned Attention Pooling
        scores = self.attn_pool(h).squeeze(-1)              # (B, S)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, S, 1)
        pooled = (h * weights).sum(dim=1)                   # (B, d_model)

        log_alpha = self.fc_log_alpha(pooled).squeeze(-1)
        log_beta = self.fc_log_beta(pooled).squeeze(-1)
        return log_alpha, log_beta


# ============================================================
# Single-river search
# ============================================================
def search_one_river(comid, cache, combos, keys, max_seq_len, fixed,
                     n_features, device_str, seed):
    seed_everything(seed)
    device = torch.device(device_str)

    print(f'\n=== COMID {comid} (Optuna {N_TRIALS} trials) [Device: {device_str}] ===', flush=True)

    best_score = None
    best_result = None
    best_params = None
    all_combo_results = []

    def objective(trial):
        nonlocal best_score, best_result, best_params

        search_params = {
            'seq_len':     trial.suggest_categorical('seq_len', PARAM_SPACE['seq_len']),
            'd_model':     trial.suggest_categorical('d_model', PARAM_SPACE['d_model']),
            'n_heads':     trial.suggest_categorical('n_heads', PARAM_SPACE['n_heads']),
            'e_layers':    trial.suggest_int('e_layers', *PARAM_SPACE['e_layers']),
            'lstm_layers': trial.suggest_int('lstm_layers', *PARAM_SPACE['lstm_layers']),
            'dropout':     trial.suggest_float('dropout', *PARAM_SPACE['dropout']),
            'lr':          trial.suggest_float('lr', *PARAM_SPACE['lr'], log=True),
        }

        if search_params['d_model'] % search_params['n_heads'] != 0:
            raise optuna.TrialPruned()

        params = {**search_params, **fixed}
        d_ff = params['d_model'] * params['d_ff_ratio']

        model = LSTMTransformerProb(
            n_features, params['seq_len'], params['d_model'], params['n_heads'],
            params['e_layers'], params['lstm_layers'], d_ff, params['dropout'],
        )

        try:
            result = train_probabilistic_model(
                model, cache, params['seq_len'], max_seq_len, params, device, seed, trial=trial
            )
        except optuna.TrialPruned:
            del model
            torch.cuda.empty_cache()
            raise
        except RuntimeError as e:
            if 'out of memory' in str(e):
                del model
                torch.cuda.empty_cache()
                raise optuna.TrialPruned()
            raise

        if result is None:
            del model
            raise optuna.TrialPruned()

        m = result['metrics_point']
        mp_ = result['metrics_prob']
        m_va = result['metrics_val']
        mp_va = result['metrics_val_prob']
        m_tr = result.get('metrics_train', {})

        all_combo_results.append({
            'comid': comid, **search_params,
            **m, **mp_,
            'val_NSE': m_va.get('NSE', np.nan),
            'val_CRPS': mp_va.get('CRPS', np.nan),
            'best_epoch': result['best_epoch'],
        })

        param_str = ' '.join(f'{k}={v}' for k, v in search_params.items())
        val_crps = mp_va.get('CRPS', 999)
        print(f'  [T{trial.number+1}/{N_TRIALS}] '
              f'ep={result["best_epoch"]}/{result["total_epochs"]} '
              f'trNSE={m_tr.get("NSE",0):.3f} '
              f'vaNSE={m_va.get("NSE",0):.3f} '
              f'teNSE={m["NSE"]:.3f} '
              f'vaCRPS={val_crps:.2f} '
              f'| {param_str}', flush=True)

        val_nse = m_va.get('NSE', np.nan)
        if not np.isnan(val_nse) and val_nse > -10:
            score = (1 if val_nse > 0 else 0, -val_crps, val_nse)
            if best_score is None or score > best_score:
                best_score = score
                best_params = search_params.copy()
                result['best_params'] = best_params
                best_result = result

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return val_crps

    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    if best_params and best_result:
        bm = best_result['metrics_point']
        bp = best_result['metrics_prob']
        btr = best_result.get('metrics_train', {})
        bva = {}
        if best_result['splits_pred'].get('val'):
            vp = best_result['splits_pred']['val']
            bva = calc_metrics(vp['true_flow'], vp['pred_mu'])

        print(f'  >> BEST: trNSE={btr.get("NSE",0):.3f} '
              f'vaNSE={bva.get("NSE",0):.3f} '
              f'teNSE={bm["NSE"]:.3f} CRPS={bp.get("CRPS",0):.2f} '
              f'PICP90={bp.get("PICP_90",0):.1%} | '
              f'{" | ".join(f"{k}={v}" for k,v in best_params.items())}',
              flush=True)

    if best_result:
        if 'state_dict' in best_result:
            best_result['state_dict'] = {
                k: v.cpu() if hasattr(v, 'cpu') else v
                for k, v in best_result['state_dict'].items()
            }
        if 'splits_pred' in best_result:
            for split_name, sp in best_result['splits_pred'].items():
                if isinstance(sp, dict):
                    for k, v in sp.items():
                        if hasattr(v, 'cpu'):
                            sp[k] = v.cpu()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc; gc.collect()

    return {
        'comid': comid, 'best_result': best_result,
        'best_params': best_params, 'all_combo_results': all_combo_results,
    }


# ============================================================
# Worker wrapper: load cache from disk to avoid pickling large data across processes
# ============================================================
def search_one_river_from_disk(comid, cache_dir, combos, keys,
                                max_seq_len, fixed, n_features,
                                device_str, seed):
    """
    Load this river's cache from disk, then call search_one_river.
    This way the main process only needs to pickle a path string when
    submitting tasks, instead of stuffing the entire river_cache into the
    ProcessPoolExecutor queue.
    """
    cache_path = Path(cache_dir) / f'{comid}.pkl'
    # Clear residual GPU memory when a new process starts
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)
    return search_one_river(
        comid, cache, combos, keys, max_seq_len, fixed,
        n_features, device_str, seed,
    )


# ============================================================
# Main program
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--region', type=str, required=True,
                        help='Region folder path, e.g. /datasets/swat_global/brazil/brazil_norte')
    parser.add_argument('--workers', type=int, default=2,
                        help='Number of parallel threads (can exceed GPU count, e.g. single card: --gpus 0 --workers 3)')
    parser.add_argument('--gpus', type=str, default='0,1',
                        help='GPU ids to use, comma-separated, e.g. 0 or 0,1')
    args = parser.parse_args()

    # Build paths from the region folder
    region_dir = Path(args.region)
    DATASET_FILE = str(region_dir / 'datasets' / 'ml_dataset_hourly.parquet')
    META_FILE    = str(region_dir / 'datasets' / 'ml_dataset_hourly_meta.json')
    OUT_DIR      = region_dir / 'LSTM_Transformer_prob'

    MODEL_TAG = 'LSTM_Transformer_prob'

    seed_everything(SEED)

    # ---- Multi-GPU device list ----
    gpu_ids = [int(g) for g in args.gpus.split(',')]
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        gpu_ids = [g for g in gpu_ids if g < n_gpus]
        if not gpu_ids:
            gpu_ids = list(range(n_gpus))
        device_list = [f'cuda:{g}' for g in gpu_ids]
        print(f'Available GPUs: {n_gpus}, Using: {gpu_ids}')
    else:
        device_list = ['cpu']
        print('No CUDA available, using CPU')

    n_workers = args.workers
    print(f'Region: {region_dir}')
    print(f'Devices: {device_list}, Workers: {n_workers}')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / 'models').mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    df, feature_cols, target_col, valid_comids = load_data(DATASET_FILE, META_FILE)
    n_features = len(feature_cols)
    print(f'Rivers: {len(valid_comids)}, Features: {n_features}')

    max_seq_len = max(PARAM_SPACE['seq_len'])

    combos = None
    keys = None

    cache_dir = OUT_DIR / '_river_cache_tmp'
    cache_dir.mkdir(exist_ok=True)
    models_dir = OUT_DIR / 'models'

    # ---- Resume: skip rivers that already have model weights ----
    existing = set()
    for c in valid_comids:
        if (models_dir / f'lstm_transformer_prob_{c}.pt').exists():
            existing.add(c)
    valid_river_list = [c for c in valid_comids if c not in existing]
    if existing:
        print(f'Resuming: skipping {len(existing)} rivers with existing models', flush=True)

    # ---- Streaming precompute: compute windows per river and flush to disk immediately, peak memory ≈ DataFrame + 1 river ----
    print(f'Precomputing (max_seq={max_seq_len}), streaming to disk...')
    cached_count = 0
    for comid in valid_river_list:
        pkl_path = cache_dir / f'{comid}.pkl'
        if pkl_path.exists():
            cached_count += 1
            continue
        one = precompute_sequences(df, feature_cols, target_col, [comid], max_seq_len)
        if comid in one:
            with open(pkl_path, 'wb') as f:
                pickle.dump(one[comid], f, protocol=pickle.HIGHEST_PROTOCOL)
            cached_count += 1
        del one
    del df
    print(f'Done: {cached_count} rivers cached to disk, {time.time()-t0:.1f}s')
    print(f'Optuna TPE search: {N_TRIALS} trials/river, '
          f'Total: {N_TRIALS*len(valid_river_list)}\n')

    all_results = {}
    all_combo_details = []
    import threading
    _lock = threading.Lock()

    def _save_one_model(comid, out):
        """Write weights to disk as soon as a river finishes, to avoid losing results on a later crash."""
        if out and out.get('best_result') and out['best_result'].get('state_dict'):
            torch.save(
                out['best_result']['state_dict'],
                models_dir / f'lstm_transformer_prob_{comid}.pt',
            )

    def _process_one_river(args_tuple):
        """Full processing flow for a single river (thread-safe)"""
        i, comid, dev = args_tuple
        try:
            out = search_one_river_from_disk(
                comid, str(cache_dir), combos, keys,
                max_seq_len, FIXED, n_features, dev, SEED,
            )
            _save_one_model(comid, out)
            if out and out.get('best_result'):
                out['best_result'].pop('state_dict', None)
            with _lock:
                all_results[comid] = out
                all_combo_details.extend(out['all_combo_results'])
        except Exception as e:
            print(f'COMID {comid} ERROR: {e}', flush=True)

    if n_workers <= 1:
        # ---- Serial ----
        for i, comid in enumerate(valid_river_list):
            dev = device_list[i % len(device_list)]
            _process_one_river((i, comid, dev))
    else:
        # ---- Multi-threaded parallelism (ThreadPoolExecutor) ----
        # Use threads rather than processes: CUDA operations release the GIL, so
        # threads can truly use the GPU in parallel, while avoiding the various
        # crashes of ProcessPoolExecutor + spawn + CUDA.
        from concurrent.futures import ThreadPoolExecutor, as_completed
        tasks = []
        for i, comid in enumerate(valid_river_list):
            dev = device_list[i % len(device_list)]
            tasks.append((i, comid, dev))

        print(f'Using {n_workers} threads on devices {device_list}', flush=True)
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_process_one_river, t): t[1]
                       for t in tasks}
            for fut in as_completed(futures):
                comid = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    print(f'COMID {comid} THREAD ERROR: {e}', flush=True)

    # Clean up the temporary cache directory
    try:
        shutil.rmtree(cache_dir)
    except Exception as e:
        print(f'Warning: failed to remove {cache_dir}: {e}', flush=True)

    print(f'\nSearch done in {(time.time()-t0)/60:.1f} min')
    print(f'Succeeded: {sum(1 for o in all_results.values() if o["best_result"])} '
          f'/ {len(valid_comids)} rivers')

    results_for_save = {
        c: out['best_result']
        for c, out in all_results.items() if out['best_result']
    }
    # Note: model weights were saved incrementally when each river finished, so no re-saving here

    # ---- If everything was skipped (resume), try loading the existing CSV instead of overwriting ----
    import pandas as pd
    metrics_csv = Path(OUT_DIR) / f'{MODEL_TAG.lower()}_metrics.csv'
    preds_csv   = Path(OUT_DIR) / f'{MODEL_TAG.lower()}_predictions.csv'

    if len(results_for_save) == 0 and metrics_csv.exists():
        print(f'\nAll rivers skipped, loading existing results: {metrics_csv}')
        fr = pd.read_csv(metrics_csv)
        if preds_csv.exists():
            fp_df = pd.read_csv(preds_csv)
            # Rebuild fp dict: {comid: {col: array}}
            fp = {}
            if 'comid' in fp_df.columns:
                for cid, grp in fp_df.groupby('comid'):
                    fp[cid] = {col: grp[col].values for col in grp.columns if col != 'comid'}
            else:
                fp = {}
        else:
            fp = {}
    else:
        fr, fp = save_results(results_for_save, all_combo_details, OUT_DIR, MODEL_TAG)

    if len(fr) > 0:
        print(f'\n=== LSTM-Transformer Probabilistic Results ===')
        print(f'{"COMID":<12s} {"SWAT+":>8s} {"trNSE":>8s} {"NSE":>8s} {"KGE":>8s} '
              f'{"PICP90":>8s} {"PINAW90":>9s} {"CRPS":>8s}')
        print('-' * 80)
        for _, row in fr.iterrows():
            print(f'{int(row["comid"]):<12d} {row["swat_NSE"]:>8.3f} '
                  f'{row.get("train_NSE",np.nan):>8.3f} '
                  f'{row["NSE"]:>8.3f} {row["KGE"]:>8.3f} '
                  f'{row["PICP_90"]:>8.1%} '
                  f'{row["PINAW_90"]:>9.3f} {row["CRPS"]:>8.2f}')

        print(f'\nMedian NSE:       SWAT+={fr["swat_NSE"].median():.3f}  '
              f'LSTM-Trans={fr["NSE"].median():.3f}')
        print(f'Median train NSE: {fr["train_NSE"].median():.3f}')
        print(f'Median PICP90:    {fr["PICP_90"].median():.1%}')
        print(f'Median CRPS:      {fr["CRPS"].median():.2f}')

        # ---- Plot each river separately, save to a subfolder ----
        plot_individual_rivers(fr, fp, MODEL_TAG, OUT_DIR, color='#2D9B56')

        # Also try to plot the combined figure (may fail due to pixel limits when there are too many rivers, just ignore)
        try:
            plot_all_rivers(fr, fp, MODEL_TAG, OUT_DIR, color='#2D9B56')
        except (ValueError, Exception) as e:
            print(f'Combined figure skipped (pixel limit exceeded): {e}')
    else:
        print('\nNo new results (possibly all skipped or all failed)')

    print(f'\nTotal: {(time.time()-t0)/60:.1f} min')
