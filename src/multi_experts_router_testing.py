"""
multi_experts_router_testing.py
─────────────────────────────────────────────────────────────────────────────
對多個 random seed 的 DFLMVOModel 做 inference，
比較三種 ensemble 策略：

  Experts                      : 各 seed 各自持倉計算報酬後取平均（非可執行單一組合）
  Ensemble (equal weight)      : 每天對所有 seed 的 w_hat 取等權平均，直接執行
  Router (z-score)             : 前 warmup_days 天使用等權平均，
                                 之後每 review_freq 天切換至累積 z-score 最高的 seed

performance_summary.csv 的列順序為：各 seed → Experts → Ensemble → Router({score_method})。
跨年度模式另外輸出 per_year_performance_summary.csv：把整段連續序列依日曆年切開的逐年指標。
"""

import sys
import json
import time
import argparse
import itertools
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tqdm import tqdm

from dataloader import create_dataloaders, prepare_data, load_price_data
from utils import load_risk_free_rate, load_price_index_returns
from backtest_utils import (
    Logger, set_seed,
    _resolve_infer_window, _load_model, _inference_one_seed,
    _build_returns_from_w, _build_returns_for_seed, _infer_multi_year,
    _save_subdir,
    _metrics, _mean_std_metrics, _print_table, _per_year_metrics,
    plot_cumulative, plot_seed_band,
)
from financial_baseline import (
    equal_weight_portfolio,
    auto_reweight_portfolio,
    single_asset_portfolio,
)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _compute_daily_cross_corr(
    seed_preds:     Dict[int, np.ndarray],
    shared_mu_star: np.ndarray,
) -> np.ndarray:
    """
    計算每個 seed 每天對 mu_star 的截面 Pearson 相關係數。

    可用於：
      IC     : seed_preds = seed_mu_hats（mu_hat vs mu_star）
      w_corr : seed_preds = seed_w_hats （w_hat  vs mu_star）

    Parameters
    ----------
    seed_preds     : Dict[seed → (T, A)]
    shared_mu_star : (T, A)

    Returns
    -------
    corr_arr : (N, T)，N = len(seed_preds)，順序與 sorted(seeds) 一致
    """
    seeds   = sorted(seed_preds.keys())
    T       = shared_mu_star.shape[0]
    s_mean  = shared_mu_star.mean(axis=1, keepdims=True)
    s_std   = shared_mu_star.std(axis=1, keepdims=True)
    s_norm  = np.where(s_std > 1e-8, (shared_mu_star - s_mean) / s_std, 0.0)
    s_valid = s_std.squeeze() > 1e-8

    corr_arr = np.zeros((len(seeds), T))
    for i, seed in enumerate(seeds):
        pred   = seed_preds[seed]                          # (T, A)
        h_mean = pred.mean(axis=1, keepdims=True)
        h_std  = pred.std(axis=1, keepdims=True)
        h_norm = np.where(h_std > 1e-8, (pred - h_mean) / h_std, 0.0)
        corr   = (h_norm * s_norm).mean(axis=1)            # (T,)
        valid  = (h_std.squeeze() > 1e-8) & s_valid
        corr[~valid]  = 0.0
        corr_arr[i]   = corr
    return corr_arr


# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────

_CSV_PCT_COLUMNS = {
    'total_ret': 'Total Return(%)',
    'ann_ret':   'Annualized Return(%)',
    'ann_vol':   'Annualized Volatility(%)',
    'max_dd':    'Max Drawdown(%)',
}


def _scale_metrics_for_csv(rows: List[Dict]) -> List[Dict]:
    """
    儲存到 performance_summary.csv 前的顯示轉換：
    total_ret / ann_ret / ann_vol 轉成百分比數值（*100），
    max_dd 轉成正值百分比（abs(*100)），sharpe / sortino 取小數點後兩位，
    百分比欄位改名為 'Total Return(%)' 等（見 _CSV_PCT_COLUMNS）。
    若欄位值為 (mean, std) tuple（例如 Experts row），格式化成 "mean ± std" 字串
    （max_dd 的 mean 取絕對值，std 維持原值，因為標準差本身恆為正）。
    只影響存檔用的副本，console 顯示與其他運算仍用原始小數。
    """
    def _fmt_pair(v, scale=1.0, absmean=False, nd=2):
        m, s = v
        m = abs(m) if absmean else m
        return f"{m * scale:.{nd}f} ± {s * scale:.{nd}f}"

    scaled = []
    for row in rows:
        r = dict(row)
        for k in ('total_ret', 'ann_ret', 'ann_vol'):
            v = r[k]
            r[k] = _fmt_pair(v, scale=100) if isinstance(v, tuple) else round(v * 100, 2)
        v = r['max_dd']
        r['max_dd'] = (_fmt_pair(v, scale=100, absmean=True) if isinstance(v, tuple)
                       else round(abs(v) * 100, 2))
        for k in ('sharpe', 'sortino'):
            v = r[k]
            r[k] = _fmt_pair(v) if isinstance(v, tuple) else round(v, 2)
        scaled.append({_CSV_PCT_COLUMNS.get(k, k): v for k, v in r.items()})
    return scaled


# ─────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────

def report_high_divergence_days(
    seed_port_rets:      Dict[str, pd.Series],
    div_threshold_sigma: float = 1.5,
    div_csv_path:        Optional[str] = None,
) -> None:
    """找出 cross-seed 報酬標準差異常偏高的日子，印出並存成 CSV（不畫圖）。"""
    seed_df    = pd.DataFrame(seed_port_rets).dropna()
    daily_mean = seed_df.mean(axis=1)
    daily_std  = seed_df.std(axis=1)

    std_threshold  = daily_std.mean() + div_threshold_sigma * daily_std.std()
    high_div_dates = daily_std[daily_std > std_threshold].index

    print(f"\n  [High divergence days] threshold={std_threshold:.5f}, "
          f"n={len(high_div_dates)} days")
    for d in high_div_dates:
        print(f"    {d.date()}  std={daily_std[d]:.5f}  "
              f"mean_ret={daily_mean[d]:.5f}")

    if div_csv_path and len(high_div_dates) > 0:
        div_df = pd.DataFrame({
            'date':            high_div_dates,
            'cross_seed_std':  daily_std[high_div_dates].values,
            'cross_seed_mean': daily_mean[high_div_dates].values,
        }).sort_values('cross_seed_std', ascending=False)
        div_df.to_csv(div_csv_path, index=False)
        print(f"  High divergence days saved → {div_csv_path}")


def plot_all_seeds(
    seed_port_rets:      Dict[str, pd.Series],
    title_suffix:        str,
    save_path:           str,
    figsize:             tuple = (16, 6),
    line_alpha:          float = 0.80,
    linewidth:           float = 1.4,
    dpi:                 int   = 150,
) -> None:
    """所有 seed 的累積報酬曲線畫在同一張圖。"""
    if not seed_port_rets:
        print(f"[plot_all_seeds] seed_port_rets is empty, skipping {save_path}")
        return

    seed_df = pd.DataFrame(seed_port_rets).dropna()
    n_seeds = len(seed_df.columns)
    cmap    = plt.cm.tab20

    fig, ax = plt.subplots(figsize=figsize)
    for i, col in enumerate(seed_df.columns):
        cum = (1 + seed_df[col]).cumprod()
        ax.plot(cum, label=col, color=cmap(i / max(n_seeds, 1)),
                linewidth=linewidth, alpha=line_alpha)
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_title(f'Cumulative Returns — All Seeds | {title_suffix}')
    ax.set_ylabel('Portfolio Value (start = 1)')
    ax.legend(fontsize=6, ncol=5, loc='upper left', framealpha=0.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved: {save_path}")


# ─────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────

def _zscore_router(
    seed_w_hats:      Dict[int, np.ndarray],
    seed_rets:        np.ndarray,
    date_labels:      pd.DatetimeIndex,
    warmup_days:      int                   = 20,
    review_freq:      int                   = 20,
    top_k:            int                   = 1,
    bottom_k:         int                   = 0,
    rolling_window:   int                   = 60,
    score_method:     str                   = 'sharpe',
    combo_beta:       float                 = 0.94,
    combo_alpha:      float                 = 0.5,
    seed_ic_arr:      Optional[np.ndarray]  = None,
    seed_wc_arr:      Optional[np.ndarray]  = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    前 warmup_days 天使用 ensemble weight mean；
    之後每 review_freq 天根據 rolling score 選出 top_k 個 seed，
    等權平均其 w_hat 作為當期權重。

    rolling_window=0 → expanding window（從第一天累積到當天）。

    score_method
    ────────────
    'sharpe'        : rolling window 內的 mean(ret) / std(ret)
    'cumret'        : rolling window 內的複利累積報酬 prod(1+ret) - 1
    'cumret_mucorr' : R_m（cumret）與 C_m（衰減加權 mu_corr，decay=combo_beta）
                      各自做 cross-seed z-score 後依 combo_alpha 加權相加：
                      Score_m = combo_alpha·z(R_m) + (1-combo_alpha)·z(C_m)
                      combo_alpha=1 等同純 cumret，combo_alpha=0 等同純 mu_corr（無衰減版）
    """
    from collections import deque

    _VALID = ('sharpe', 'cumret', 'mu_corr', 'ic_ir', 'w_corr', 'cumret_mucorr')
    if score_method not in _VALID:
        raise ValueError(f"score_method must be one of {_VALID}, got {score_method!r}")
    if score_method in ('mu_corr', 'ic_ir', 'cumret_mucorr') and seed_ic_arr is None:
        raise ValueError(f"score_method='{score_method}' requires seed_ic_arr")
    if score_method == 'w_corr' and seed_wc_arr is None:
        raise ValueError("score_method='w_corr' requires seed_wc_arr")

    seeds  = sorted(seed_w_hats.keys())
    N      = len(seeds)
    T, A   = seed_w_hats[seeds[0]].shape
    k      = max(1, min(top_k, N))
    bk     = max(0, min(bottom_k, N - 1))   # 最多剔除 N-1 個，至少保留 1 個

    # Ensemble（warmup 期用）
    w_ens    = np.stack(list(seed_w_hats.values()), axis=0).mean(axis=0)
    row_sums = w_ens.sum(axis=1, keepdims=True)
    w_ens    = w_ens / np.where(row_sums > 0, row_sums, 1)

    # rolling_window=0 → expanding（deque 無 maxlen）
    maxlen  = None if rolling_window == 0 else rolling_window
    bufs    = [deque(maxlen=maxlen) for _ in range(N)]
    # IC buffer：mu_corr / ic_ir 使用
    ic_bufs = ([deque(maxlen=maxlen) for _ in range(N)]
               if seed_ic_arr is not None else None)
    # w_corr buffer：w_hat vs mu_star 截面相關
    wc_bufs = ([deque(maxlen=maxlen) for _ in range(N)]
               if seed_wc_arr is not None else None)

    def _scores() -> np.ndarray:
        out = np.zeros(N)

        # mu_corr / ic_ir：使用 IC buffer（mu_hat vs mu_star）
        if score_method in ('mu_corr', 'ic_ir'):
            for i in range(N):
                arr = np.array(ic_bufs[i])
                if len(arr) < 2:
                    continue
                if score_method == 'mu_corr':
                    out[i] = arr.mean()
                else:  # ic_ir：mean(IC) / std(IC)，獎勵穩定預測品質
                    s = arr.std()
                    out[i] = arr.mean() / s if s > 1e-8 else 0.0
            return out

        # cumret_mucorr：R_m（累積報酬）與 C_m（衰減加權 cosine 相似度）各自做
        # cross-seed z-score 後依 combo_alpha 加權相加，兼顧「近期績效」與「預測品質」
        if score_method == 'cumret_mucorr':
            R = np.zeros(N)
            C = np.zeros(N)
            for i in range(N):
                r_arr = np.array(bufs[i])
                if len(r_arr) >= 2:
                    R[i] = np.prod(1.0 + r_arr) - 1.0
                c_arr = np.array(ic_bufs[i])
                if len(c_arr) >= 2:
                    n = len(c_arr)
                    w = combo_beta ** (n - 1 - np.arange(n))   # 越新（越靠近 t）權重越高
                    C[i] = float(np.sum(w * c_arr) / np.sum(w))

            def _z(x: np.ndarray) -> np.ndarray:
                s = x.std()
                return (x - x.mean()) / s if s > 1e-8 else np.zeros_like(x)

            return combo_alpha * _z(R) + (1.0 - combo_alpha) * _z(C)

        # w_corr：使用 w_corr buffer（w_hat vs mu_star），衡量持倉與真實報酬的截面對齊程度
        if score_method == 'w_corr':
            for i in range(N):
                arr = np.array(wc_bufs[i])
                if len(arr) >= 2:
                    out[i] = arr.mean()
            return out

        for i in range(N):
            arr = np.array(bufs[i])
            if len(arr) < 2:
                continue
            if score_method == 'sharpe':
                s = arr.std()
                out[i] = arr.mean() / s if s > 0 else 0.0
            else:  # cumret
                out[i] = np.prod(1.0 + arr) - 1.0
        return out

    selected_idxs = [0]
    w_final       = np.zeros((T, A))
    records: list = []

    for t in range(T):
        is_review = (t == warmup_days) or (
            t > warmup_days and (t - warmup_days) % review_freq == 0
        )
        if is_review:
            raw_scores = _scores()

            if bk > 0:
                # 剔除分數最低的 bk 個，其餘等權平均
                new_idxs = list(np.argsort(raw_scores)[bk:])
            else:
                new_idxs = list(np.argsort(raw_scores)[::-1][:k])

            selected_idxs = new_idxs

        # 分配權重
        if t < warmup_days:
            w_final[t] = w_ens[t]
            phase = 'warmup'
        else:
            w_target = np.stack(
                [seed_w_hats[seeds[i]][t] for i in selected_idxs], axis=0
            ).mean(axis=0)
            row_s = w_target.sum()
            w_final[t] = w_target / row_s if row_s > 1e-8 else w_target
            phase = 'selected'

        day_rets = seed_rets[:, t]
        for i in range(N):
            bufs[i].append(float(day_rets[i]))
        if ic_bufs is not None:
            for i in range(N):
                ic_bufs[i].append(float(seed_ic_arr[i, t]))
        if wc_bufs is not None:
            for i in range(N):
                wc_bufs[i].append(float(seed_wc_arr[i, t]))

        cur_scores = _scores()
        selected_seeds_str = (
            ','.join(str(seeds[i]) for i in selected_idxs)
            if t >= warmup_days else ''
        )
        records.append({
            'date':           date_labels[t],
            'phase':          phase,
            'is_review':      is_review,
            'selected_seed':  seeds[selected_idxs[0]] if t >= warmup_days else -1,
            'selected_seeds': selected_seeds_str,
            **{f'score_seed{s}': round(float(cur_scores[i]), 4)
               for i, s in enumerate(seeds)},
        })

    return w_final, pd.DataFrame(records)


def plot_zscore_router_analysis(
    record_df:    pd.DataFrame,
    port_zscore:  pd.Series,
    port_equal:   pd.Series,
    seeds:        List[int],
    title_suffix: str,
    save_path:    str,
    dpi:          int = 150,
) -> None:
    """三格圖：rolling Sharpe score / 選中的 seed / 累積報酬比較。"""
    fig, axes = plt.subplots(3, 1, figsize=(16, 14),
                             gridspec_kw={'height_ratios': [2.5, 1, 2]})
    fig.subplots_adjust(hspace=0.40)
    dates = pd.to_datetime(record_df['date'])

    # ── ① 各 seed rolling Sharpe score ─────────────────────────
    ax = axes[0]
    cmap = plt.cm.tab20
    for i, s in enumerate(seeds):
        col = f'score_seed{s}'
        if col in record_df.columns:
            ax.plot(dates, record_df[col], linewidth=1.0,
                    color=cmap(i / max(len(seeds), 1)), label=f'seed{s}')
    # review 點標記
    review_dates = dates[record_df['is_review']]
    for rd in review_dates:
        ax.axvline(rd, color='gray', linewidth=0.7, linestyle='--', alpha=0.6)
    ax.axhline(0, color='black', linewidth=0.5, linestyle=':')
    ax.set_title(f'Rolling Sharpe Score per Seed — {title_suffix}')
    ax.set_ylabel('Rolling Sharpe')
    ax.legend(fontsize=6, ncol=5, loc='upper left')
    ax.grid(True, alpha=0.3)

    # ── ② 每天選中的 seed（色塊，top-k 堆疊）──────────────────
    ax = axes[1]
    seed_color = {s: cmap(i / max(len(seeds), 1)) for i, s in enumerate(seeds)}
    bar_width  = pd.Timedelta(days=1)
    selected_seeds_col = record_df['selected_seeds'].values
    for t in range(len(dates)):
        raw_str = selected_seeds_col[t]
        d       = dates.iloc[t]
        if not isinstance(raw_str, str) or raw_str == '':
            # warmup
            ax.bar(d, 1.0, bottom=0, color='#DDDDDD',
                   width=bar_width, alpha=0.85)
        else:
            sel = [int(x) for x in raw_str.split(',')]
            seg = 1.0 / len(sel)
            for j, s in enumerate(sel):
                ax.bar(d, seg, bottom=j * seg,
                       color=seed_color.get(s, '#AAAAAA'),
                       width=bar_width, alpha=0.85)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    # 圖例
    patches = [Patch(color=seed_color[s], label=f'seed{s}') for s in seeds]
    patches.append(Patch(color='#DDDDDD', label='warmup (ensemble)'))
    ax.legend(handles=patches, fontsize=6, ncol=6, loc='upper right')
    for rd in review_dates:
        ax.axvline(rd, color='gray', linewidth=0.7, linestyle='--', alpha=0.6)
    ax.set_title('Selected Seeds per Day  (dashed = review point, stacked = top-k)')
    ax.grid(False)

    # ── ③ 累積報酬比較 ──────────────────────────────────────────
    ax = axes[2]
    common_idx  = port_zscore.index.intersection(port_equal.index)
    cum_zscore  = (1 + port_zscore.loc[common_idx].dropna()).cumprod()
    cum_equal   = (1 + port_equal.loc[common_idx].dropna()).cumprod()
    ax.plot(cum_zscore, color='#E8A020', linewidth=2.0,
            label='Router', zorder=5)
    ax.plot(cum_equal,  color='steelblue', linewidth=1.5,
            linestyle='--', label='Ensemble', zorder=4)
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.7)
    for rd in review_dates:
        if rd in common_idx:
            ax.axvline(rd, color='gray', linewidth=0.7, linestyle='--', alpha=0.5)
    ax.set_title('Cumulative Return: Router vs Equal Weight')
    ax.set_ylabel('Portfolio Value (start = 1)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Router Analysis — {title_suffix}', fontsize=12)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


# ─────────────────────────────────────────────────────────────
# Router signal arrays  (main() 與 grid search 共用)
# ─────────────────────────────────────────────────────────────

def _build_router_signal_arrays(
    seed_w_hats:        Dict[int, np.ndarray],
    seed_mu_hats_raw:   Dict[int, np.ndarray],
    seed_port_rets:     Dict[str, pd.Series],
    shared_mu_star:     np.ndarray,
    shared_date_labels: pd.DatetimeIndex,
) -> Tuple[List[int], np.ndarray, np.ndarray, np.ndarray]:
    """把 per-seed 的每日報酬 / mu_hat / w_hat 整理成 _zscore_router 需要的 (N, T) 陣列。

    Returns
    -------
    seeds_sorted  : sorted(seed_w_hats.keys())
    seed_rets_arr : (N, T)  每個 seed 的每日淨報酬
    seed_ic_arr   : (N, T)  mu_hat vs mu_star 的每日截面相關（IC）
    seed_wc_arr   : (N, T)  w_hat  vs mu_star 的每日截面相關
    """
    seeds_sorted  = sorted(seed_w_hats.keys())
    seed_rets_arr = np.stack(
        [seed_port_rets[f'seed{s}'].reindex(shared_date_labels).fillna(0).values
         for s in seeds_sorted], axis=0
    )
    _ic_seeds   = sorted(seed_mu_hats_raw.keys())
    seed_ic_arr = _compute_daily_cross_corr(seed_mu_hats_raw, shared_mu_star)
    if _ic_seeds != seeds_sorted:
        seed_ic_arr = np.stack(
            [seed_ic_arr[_ic_seeds.index(s)] for s in seeds_sorted], axis=0)
    seed_wc_arr = _compute_daily_cross_corr(seed_w_hats, shared_mu_star)
    return seeds_sorted, seed_rets_arr, seed_ic_arr, seed_wc_arr


# ─────────────────────────────────────────────────────────────
# Grid search on validation set  (no test-set leakage)
# ─────────────────────────────────────────────────────────────

_GS_PARAM_KEYS = ('warmup_days', 'review_freq', 'top_k', 'bottom_k',
                  'rolling_window', 'score_method', 'combo_beta', 'combo_alpha')
_GS_INT_KEYS   = ('warmup_days', 'review_freq', 'top_k', 'bottom_k', 'rolling_window')
_GS_METRIC_KEYS = ('total_ret', 'ann_ret', 'ann_vol', 'sharpe', 'sortino', 'max_dd')


def _infer_val_segment(
    args, price_df, risk_free_returns, device, ckpt_year: int,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray],
           np.ndarray, pd.DatetimeIndex, List[str]]:
    """對 testYear{ckpt_year} 的 checkpoint，在其「驗證窗（= ckpt_year-1 整年）」上做 inference。

    驗證窗來自 prepare_data 的 val_start / val_end，與訓練時的 val set 完全一致，
    不觸及任何測試年資料。
    """
    returns_df, si = prepare_data(
        price_df, lookback_days=args.lookback_days, test_year=ckpt_year)
    v0, v1 = si['val_start'], si['val_end']
    infer_si = dict(si)
    infer_si['test_start'], infer_si['test_end'] = v0, v1
    _, _, loader = create_dataloaders(
        returns_df, infer_si, lookback_days=args.lookback_days, batch_size=1)

    asset_names = [n.replace('.csv', '') for n in si['asset_names']]
    vs, ve = returns_df.index[v0].date(), returns_df.index[v1].date()
    print(f"\n─── [grid-search val] testYear{ckpt_year} checkpoints "
          f"→ val window [{vs} → {ve}] ───")

    seg_w, seg_muh = {}, {}
    seg_mu = seg_dt = None
    for seed in args.seeds:
        ckpt = (Path(args.base_checkpoint_dir) / args.model_type
                / f"testYear{ckpt_year}" / f"seed{seed}" / args.checkpoint_type)
        if not ckpt.exists():
            print(f"  ✗ seed={seed}: 無 testYear{ckpt_year} checkpoint，跳過")
            continue
        model = _load_model(ckpt, device)
        raw   = _inference_one_seed(model, loader, device)
        seg_w[seed], seg_muh[seed] = raw['w_hat'], raw['mu_hat']
        if seg_mu is None:
            seg_mu = raw['mu_star']
            seg_dt = returns_df.index[raw['dates']]
    return seg_w, seg_muh, seg_mu, seg_dt, asset_names


def _expand_grid(args) -> List[Dict]:
    """把 --gs_* 清單展開成去重後的 router 參數組合清單。

    未指定的維度沿用對應的單一 --<name> 參數。與 score_method 無關的
    combo_beta / combo_alpha（非 cumret_mucorr 時）、被 bottom_k 蓋過的 top_k
    會收斂成單一值以避免重複組合。
    """
    grids = {
        'warmup_days':    args.gs_warmup_days    or [args.warmup_days],
        'review_freq':    args.gs_review_freq    or [args.review_freq],
        'top_k':          args.gs_top_k          or [args.top_k],
        'bottom_k':       args.gs_bottom_k       or [args.bottom_k],
        'rolling_window': args.gs_rolling_window or [args.rolling_window],
        'score_method':   args.gs_score_method   or [args.score_method],
        'combo_beta':     args.gs_combo_beta     or [args.combo_beta],
        'combo_alpha':    args.gs_combo_alpha    or [args.combo_alpha],
    }
    keys = list(_GS_PARAM_KEYS)
    seen, combos = set(), []
    for vals in itertools.product(*(grids[k] for k in keys)):
        p = dict(zip(keys, vals))
        if p['score_method'] != 'cumret_mucorr':
            p['combo_beta']  = grids['combo_beta'][0]
            p['combo_alpha'] = grids['combo_alpha'][0]
        if p['bottom_k'] > 0:                      # _zscore_router 於 bottom_k>0 時忽略 top_k
            p['top_k'] = grids['top_k'][0]
        sig = tuple(p[k] for k in keys)
        if sig in seen:
            continue
        seen.add(sig)
        combos.append(p)
    return combos


def _grid_search_on_val(args, price_df, risk_free_returns, device, out_dir):
    """在驗證集上跑 router 超參數 grid search，回傳最佳參數 dict。

    驗證窗一律取「測試年的前一年」；多年度模式下只取『前一年不在測試年清單內』
    的年份，確保 grid search 完全不碰到任何測試年資料（no test-set data leakage）。
    模型 checkpoint 與測試時相同（testYear{year}），僅在其原始 val set 上前向推論。
    """
    test_years = sorted(args.test_year)
    if len(test_years) == 1:
        ckpt_years = [test_years[0]]
    else:
        tset = set(test_years)
        ckpt_years = [y for y in test_years if (y - 1) not in tset]
    if not ckpt_years:
        raise ValueError(
            "grid search 找不到乾淨的驗證年份："
            "所有測試年的前一年都落在測試年清單內，無法避免 data leakage。")

    segs = [_infer_val_segment(args, price_df, risk_free_returns, device, cy)
            for cy in ckpt_years]
    common_seeds = sorted(set.intersection(*[set(s[0].keys()) for s in segs]))
    if not common_seeds:
        raise ValueError("grid search：沒有任何 seed 在所有驗證段都有 checkpoint。")

    seed_w_hats      = {s: np.vstack([seg[0][s] for seg in segs]) for s in common_seeds}
    seed_mu_hats_raw = {s: np.vstack([seg[1][s] for seg in segs]) for s in common_seeds}
    shared_mu_star   = np.vstack([seg[2] for seg in segs])
    shared_date_labels = pd.DatetimeIndex(
        np.concatenate([seg[3].values for seg in segs]))

    seed_port_rets = {}
    for s in common_seeds:
        pr, _, _ = _build_returns_from_w(
            w_hat=seed_w_hats[s], mu_star_arr=shared_mu_star,
            date_labels=shared_date_labels, risk_free_returns=risk_free_returns,
            stock_weight=args.stock_weight, transaction_cost=args.transaction_cost,
            rebalance_freq=args.weight_rebalance_freq)
        seed_port_rets[f'seed{s}'] = pr

    _, seed_rets_arr, seed_ic_arr, seed_wc_arr = _build_router_signal_arrays(
        seed_w_hats, seed_mu_hats_raw, seed_port_rets,
        shared_mu_star, shared_date_labels)

    # equal-weight ensemble 在驗證集上的表現（對照基準，無超參數）
    w_eq = np.mean(np.stack(list(seed_w_hats.values()), axis=0), axis=0)
    rs   = w_eq.sum(axis=1, keepdims=True)
    w_eq = w_eq / np.where(rs > 0, rs, 1)
    port_eq_val, _, _ = _build_returns_from_w(
        w_hat=w_eq, mu_star_arr=shared_mu_star, date_labels=shared_date_labels,
        risk_free_returns=risk_free_returns, stock_weight=args.stock_weight,
        transaction_cost=args.transaction_cost, rebalance_freq=args.weight_rebalance_freq)
    eq_val_m = _metrics(port_eq_val, label='val equal-weight', risk_free=risk_free_returns)

    combos = _expand_grid(args)
    metric = args.grid_search_metric
    print(f"\n{'='*60}")
    print(f"Grid search on validation set")
    print(f"  val ckpt years : {ckpt_years}  "
          f"(window [{shared_date_labels[0].date()} → {shared_date_labels[-1].date()}])")
    print(f"  val seeds      : {common_seeds}")
    print(f"  combos         : {len(combos)}")
    print(f"  objective      : max({metric})")
    print(f"  val equal-weight {metric} = {eq_val_m[metric]:.4f}")
    print(f"{'='*60}")
    if len(combos) == 1:
        print("[Grid search][WARN] 只有 1 組參數（未提供任何 --gs_* 清單），"
              "grid search 形同僅驗證預設值。")

    results = []
    for i, p in enumerate(tqdm(combos, desc='grid search')):
        w_router, _ = _zscore_router(
            seed_w_hats=seed_w_hats, seed_rets=seed_rets_arr,
            date_labels=shared_date_labels,
            warmup_days=p['warmup_days'], review_freq=p['review_freq'],
            top_k=p['top_k'], bottom_k=p['bottom_k'],
            rolling_window=p['rolling_window'], score_method=p['score_method'],
            combo_beta=p['combo_beta'], combo_alpha=p['combo_alpha'],
            seed_ic_arr=seed_ic_arr, seed_wc_arr=seed_wc_arr)
        pr, _, _ = _build_returns_from_w(
            w_hat=w_router, mu_star_arr=shared_mu_star,
            date_labels=shared_date_labels, risk_free_returns=risk_free_returns,
            stock_weight=args.stock_weight, transaction_cost=args.transaction_cost,
            rebalance_freq=args.weight_rebalance_freq)
        m = _metrics(pr, label=f'combo{i}', risk_free=risk_free_returns)
        results.append({**p, **{k: m[k] for k in _GS_METRIC_KEYS}})

    res_df = (pd.DataFrame(results)
              .sort_values(metric, ascending=False)
              .reset_index(drop=True))
    best = res_df.iloc[0]
    best_params = {}
    for k in _GS_PARAM_KEYS:
        if k in _GS_INT_KEYS:
            best_params[k] = int(best[k])
        elif k == 'score_method':
            best_params[k] = str(best[k])
        else:
            best_params[k] = float(best[k])

    print(f"\n[Grid search — top 10 by {metric}]")
    with pd.option_context('display.max_columns', None, 'display.width', 240):
        print(res_df.head(10).to_string(index=False))
    print(f"\n[Grid search] BEST params  → {best_params}")
    print(f"[Grid search] BEST val {metric} = {best[metric]:.4f}  "
          f"(equal-weight {eq_val_m[metric]:.4f})\n")

    if out_dir is not None:
        gs_dir = out_dir / 'grid_search'
        gs_dir.mkdir(parents=True, exist_ok=True)
        res_df.to_csv(gs_dir / 'grid_search_results.csv', index=False)
        with open(gs_dir / 'grid_search_best.json', 'w', encoding='utf-8') as f:
            json.dump({
                'objective':       metric,
                'val_ckpt_years':  ckpt_years,
                'val_window':      [str(shared_date_labels[0].date()),
                                    str(shared_date_labels[-1].date())],
                'val_seeds':       common_seeds,
                'best_params':     best_params,
                'best_val_metrics': {k: float(best[k]) for k in _GS_METRIC_KEYS},
                'val_equal_weight_metrics': {k: float(eq_val_m[k]) for k in _GS_METRIC_KEYS},
            }, f, indent=2, ensure_ascii=False)
        print(f"[Grid search] results saved → {gs_dir}")

    return best_params


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Multi-expert freeze-based router testing')

    parser.add_argument('--model_type', type=str, default='transformer',
                        choices=['mlp', 'lstm', 'transformer'])
    parser.add_argument('--test_year', type=int, nargs='+', required=True,
                        choices=[2022, 2023, 2024, 2025],
                        help='測試年份；單一年份（2024）為標準模式，多個年份（2022 2023 2024）為跨年度模式')
    parser.add_argument('--seeds', type=int, nargs='+',
                        default=[0, 2, 4, 42, 181, 276, 370, 413, 475,
                                 512, 518, 783, 788, 831, 1091, 1231,
                                 1405, 1497, 1560, 2024])
    parser.add_argument('--checkpoint_type', type=str,
                        default='checkpoint_best.pt',
                        choices=['checkpoint_best.pt', 'checkpoint_best_sharpe.pt'])
    parser.add_argument('--base_checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--data_path',  type=str, default='./dataset/market_index/')
    parser.add_argument('--risk_free_source', type=str, default='bill',
                        choices=['bill', 'g0q0'],
                        help='無風險利率來源：bill=3個月期零息公債殖利率換算；'
                             'g0q0=G0Q0公債指數日報酬（預設）。'
                             '同時作為 Sharpe/Sortino 的超額報酬基準，以及投組「無風險資產」部位的報酬來源')
    parser.add_argument('--risk_free_path', type=str,
                        default='./dataset/risk_free_index/US zero coupon bond-rf.csv',
                        help='risk_free_source=bill 時使用的無風險利率檔（Bloomberg 零息公債殖利率匯出檔）')
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--stock_weight',     type=float, default=0.6)
    parser.add_argument('--transaction_cost', type=float, default=0.002)
    parser.add_argument('--lookback_days',    type=int,   default=60)
    parser.add_argument('--rebalance_freq',   type=float, default=6,
                        choices=[0.25, 0.5, 1, 3, 6, 12],
                        help='Auto Reweight baseline 的再平衡頻率（單位：月）：'
                             '0.25=每週、0.5=每兩週、1/3/6/12=每 n 個月（default=6）')
    parser.add_argument('--weight_rebalance_freq', type=str, default='daily',
                        choices=['daily', 'weekly', 'biweekly', 'monthly',
                                 'quarterly', 'semiannual'],
                        help='模型投組權重多久調整一次：daily=每天(原始做法，預設)、'
                             'weekly=每週、biweekly=每兩週、monthly=每月、'
                             'quarterly=每季、semiannual=每半年，皆取該週期第一個交易日調整；'
                             '非調整日持倉沿用前一天、隨當天報酬drift、不交易（換手率為0）。'
                             '套用在每個seed、equal-weight ensemble、router ensemble。')
    parser.add_argument('--single_stock_file', type=str, default='MXWO.csv')
    parser.add_argument('--run_baseline', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed',   type=int, default=42)
    parser.add_argument('--start_day', type=str, default=None,
                        help='自訂推論起始日（格式：YYYY-MM-DD）；未指定時使用 test_year 對應的測試起始日')
    parser.add_argument('--end_day',   type=str, default=None,
                        help='自訂推論結束日（格式：YYYY-MM-DD）；未指定時使用 test_year 對應的測試結束日')

    # Z-score router hyperparameters
    parser.add_argument('--warmup_days', type=int, default=20,
                        help='初始 ensemble warmup 天數（default=20）')
    parser.add_argument('--review_freq', type=int, default=60,
                        help='Rolling Sharpe review 週期天數（default=30）')
    parser.add_argument('--top_k', type=int, default=3,
                        help='每次 review 選出 rolling Sharpe 最高的 top-k seeds 等權平均（default=7）')
    parser.add_argument('--rolling_window', type=int, default=0,
                        help='rolling buffer 視窗天數；0 = expanding window（從第一天累積到當天）（default=90）')
    parser.add_argument('--no_save', action='store_true', default=False,
                        help='跳過所有檔案輸出，只在 stdout 輸出 METRICS_JSON（供 hyperparam search 使用）')
    parser.add_argument('--score_method', type=str, default='cumret',
                        choices=['sharpe', 'cumret', 'mu_corr', 'ic_ir', 'w_corr', 'cumret_mucorr'],
                        help='選 seed 的評分方法：sharpe / cumret / mu_corr / ic_ir / w_corr / '
                             'cumret_mucorr（cumret 與 mu_corr 的 cross-seed z-score 相加，default=cumret）')
    parser.add_argument('--combo_beta', type=float, default=0.94,
                        help='cumret_mucorr 的 C_m 衰減加權因子 β（越接近 1 衰減越慢）（default=0.94）')
    parser.add_argument('--combo_alpha', type=float, default=0.5,
                        help='cumret_mucorr 的混合權重 α：Score=α·z(cumret)+(1-α)·z(mu_corr)；'
                             'α=1 等同純 cumret，α=0 等同純 mu_corr（default=0.5）')
    parser.add_argument('--bottom_k', type=int, default=0,
                        help='剔除 rolling score 最低的 bottom_k 個 seed，其餘等權平均；>0 時忽略 top_k（default=0）')

    # ── Grid search on validation set（inference 前先在 val set 上找最佳 router 參數）──
    parser.add_argument('--grid_search', action='store_true', default=False,
                        help='inference 前先在驗證集（測試年的前一年，與訓練時的 val set 相同）上'
                             '對 router 超參數跑 grid search，取最佳組合再跑測試集，'
                             '確保 router 參數挑選不觸及 test-set（no data leakage）')
    parser.add_argument('--grid_search_metric', type=str, default='sharpe',
                        choices=['sharpe', 'sortino', 'total_ret', 'ann_ret'],
                        help='grid search 目標（取最大值）（default=sharpe）')
    parser.add_argument('--gs_warmup_days',    type=int,   nargs='+', default=None,
                        help='grid search 掃描的 warmup_days 清單；未指定則固定用 --warmup_days')
    parser.add_argument('--gs_review_freq',    type=int,   nargs='+', default=None,
                        help='grid search 掃描的 review_freq 清單；未指定則固定用 --review_freq')
    parser.add_argument('--gs_top_k',          type=int,   nargs='+', default=None,
                        help='grid search 掃描的 top_k 清單；未指定則固定用 --top_k')
    parser.add_argument('--gs_bottom_k',       type=int,   nargs='+', default=None,
                        help='grid search 掃描的 bottom_k 清單；未指定則固定用 --bottom_k')
    parser.add_argument('--gs_rolling_window', type=int,   nargs='+', default=None,
                        help='grid search 掃描的 rolling_window 清單；未指定則固定用 --rolling_window')
    parser.add_argument('--gs_score_method',   type=str,   nargs='+', default=None,
                        choices=['sharpe', 'cumret', 'mu_corr', 'ic_ir', 'w_corr', 'cumret_mucorr'],
                        help='grid search 掃描的 score_method 清單；未指定則固定用 --score_method')
    parser.add_argument('--gs_combo_beta',     type=float, nargs='+', default=None,
                        help='grid search 掃描的 combo_beta 清單（僅 score_method=cumret_mucorr 有效）')
    parser.add_argument('--gs_combo_alpha',    type=float, nargs='+', default=None,
                        help='grid search 掃描的 combo_alpha 清單（僅 score_method=cumret_mucorr 有效）')


    args      = parser.parse_args()

    # ── 模式判斷 ─────────────────────────────────────────────
    is_multi_year = len(args.test_year) > 1
    if is_multi_year:
        _dir_tag = f"multiYear_{'_'.join(str(y) for y in args.test_year)}"
    else:
        _dir_tag = f"testYear{args.test_year[0]}"
    set_seed(args.seed)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    device    = args.device if torch.cuda.is_available() else 'cpu'

    if not args.no_save:
        out_dir = (Path(args.output_dir) / args.model_type
                   / _dir_tag / f"testTime{timestamp}")
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path   = out_dir / f"console_{timestamp}.log"
        sys.stdout = Logger(str(log_path))
    else:
        out_dir = None

    _year_label = (f"multi-year {args.test_year}" if is_multi_year
                   else f"test_year={args.test_year[0]}")
    print(f"--- Multi-Seed Ensemble | {args.model_type} | {_year_label} ---")

    # ── data ─────────────────────────────────────────────────
    price_df = load_price_data(args.data_path)

    if args.risk_free_source == 'g0q0':
        g0q0_path = Path(args.data_path) / 'G0Q0_Index.csv'
        risk_free_returns = load_price_index_returns(g0q0_path)
        print(f"Risk-free rate source: G0Q0_Index ({g0q0_path})")
    else:
        risk_free_returns = load_risk_free_rate(args.risk_free_path)
        print(f"Risk-free rate source: 3-month T-bill ({args.risk_free_path})")
    print(f"  ({risk_free_returns.index[0].date()} → {risk_free_returns.index[-1].date()}), "
          f"用於 Sharpe/Sortino 超額報酬與投組無風險資產部位")

    # ──────────────────────────────────────────────────────────
    # (Optional) Grid search on validation set → 鎖定 router 超參數
    # ──────────────────────────────────────────────────────────
    if args.grid_search:
        best_params = _grid_search_on_val(
            args, price_df, risk_free_returns, device, out_dir)
        for k, v in best_params.items():
            setattr(args, k, v)
        print(f"--- Grid search done. Router 參數已鎖定為 val-optimal："
              f"{best_params}；接著跑測試集 inference ---")

    # ──────────────────────────────────────────────────────────
    # Inference: all seeds
    # ──────────────────────────────────────────────────────────
    if is_multi_year:
        (seed_port_rets, seed_w_hats, seed_mu_hats_raw, shared_mu_star,
         shared_date_labels, asset_names, seed_metrics,
         title_suffix) = _infer_multi_year(
            args, price_df, risk_free_returns, args.test_year, device, out_dir)
        # baseline 用：整段連續序列的起訖日
        infer_start_str = shared_date_labels[0].strftime('%Y-%m-%d')
        infer_end_str   = shared_date_labels[-1].strftime('%Y-%m-%d')
    else:
        test_year = args.test_year[0]
        returns_df, split_info = prepare_data(
            price_df, lookback_days=args.lookback_days, test_year=test_year)

        infer_start_idx, infer_end_idx = _resolve_infer_window(
            returns_df, split_info, args.start_day, args.end_day, args.lookback_days)

        infer_split_info              = dict(split_info)
        infer_split_info['test_start'] = infer_start_idx
        infer_split_info['test_end']   = infer_end_idx

        _, _, test_loader = create_dataloaders(
            returns_df, infer_split_info,
            lookback_days=args.lookback_days, batch_size=1)

        infer_start_str = returns_df.index[infer_start_idx].strftime('%Y-%m-%d')
        infer_end_str   = returns_df.index[infer_end_idx].strftime('%Y-%m-%d')
        title_suffix    = (f"Test Year {test_year}  "
                           f"[{infer_start_str} → {infer_end_str}]")
        print(f"Inference period: {infer_start_str} → {infer_end_str}")
        asset_names = [n.replace('.csv', '') for n in split_info['asset_names']]

        seed_port_rets     = {}
        seed_w_hats        = {}
        seed_mu_hats_raw   = {}
        seed_metrics       = []
        shared_date_labels = None
        shared_mu_star     = None

        for seed in args.seeds:
            ckpt_path = (
                Path(args.base_checkpoint_dir) / args.model_type
                / f"testYear{test_year}" / f"seed{seed}"
                / args.checkpoint_type
            )
            if not ckpt_path.exists():
                print(f"  ✗ seed={seed}: not found, skipping")
                continue

            print(f"\n[seed={seed}] Loading ...")
            model = _load_model(ckpt_path, device)
            raw   = _inference_one_seed(model, test_loader, device)

            port_ret, tc, date_labels, w_actual = _build_returns_for_seed(
                raw, returns_df, risk_free_returns,
                stock_weight=args.stock_weight,
                transaction_cost=args.transaction_cost,
                rebalance_freq=args.weight_rebalance_freq)

            if not args.no_save:
                _save_subdir(
                    subdir=out_dir / f"seed{seed}", port_ret=port_ret,
                    turnover_cost=tc, date_labels=date_labels,
                    asset_names=asset_names, stock_weight=args.stock_weight,
                    title=f"seed {seed}", w_hat=w_actual,
                    mu_hat=raw['mu_hat'])

            label                  = f'seed{seed}'
            seed_port_rets[label]  = port_ret
            seed_w_hats[seed]      = raw['w_hat']
            seed_mu_hats_raw[seed] = raw['mu_hat']
            if shared_date_labels is None:
                shared_date_labels = date_labels
            if shared_mu_star is None:
                shared_mu_star = raw['mu_star']

            m = _metrics(port_ret, label=label, risk_free=risk_free_returns)
            seed_metrics.append(m)
            print(f"  Sharpe={m['sharpe']:.3f} | AnnRet={m['ann_ret']:.2%} | "
                  f"MaxDD={m['max_dd']:.2%}")

    if not seed_w_hats:
        print("No valid checkpoints. Exiting.")
        return

    if not args.no_save:
        pd.DataFrame(shared_mu_star, index=shared_date_labels,
                     columns=asset_names).to_csv(
            out_dir / 'ground_truth_return_history.csv')

        asset_metrics = [
            _metrics(pd.Series(shared_mu_star[:, i], index=shared_date_labels),
                     label=name, risk_free=risk_free_returns)
            for i, name in enumerate(asset_names)
        ]
        pd.DataFrame(asset_metrics).to_csv(
            out_dir / 'asset_performance_summary.csv', index=False)
        print(f"\n[Asset performance — buy & hold each asset alone | {title_suffix}]")
        _print_table(asset_metrics)

    # ──────────────────────────────────────────────────────────
    # Experts（各 metric 呈現所有 seed 的 cross-seed mean ± std，
    #          而非單一可執行組合；daily_returns 仍保留各 seed 報酬取平均，
    #          僅供 cumulative_returns.png 畫一條參考線用）
    # ──────────────────────────────────────────────────────────
    experts_ret = pd.DataFrame(seed_port_rets).dropna().mean(axis=1)
    experts_m   = _mean_std_metrics(seed_metrics, label='Experts')

    if not args.no_save:
        _save_subdir(
            subdir=out_dir / 'ensemble_experts', port_ret=experts_ret,
            turnover_cost=np.zeros(len(experts_ret)), date_labels=experts_ret.index,
            asset_names=asset_names, stock_weight=args.stock_weight,
            title='Experts', w_hat=None)
    print(f"\n  [Experts] "
          f"Sharpe={experts_m['sharpe'][0]:.3f}±{experts_m['sharpe'][1]:.3f} | "
          f"AnnRet={experts_m['ann_ret'][0]:.2%}±{experts_m['ann_ret'][1]:.2%} | "
          f"MaxDD={experts_m['max_dd'][0]:.2%}±{experts_m['max_dd'][1]:.2%}")

    # ──────────────────────────────────────────────────────────
    # Ensemble（等權平均所有 seed 的 w_hat，單一可執行組合）
    # ──────────────────────────────────────────────────────────
    w_equal  = np.mean(np.stack(list(seed_w_hats.values()), axis=0), axis=0)
    row_sums = w_equal.sum(axis=1, keepdims=True)
    w_equal  = w_equal / np.where(row_sums > 0, row_sums, 1)

    port_equal, tc_equal, w_equal_actual = _build_returns_from_w(
        w_hat=w_equal, mu_star_arr=shared_mu_star,
        date_labels=shared_date_labels, risk_free_returns=risk_free_returns,
        stock_weight=args.stock_weight, transaction_cost=args.transaction_cost,
        rebalance_freq=args.weight_rebalance_freq)
    equal_m = _metrics(port_equal, label='Ensemble', risk_free=risk_free_returns)

    if not args.no_save:
        _save_subdir(
            subdir=out_dir / 'ensemble_equal_weight', port_ret=port_equal,
            turnover_cost=tc_equal, date_labels=shared_date_labels,
            asset_names=asset_names, stock_weight=args.stock_weight,
            title='Ensemble', w_hat=w_equal_actual)
    print(f"\n  [Ensemble] Sharpe={equal_m['sharpe']:.3f} | "
          f"AnnRet={equal_m['ann_ret']:.2%} | MaxDD={equal_m['max_dd']:.2%}")

    # ──────────────────────────────────────────────────────────
    # Router
    # ──────────────────────────────────────────────────────────
    router_label = f'Router({args.score_method})'
    rw_label = 'expanding' if args.rolling_window == 0 else f'{args.rolling_window}d'
    _sel_desc = (f"bottom_k={args.bottom_k}" if args.bottom_k > 0
                 else f"top_k={args.top_k}")
    router_params_used = {
        'param_source':   'grid_search(val)' if args.grid_search else 'cli/default',
        'warmup_days':    args.warmup_days,
        'review_freq':    args.review_freq,
        'top_k':          args.top_k,
        'bottom_k':       args.bottom_k,
        'rolling_window': args.rolling_window,
        'score_method':   args.score_method,
        'combo_beta':     args.combo_beta,
        'combo_alpha':    args.combo_alpha,
        'grid_search_metric': args.grid_search_metric if args.grid_search else None,
    }
    _combo_desc = (f", combo_beta={args.combo_beta}, combo_alpha={args.combo_alpha}"
                   if args.score_method == 'cumret_mucorr' else "")
    print(f"\nRunning router  [params from "
          f"{'grid search on val set' if args.grid_search else 'CLI/defaults'}]")
    print(f"  (warmup={args.warmup_days}, review_freq={args.review_freq}, "
          f"{_sel_desc}, rolling_window={rw_label}, "
          f"score_method={args.score_method}{_combo_desc}) ...")
    seeds_sorted, seed_rets_arr, seed_ic_arr, seed_wc_arr = _build_router_signal_arrays(
        seed_w_hats, seed_mu_hats_raw, seed_port_rets,
        shared_mu_star, shared_date_labels)  # seed_*_arr 皆為 (N, T)

    w_router, router_record = _zscore_router(
        seed_w_hats = seed_w_hats,
        seed_rets   = seed_rets_arr,
        date_labels = shared_date_labels,
        warmup_days    = args.warmup_days,
        review_freq    = args.review_freq,
        top_k          = args.top_k,
        bottom_k       = args.bottom_k,
        rolling_window = args.rolling_window,
        score_method     = args.score_method,
        combo_beta       = args.combo_beta,
        combo_alpha      = args.combo_alpha,
        seed_ic_arr      = seed_ic_arr,
        seed_wc_arr      = seed_wc_arr,
    )
    port_router, tc_router, w_router_actual = _build_returns_from_w(
        w_hat=w_router, mu_star_arr=shared_mu_star,
        date_labels=shared_date_labels, risk_free_returns=risk_free_returns,
        stock_weight=args.stock_weight, transaction_cost=args.transaction_cost,
        rebalance_freq=args.weight_rebalance_freq)
    router_m = _metrics(port_router, label=router_label, risk_free=risk_free_returns)

    if not args.no_save:
        _save_subdir(
            subdir=out_dir / f'ensemble_router({args.score_method})', port_ret=port_router,
            turnover_cost=tc_router, date_labels=shared_date_labels,
            asset_names=asset_names, stock_weight=args.stock_weight,
            title=router_label, w_hat=w_router_actual)
        router_record.to_csv(
            out_dir / f'ensemble_router({args.score_method})' / 'router_decisions.csv', index=False)
        with open(out_dir / f'ensemble_router({args.score_method})' / 'router_params_used.json',
                  'w', encoding='utf-8') as f:
            json.dump(router_params_used, f, indent=2, ensure_ascii=False)
    print(f"\n  [{router_label}] Sharpe={router_m['sharpe']:.3f} | "
          f"AnnRet={router_m['ann_ret']:.2%} | MaxDD={router_m['max_dd']:.2%}")

    if not args.no_save:
        plot_zscore_router_analysis(
            record_df    = router_record,
            port_zscore  = port_router,
            port_equal   = port_equal,
            seeds        = sorted(seed_w_hats.keys()),
            title_suffix = title_suffix,
            save_path    = str(out_dir / f'ensemble_router({args.score_method})' / 'router_analysis.png'),
        )

    # ──────────────────────────────────────────────────────────
    # Baselines（optional）
    # ──────────────────────────────────────────────────────────
    baseline_rets    = {}
    baseline_metrics = []

    if args.run_baseline:
        print("\nRunning baselines ...")
        ew_ret, *_ = equal_weight_portfolio(
            Start_Day=infer_start_str, end_day=infer_end_str,
            transaction_cost=args.transaction_cost,
            stock_pool_weight=args.stock_weight,
            risk_free=risk_free_returns)
        baseline_rets['Equal Weight'] = ew_ret

        mpt_ret, *_ = auto_reweight_portfolio(
            rebalance_frequency=args.rebalance_freq,
            Start_Day=infer_start_str, end_day=infer_end_str,
            lookback_window=args.lookback_days,
            transaction_cost=args.transaction_cost,
            stock_pool_weight=args.stock_weight,
            risk_free=risk_free_returns)
        baseline_rets['Auto Reweight'] = mpt_ret

        stock_label    = args.single_stock_file.replace('.csv', '')
        single_ret, *_ = single_asset_portfolio(
            StockPoolWeight=args.stock_weight,
            stock_file=args.single_stock_file,
            Start_Day=infer_start_str, end_day=infer_end_str,
            transaction_cost=args.transaction_cost,
            risk_free=risk_free_returns)
        baseline_rets[f'Single ({stock_label})'] = single_ret
        baseline_metrics = [_metrics(v, k, risk_free=risk_free_returns) for k, v in baseline_rets.items()]
    else:
        print("\nBaselines skipped.")

    # ──────────────────────────────────────────────────────────
    # Metrics table
    # ──────────────────────────────────────────────────────────
    all_metrics = seed_metrics + [experts_m, equal_m, router_m] + baseline_metrics
    _print_table(all_metrics)

    if is_multi_year:
        per_year_rows = _per_year_metrics(
            seed_port_rets    = seed_port_rets,
            strategy_rets     = {'Ensemble': port_equal, router_label: port_router,
                                 **baseline_rets},
            risk_free_returns = risk_free_returns,
        )
        if not args.no_save:
            pd.DataFrame(_scale_metrics_for_csv(per_year_rows)).to_csv(
                out_dir / 'per_year_performance_summary.csv', index=False)

    if not args.no_save:
        pd.DataFrame(_scale_metrics_for_csv(all_metrics)).to_csv(
            out_dir / 'performance_summary.csv', index=False)

        common_idx = pd.DataFrame(seed_port_rets).dropna().index
        for k in baseline_rets:
            common_idx = common_idx.intersection(baseline_rets[k].index)

        all_port_aligned = {
            **{k: v.loc[common_idx] for k, v in seed_port_rets.items()},
            'Experts':   experts_ret.loc[common_idx],
            'Ensemble':  port_equal.loc[common_idx],
            router_label: port_router.loc[common_idx],
            **{k: v.loc[common_idx] for k, v in baseline_rets.items()},
        }

        plot_all_seeds(
            seed_port_rets = seed_port_rets,
            title_suffix   = title_suffix,
            save_path      = str(out_dir / 'all_seeds_daily_returns.png'),
        )
        report_high_divergence_days(
            seed_port_rets = seed_port_rets,
            div_csv_path   = str(out_dir / 'high_divergence_days.csv'),
        )
        plot_cumulative(all_port_aligned,
                        str(out_dir / 'cumulative_returns.png'), title_suffix)
        plot_seed_band(seed_port_rets,
                       str(out_dir / 'seed_robustness_band.png'), title_suffix)

        print(f"\n--- Testing completed. All outputs in: {out_dir} ---")
    if args.no_save:
        # 保留原本的 4 個 key（router_hyperparam_search.py 依賴這幾個欄位），
        # 額外附上完整的 router / equal-weight 指標，供 bootstrap variance 等
        # 需要更完整績效資訊的下游腳本使用。
        _metrics_json = {
            "eq_sharpe": equal_m["sharpe"], "eq_ret": equal_m["total_ret"],
            "router_sharpe": router_m["sharpe"], "router_ret": router_m["total_ret"],
            "router_ann_ret": router_m["ann_ret"], "router_ann_vol": router_m["ann_vol"],
            "router_sortino": router_m["sortino"], "router_max_dd": router_m["max_dd"],
            "eq_ann_ret": equal_m["ann_ret"], "eq_ann_vol": equal_m["ann_vol"],
            "eq_sortino": equal_m["sortino"], "eq_max_dd": equal_m["max_dd"],
        }
        print(f'METRICS_JSON:{json.dumps(_metrics_json)}')
        print(f'ROUTER_PARAMS_JSON:{json.dumps(router_params_used)}')


if __name__ == '__main__':
    main()