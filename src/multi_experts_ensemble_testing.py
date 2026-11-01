"""
multi_experts_ensemble_testing.py
─────────────────────────────────────────────────────────────────────────────
對多個 random seed 的 DFLMVOModel 做 inference，
專注比較 Experts 與 Ensemble 兩種多 seed 做法（router 請用 multi_experts_router_testing.py）：

  Experts                      : 各 seed 各自持倉計算報酬；指標為 cross-seed mean ± std
                                 （daily_returns 為各 seed 報酬取平均，僅供畫參考線，非可執行組合）
  Ensemble (equal weight)      : 每天對所有 seed 的 w_hat 取等權平均，直接執行

performance_summary.csv 的列順序為：各 seed → Experts → Ensemble (equal weight) → baselines。
跨年度模式另外輸出 per_year_performance_summary.csv：把整段連續序列依日曆年切開的逐年指標。
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataloader import create_dataloaders, prepare_data, load_price_data
from utils import load_risk_free_rate
from backtest_utils import (
    _METRIC_KEYS, Logger, set_seed,
    _resolve_group_indices, _resolve_infer_window, _load_model, _inference_one_seed,
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
# Constants
# ─────────────────────────────────────────────────────────────
_BAND_COLOR = 'steelblue'


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _resolve_group_min_weight(threshold: float, stock_weight: float) -> float:
    """
    把「整體投組視角」的門檻換算成「股票池內部視角」的下限，供 MVO 限制式使用：

        threshold(整體投組) = group_sum(股票池) * stock_weight
        ⟹ group_sum(股票池) >= threshold / stock_weight

    threshold 不可能超過 stock_weight（股票整體能拿到的資金上限），超過就無解。
    """
    if stock_weight <= 0:
        raise ValueError(f"stock_weight 必須 > 0，收到：{stock_weight}")
    floor = threshold / stock_weight
    if floor > 1.0:
        raise ValueError(
            f"--group_weight_threshold={threshold} 在 --stock_weight={stock_weight} "
            f"下無法達成：股票池需保留 {floor:.4f}（>100%）才夠，"
            f"threshold 不能超過 stock_weight。")
    return floor


def _group_weight_coverage(
    w_actual:     np.ndarray,
    asset_names:  List[str],
    group_assets: Optional[str],
    threshold:    float = 0.3,
    stock_weight: float = 1.0,
) -> Dict:
    """
    計算 group_assets 這幾檔標的，在整段測試期間「佔整體投資組合(股+債)的合計權重
    >= threshold」的天數佔全部交易日的比例。

    w_actual 這19檔加總=1（股票池內部視角），乘上 stock_weight 後才是這幾檔
    佔「股+債整體投資組合」的實際比例，因此會隨 stock_weight 變動。

    Returns
    -------
    dict，group_assets 為 None/空字串時回傳空 dict。
    """
    group_indices = _resolve_group_indices(asset_names, group_assets)
    if not group_indices:
        return {}

    group_sum_portfolio = w_actual[:, group_indices].sum(axis=1) * stock_weight
    n_total = len(group_sum_portfolio)
    n_ge    = int((group_sum_portfolio >= threshold).sum())
    pct     = n_ge / n_total if n_total > 0 else float('nan')

    return {
        'group_assets':          group_assets,
        'threshold':             threshold,
        'stock_weight':          stock_weight,
        'n_days':                n_total,
        'n_days_ge_threshold':   n_ge,
        'pct_days_ge_threshold': round(pct, 4),  # 4位小數的比例 = 2位小數的百分比
        'mean_group_weight':     round(float(group_sum_portfolio.mean()), 2),
        'min_group_weight':      round(float(group_sum_portfolio.min()), 2),
        'max_group_weight':      round(float(group_sum_portfolio.max()), 2),
    }


# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────

def _asset_metrics(
    mu_star_arr: np.ndarray,
    asset_names: List[str],
    date_labels: pd.DatetimeIndex,
    risk_free:   Optional[pd.Series] = None,
) -> List[Dict]:
    """
    個別標的「買進持有」（100% 單押該標的，全程不調整）在測試期間的績效，
    用跟 portfolio / seed / ensemble 完全相同的財務指標（見 _metrics）。
    """
    return [
        _metrics(pd.Series(mu_star_arr[:, i], index=date_labels), label=name,
                 risk_free=risk_free)
        for i, name in enumerate(asset_names)
    ]


def _format_summary_for_csv(rows: List[Dict]) -> pd.DataFrame:
    """
    performance_summary.csv 的存檔格式：
    total_ret / ann_ret / ann_vol 轉成百分比（*100）、max_dd 取絕對值的百分比，
    全部取小數點後兩位，欄位改名為 'Total Return(%)' 等。
    Experts 這種 (mean, std) tuple 欄位格式化成 "mean ± std" 字串
    （max_dd 的 mean 取絕對值，std 維持原值）。
    rows 帶 'year' 欄位時（per_year_performance_summary.csv）保留在第一欄。
    """
    pct_cols = {'total_ret': 'Total Return(%)', 'ann_ret': 'Annualized Return(%)',
                'ann_vol':   'Annualized Volatility(%)', 'max_dd': 'Max Drawdown(%)'}
    out = []
    for row in rows:
        r = {'year': row['year']} if 'year' in row else {}
        r['label'] = row['label']
        for k in _METRIC_KEYS:
            v     = row[k]
            scale = 100 if k in pct_cols else 1
            if isinstance(v, tuple):
                m, s = v
                m = abs(m) if k == 'max_dd' else m
                r[k] = f"{m * scale:.2f} ± {s * scale:.2f}"
            else:
                v = abs(v) if k == 'max_dd' else v
                r[k] = round(v * scale, 2)
        out.append(r)
    return pd.DataFrame(out).rename(columns=pct_cols)


# ─────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────

def plot_all_seeds(
    seed_port_rets:      Dict[str, pd.Series],
    title_suffix:        str,
    save_path:           str,
    mode:                str   = 'cumulative',
    div_threshold_sigma: float = 1.5,
    div_csv_path:        Optional[str] = None,
    figsize:             Optional[tuple] = None,
    line_alpha:          float = 0.80,
    linewidth:           float = 1.4,
    dpi:                 int   = 150,
) -> None:
    if mode not in ('cumulative', 'daily', 'daily_lines'):
        raise ValueError(f"mode must be 'cumulative', 'daily', or 'daily_lines', got {mode!r}")
    if not seed_port_rets:
        print(f"[plot_all_seeds] seed_port_rets is empty, skipping {save_path}")
        return

    seed_df    = pd.DataFrame(seed_port_rets).dropna()
    n_seeds    = len(seed_df.columns)
    cmap       = plt.cm.tab20
    daily_mean = seed_df.mean(axis=1)
    daily_std  = seed_df.std(axis=1)

    if figsize is None:
        figsize = (16, 6) if mode == 'cumulative' else (16, 12)

    if mode == 'cumulative':
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

    elif mode == 'daily':
        std_threshold  = daily_std.mean() + div_threshold_sigma * daily_std.std()
        high_div_dates = daily_std[daily_std > std_threshold].index

        print(f"\n  [High divergence days] threshold={std_threshold:.5f}, "
              f"n={len(high_div_dates)} days")
        for d in high_div_dates:
            print(f"    {d.date()}  std={daily_std[d]:.5f}  "
                  f"mean_ret={daily_mean[d]:.5f}")

        fig, axes = plt.subplots(
            3, 1, figsize=figsize,
            gridspec_kw={'height_ratios': [2, 1, 1]},
            sharex=True,
        )
        fig.subplots_adjust(hspace=0.35)

        for i, col in enumerate(seed_df.columns):
            axes[0].plot(seed_df.index, seed_df[col].values, label=col,
                         color=cmap(i / max(n_seeds, 1)),
                         linewidth=linewidth, alpha=line_alpha)

        for d in high_div_dates:
            for ax in axes[:2]:
                ax.axvline(d, color='red', linewidth=1.0, linestyle='--',
                           alpha=0.5, zorder=4)
            axes[0].annotate(
                f"▼ {d.strftime('%m/%d')}",
                xy=(d, -0.04), xycoords=('data', 'axes fraction'),
                xytext=(0, 0), textcoords='offset points',
                fontsize=8, color='red', rotation=270,
                ha='center', va='top', annotation_clip=False, zorder=6,
            )

        axes[0].axhline(0, color='gray', linestyle='--', linewidth=0.8)
        axes[0].set_title(
            f'Daily Return (All Seeds) — {title_suffix}\n'
            f'Red dashed: high divergence days (cross-seed std > {std_threshold:.4f})'
        )
        axes[0].set_ylabel('Daily Return')
        axes[0].legend(fontsize=6, ncol=5, loc='upper left', framealpha=0.5)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(daily_mean, color=_BAND_COLOR, linewidth=1.0, label='Cross-seed mean')
        axes[1].fill_between(daily_mean.index, daily_mean - daily_std,
                             daily_mean + daily_std,
                             alpha=0.25, color=_BAND_COLOR, label='±1 std')
        axes[1].axhline(0, color='gray', linestyle='--', linewidth=0.7)
        axes[1].set_title('Daily Return — Cross-seed Mean ± Std')
        axes[1].set_ylabel('Return')
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        axes[2].fill_between(daily_std.index, 0, daily_std.values,
                             color=_BAND_COLOR, alpha=0.4, label='Cross-seed std')
        axes[2].plot(daily_std, color=_BAND_COLOR, linewidth=1.0)
        axes[2].axhline(std_threshold, color='red', linestyle='--', linewidth=1.0,
                        label=f'Threshold ({std_threshold:.4f})')
        axes[2].scatter(high_div_dates, daily_std[high_div_dates].values,
                        color='red', s=60, marker='D', zorder=5,
                        edgecolors='#A00000', linewidths=0.8)
        axes[2].set_title('Cross-seed Std of Daily Return')
        axes[2].set_ylabel('Std')
        axes[2].set_xlabel('Date')
        axes[2].legend(fontsize=8)
        axes[2].grid(True, alpha=0.3)

        if div_csv_path and len(high_div_dates) > 0:
            div_df = pd.DataFrame({
                'date':            high_div_dates,
                'cross_seed_std':  daily_std[high_div_dates].values,
                'cross_seed_mean': daily_mean[high_div_dates].values,
            }).sort_values('cross_seed_std', ascending=False)
            div_df.to_csv(div_csv_path, index=False)
            print(f"  High divergence days saved → {div_csv_path}")

    elif mode == 'daily_lines':
        fig, axes = plt.subplots(
            2, 1, figsize=figsize,
            gridspec_kw={'height_ratios': [3, 1]},
            sharex=True,
        )
        fig.subplots_adjust(hspace=0.25)

        for i, col in enumerate(seed_df.columns):
            axes[0].plot(seed_df.index, seed_df[col].values, label=col,
                         color=cmap(i / max(n_seeds, 1)),
                         linewidth=linewidth * 0.7, alpha=line_alpha * 0.6, zorder=3)

        axes[0].plot(daily_mean.index, daily_mean.values,
                     color='white', linewidth=3.0, zorder=4)
        axes[0].plot(daily_mean.index, daily_mean.values,
                     label='Cross-seed mean', color='#E84040',
                     linewidth=2.0, zorder=5)
        axes[0].axhline(0, color='gray', linestyle='--', linewidth=0.7)
        axes[0].set_title(f'Daily Return — All Seeds | {title_suffix}')
        axes[0].set_ylabel('Daily Return')
        axes[0].legend(fontsize=6, ncol=5, loc='upper left', framealpha=0.5)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(daily_mean.index, daily_mean.values,
                     color='#E84040', linewidth=1.5, label='Cross-seed mean')
        axes[1].fill_between(daily_mean.index, daily_mean - daily_std,
                             daily_mean + daily_std,
                             alpha=0.25, color=_BAND_COLOR, label='±1 std')
        axes[1].axhline(0, color='gray', linestyle='--', linewidth=0.7)
        axes[1].set_title('Cross-seed Mean ± Std')
        axes[1].set_ylabel('Return')
        axes[1].set_xlabel('Date')
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved: {save_path}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Multi-expert testing: Experts vs equal-weight Ensemble')

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
    parser.add_argument('--risk_free_path', type=str,
                        default='./dataset/risk_free_index/US zero coupon bond-rf.csv',
                        help='無風險利率檔（Bloomberg 零息公債殖利率匯出檔）：'
                             '同時作為 Sharpe/Sortino 的超額報酬基準，以及投組「無風險資產」部位的報酬來源')
    parser.add_argument('--output_dir', type=str, default='./ablation_study_results')
    parser.add_argument('--stock_weight',     type=float, default=1.0)
    parser.add_argument('--transaction_cost', type=float, default=0.002)
    parser.add_argument('--group_assets', type=str,
                        default='SPX,CCMP,MXWO,MXWD,C0A0_Index',
                        help='關注的標的組合，逗號分隔。用途1: ensemble (equal weight) 合計權重'
                             '覆蓋率統計（見--group_weight_threshold）。'
                             '用途2: --enforce_group_floor開啟時，MVO硬性下限套用的標的')
    parser.add_argument('--group_weight_threshold', type=float, default=0.3,
                        help='--group_assets合計權重（整體投組視角，已乘上stock_weight）的門檻，'
                             '預設只用來統計coverage報表；加上--enforce_group_floor後同一個值'
                             '會變成MVO裡的硬性下限（default=0.3）')
    parser.add_argument('--enforce_group_floor', action='store_true', default=False,
                        help='開啟後，把--group_weight_threshold（整體投組視角）換算成股票池'
                             '內部下限，直接寫進MVO的限制式：cp.sum(w[group_idx]) >= '
                             'threshold/stock_weight，由優化器依當天mu/Sigma自行決定組內怎麼'
                             '分配。預設關閉，維持原本做法。')
    parser.add_argument('--lookback_days',    type=int,   default=60)
    parser.add_argument('--rebalance_freq',   type=float, default=1,
                        choices=[0.25, 0.5, 1, 3, 6, 12],
                        help='Auto Reweight baseline 的再平衡頻率（單位：月）：'
                             '0.25=每週、0.5=每兩週、1/3/6/12=每 n 個月（default=1）')
    parser.add_argument('--weight_rebalance_freq', type=str, default='daily',
                        choices=['daily', 'weekly', 'biweekly', 'monthly',
                                 'quarterly', 'semiannual'],
                        help='模型投組權重多久調整一次：daily=每天(原始做法，預設)、'
                             'weekly=每週、biweekly=每兩週、monthly=每月、'
                             'quarterly=每季、semiannual=每半年，皆取該週期第一個交易日調整；'
                             '非調整日持倉沿用前一天、隨當天報酬drift、不交易（換手率為0）。'
                             '套用在每個seed、equal-weight ensemble。')
    parser.add_argument('--single_stock_file', type=str, default='MXWO.csv')
    parser.add_argument('--run_baseline', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed',   type=int, default=42)
    parser.add_argument('--start_day', type=str, default=None,
                        help='自訂推論起始日（格式：YYYY-MM-DD）；未指定時使用 test_year 對應的測試起始日')
    parser.add_argument('--end_day',   type=str, default=None,
                        help='自訂推論結束日（格式：YYYY-MM-DD）；未指定時使用 test_year 對應的測試結束日')
    parser.add_argument('--no_save', action='store_true', default=False,
                        help='跳過所有檔案輸出，只在 stdout 輸出 METRICS_JSON')

    args      = parser.parse_args()

    # ── 模式判斷 ─────────────────────────────────────────────
    is_multi_year = len(args.test_year) > 1
    if is_multi_year:
        _dir_tag = f"multiYear_{'_'.join(str(y) for y in args.test_year)}"
    else:
        _dir_tag = f"testYear{args.test_year[0]}"
    if args.enforce_group_floor:
        _dir_tag += f"_gfloor{args.group_weight_threshold}"

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
    if args.enforce_group_floor:
        _floor_stock_pool = _resolve_group_min_weight(args.group_weight_threshold, args.stock_weight)
        print(f"enforce_group_floor=True → MVO硬性下限: {args.group_assets} 合計權重 "
              f">= {_floor_stock_pool:.4f}（股票池視角，換算自整體投組threshold="
              f"{args.group_weight_threshold} / stock_weight={args.stock_weight}），"
              f"組內分配由優化器自行決定")

    # ── data ─────────────────────────────────────────────────
    price_df = load_price_data(args.data_path)

    risk_free_returns = load_risk_free_rate(args.risk_free_path)
    print(f"Risk-free rate loaded from {args.risk_free_path} "
          f"({risk_free_returns.index[0].date()} → {risk_free_returns.index[-1].date()}), "
          f"用於 Sharpe/Sortino 超額報酬與投組無風險資產部位")

    # ──────────────────────────────────────────────────────────
    # Inference: all seeds
    # ──────────────────────────────────────────────────────────
    group_min_weight = (
        _resolve_group_min_weight(args.group_weight_threshold, args.stock_weight)
        if args.enforce_group_floor else None)

    if is_multi_year:
        (seed_port_rets, seed_w_hats, seed_mu_hats_raw, shared_mu_star,
         shared_date_labels, asset_names, seed_metrics,
         title_suffix) = _infer_multi_year(
            args, price_df, risk_free_returns, args.test_year, device, out_dir,
            group_assets=args.group_assets, group_min_weight=group_min_weight)
        # baseline 用：整段連續序列的起訖日
        infer_start_str = shared_date_labels[0].strftime('%Y-%m-%d')
        infer_end_str   = shared_date_labels[-1].strftime('%Y-%m-%d')
    else:
        test_year = args.test_year[0]
        returns_df, split_info = prepare_data(
            price_df, lookback_days=args.lookback_days, test_year=test_year)

        infer_start_idx, infer_end_idx = _resolve_infer_window(
            returns_df, split_info, args.start_day, args.end_day, args.lookback_days)

        infer_split_info               = dict(split_info)
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
        asset_names   = [n.replace('.csv', '') for n in split_info['asset_names']]
        group_indices = _resolve_group_indices(asset_names, args.group_assets)

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
            model = _load_model(ckpt_path, device, group_indices=group_indices,
                                 group_min_weight=group_min_weight)
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

    # ──────────────────────────────────────────────────────────
    # 個別標的（買進持有）績效
    # ──────────────────────────────────────────────────────────
    asset_metrics = _asset_metrics(shared_mu_star, asset_names, shared_date_labels,
                                    risk_free=risk_free_returns)
    print("\n  [Individual Asset Buy-and-Hold Performance]")
    _print_table(asset_metrics)
    if not args.no_save:
        pd.DataFrame(asset_metrics).to_csv(
            out_dir / 'asset_performance_summary.csv', index=False)

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
    # Ensemble (equal weight)
    # ──────────────────────────────────────────────────────────
    w_equal  = np.mean(np.stack(list(seed_w_hats.values()), axis=0), axis=0)
    row_sums = w_equal.sum(axis=1, keepdims=True)
    w_equal  = w_equal / np.where(row_sums > 0, row_sums, 1)

    port_equal, tc_equal, w_equal_actual = _build_returns_from_w(
        w_hat=w_equal, mu_star_arr=shared_mu_star,
        date_labels=shared_date_labels, risk_free_returns=risk_free_returns,
        stock_weight=args.stock_weight, transaction_cost=args.transaction_cost,
        rebalance_freq=args.weight_rebalance_freq)
    equal_m = _metrics(port_equal, label='Ensemble (equal weight)', risk_free=risk_free_returns)

    if not args.no_save:
        _save_subdir(
            subdir=out_dir / 'ensemble_equal_weight', port_ret=port_equal,
            turnover_cost=tc_equal, date_labels=shared_date_labels,
            asset_names=asset_names, stock_weight=args.stock_weight,
            title='Ensemble (equal weight)', w_hat=w_equal_actual)

    group_cov = _group_weight_coverage(
        w_equal_actual, asset_names, args.group_assets, args.group_weight_threshold,
        stock_weight=args.stock_weight)
    if group_cov:
        print(f"  [Ensemble (equal weight)] {group_cov['group_assets']} "
              f"佔整體投組合計權重(x{group_cov['stock_weight']}stock_weight) >= "
              f"{group_cov['threshold']}: {group_cov['pct_days_ge_threshold']:.2%} "
              f"({group_cov['n_days_ge_threshold']}/{group_cov['n_days']} 天)  "
              f"mean={group_cov['mean_group_weight']:.2f}  "
              f"min={group_cov['min_group_weight']:.2f}  "
              f"max={group_cov['max_group_weight']:.2f}")
        if not args.no_save:
            pd.DataFrame([group_cov]).to_csv(
                out_dir / 'ensemble_equal_weight' / 'group_weight_coverage.csv', index=False)

    print(f"\n  [Ensemble (equal weight)] Sharpe={equal_m['sharpe']:.3f} | "
          f"AnnRet={equal_m['ann_ret']:.2%} | MaxDD={equal_m['max_dd']:.2%}")

    # ──────────────────────────────────────────────────────────
    # Baselines（optional）
    # ──────────────────────────────────────────────────────────
    baseline_rets    = {}
    baseline_weights = {}
    baseline_metrics = []

    if args.run_baseline:
        print("\nRunning baselines ...")
        ew_ret, *_, ew_w = equal_weight_portfolio(
            Start_Day=infer_start_str, end_day=infer_end_str,
            transaction_cost=args.transaction_cost,
            stock_pool_weight=args.stock_weight,
            risk_free=risk_free_returns, return_weights=True)
        baseline_rets['Equal Weight']    = ew_ret
        baseline_weights['equal_weight'] = ew_w

        mpt_ret, *_, mpt_w = auto_reweight_portfolio(
            rebalance_frequency=args.rebalance_freq,
            Start_Day=infer_start_str, end_day=infer_end_str,
            lookback_window=args.lookback_days,
            transaction_cost=args.transaction_cost,
            stock_pool_weight=args.stock_weight,
            risk_free=risk_free_returns, return_weights=True)
        baseline_rets['Auto Reweight']    = mpt_ret
        baseline_weights['auto_reweight'] = mpt_w

        stock_label    = args.single_stock_file.replace('.csv', '')
        single_ret, *_, single_w = single_asset_portfolio(
            StockPoolWeight=args.stock_weight,
            stock_file=args.single_stock_file,
            Start_Day=infer_start_str, end_day=infer_end_str,
            transaction_cost=args.transaction_cost,
            risk_free=risk_free_returns, return_weights=True)
        baseline_rets[f'Single ({stock_label})']    = single_ret
        baseline_weights[f'single_{stock_label}']  = single_w
        baseline_metrics = [_metrics(v, k, risk_free=risk_free_returns) for k, v in baseline_rets.items()]

        # 各 baseline 在每個再平衡日的目標權重（整體投組視角，含 Risk_Free）
        if not args.no_save:
            for name, w_df in baseline_weights.items():
                subdir = out_dir / 'baselines' / name
                subdir.mkdir(parents=True, exist_ok=True)
                # + 0.0 把 MVO 解出的 -0.0 數值雜訊轉成 0.0
                (w_df.round(4) + 0.0).to_csv(subdir / 'weights_history.csv')
    else:
        print("\nBaselines skipped.")

    # ──────────────────────────────────────────────────────────
    # Metrics table
    # ──────────────────────────────────────────────────────────
    all_metrics = seed_metrics + [experts_m, equal_m] + baseline_metrics
    _print_table(all_metrics)

    if is_multi_year:
        per_year_rows = _per_year_metrics(
            seed_port_rets    = seed_port_rets,
            strategy_rets     = {'Ensemble (equal weight)': port_equal, **baseline_rets},
            risk_free_returns = risk_free_returns,
        )
        if not args.no_save:
            _format_summary_for_csv(per_year_rows).to_csv(
                out_dir / 'per_year_performance_summary.csv', index=False)

    if not args.no_save:
        _format_summary_for_csv(all_metrics).to_csv(
            out_dir / 'performance_summary.csv', index=False)

        common_idx = pd.DataFrame(seed_port_rets).dropna().index
        for k in baseline_rets:
            common_idx = common_idx.intersection(baseline_rets[k].index)

        all_port_aligned = {
            **{k: v.loc[common_idx] for k, v in seed_port_rets.items()},
            'Experts':                 experts_ret.loc[common_idx],
            'Ensemble (equal weight)': port_equal.loc[common_idx],
            **{k: v.loc[common_idx] for k, v in baseline_rets.items()},
        }

        plot_all_seeds(
            seed_port_rets = seed_port_rets,
            title_suffix   = title_suffix,
            save_path      = str(out_dir / 'all_seeds_daily_returns.png'),
            mode           = 'cumulative',
        )
        plot_cumulative(all_port_aligned,
                        str(out_dir / 'cumulative_returns.png'), title_suffix,
                        ensemble_label='Ensemble (equal weight)')
        plot_seed_band(seed_port_rets,
                       str(out_dir / 'seed_robustness_band.png'), title_suffix)

        print(f"\n--- Testing completed. All outputs in: {out_dir} ---")
    if args.no_save:
        metrics_json = {"eq_sharpe": equal_m["sharpe"], "eq_ret": equal_m["total_ret"],
                        "experts_sharpe_mean": experts_m["sharpe"][0],
                        "experts_sharpe_std":  experts_m["sharpe"][1]}
        print(f'METRICS_JSON:{json.dumps(metrics_json)}')


if __name__ == '__main__':
    main()