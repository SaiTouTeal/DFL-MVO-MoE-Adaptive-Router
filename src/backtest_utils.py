"""
backtest_utils.py
─────────────────────────────────────────────────────────────────────────────
multi_experts_router_testing.py 與 multi_experts_ensemble_testing.py 共用的程式：
checkpoint 載入與 inference、依權重計算每日淨報酬（含 rebalance 頻率）、
跨年度 inference、每個策略子資料夾的存檔、績效指標與共用圖表。

兩支腳本各自特有的部分（router 演算法、grid search、group floor 換算與
coverage 統計、performance_summary.csv 格式化等）仍留在各自的檔案裡。
"""

import sys
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import skew, kurtosis, gaussian_kde
from tqdm import tqdm

from agents import DFLMVOModel
from dataloader import create_dataloaders, prepare_data
from utils import (
    Total_Return, Annualized_Return, Annualized_Volatility,
    Sharpe_Ratio, Sortino_Ratio, Max_Drawdown,
    calculate_cost_to_alpha_ratio,
)


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

# 20 檔 market_index 資產裡的 8 檔債券指數；其餘 12 檔視為全球股票指數。
_BOND_TICKERS = {
    'C0A0_Index', 'EMUSTRUU_Index', 'G0Q0_Index', 'H0A0_Index',
    'LBUSTRUU_Index', 'LEGATRUU_Index', 'LP01TREU_Index', 'LP06TREU_Index',
}
_GROUP_COLORS = {
    'Global Equity (12)': '#2a78d6',
    'Bond Index (8)':     '#008300',
    'Risk-Free':          '#e87ba4',
}

_METRIC_KEYS = ('total_ret', 'ann_ret', 'ann_vol', 'sharpe', 'sortino', 'max_dd')

# 舊版 checkpoint 才有、已從 DFLMVOModel 移除的 buffer；載入時忽略
_LEGACY_STATE_KEYS = {'floor_add'}


# ─────────────────────────────────────────────────────────────
# Logging / reproducibility
# ─────────────────────────────────────────────────────────────

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────
# Model loading & inference
# ─────────────────────────────────────────────────────────────

def _resolve_group_indices(
    asset_names:   List[str],
    group_assets:  Optional[str],
) -> Optional[List[int]]:
    """把逗號分隔的標的名稱字串轉成在 asset_names 裡的索引清單。"""
    if not group_assets:
        return None
    names = [n.strip() for n in group_assets.split(',') if n.strip()]
    return [asset_names.index(n) for n in names]


def _load_model(
    ckpt_path:        Path,
    device:           str,
    group_indices:    Optional[List[int]] = None,
    group_min_weight: Optional[float]     = None,
) -> DFLMVOModel:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_path = ckpt_path.parent / 'config.json'
    with open(cfg_path) as f:
        cfg = json.load(f)
    model = DFLMVOModel(
        n_assets         = cfg['n_assets'],
        lookback_days    = cfg['lookback_days'],
        lambda_risk      = cfg['risk_aversion'],
        max_weight       = cfg['max_weight'],
        alpha            = cfg['alpha'],
        transaction_cost = cfg['transaction_cost'],
        model_type       = cfg['model_type'],
        backbone_config  = cfg.get('backbone_config', {}),
        device           = device,
        group_indices    = group_indices,
        group_min_weight = group_min_weight,
    ).to(device)
    saved_sd = ckpt['model_state_dict']
    # cvx_layer 的參數由 MVO 層建構時自行產生，不從 checkpoint 載入
    filtered = {k: v for k, v in saved_sd.items() if 'cvx_layer' not in k}
    result   = model.load_state_dict(filtered, strict=False)

    # strict=False 只是為了容許上面兩類已知差異；其餘對不上的 key 不能默默略過
    missing    = [k for k in result.missing_keys if 'cvx_layer' not in k]
    unexpected = [k for k in result.unexpected_keys if k not in _LEGACY_STATE_KEYS]
    weight_mismatch = [k for k in missing + unexpected if k.startswith('prediction_model.')]
    if weight_mismatch:
        raise RuntimeError(
            f"{ckpt_path}: prediction_model 權重與目前模型架構對不上 {weight_mismatch}，"
            "推論結果會不正確，請確認 config.json / agents.py 與訓練時一致。")
    if missing or unexpected:
        print(f"  [WARN] {ckpt_path}: state_dict 有未預期的差異 "
              f"missing={missing} unexpected={unexpected}")
    model.eval()
    return model


def _resolve_infer_window(
    returns_df:    pd.DataFrame,
    split_info:    Dict,
    start_day:     Optional[str],
    end_day:       Optional[str],
    lookback_days: int,
) -> Tuple[int, int]:
    """
    單年度模式：把 --start_day / --end_day（未指定時用 test_year 的測試起訖日）
    轉成 returns_df 上的 (infer_start_idx, infer_end_idx)，並檢查：
      - 起始日不可晚於結束日
      - 起始日之前至少要有 lookback_days 天歷史
      - 區間內至少有一個交易日
    """
    custom_start = (pd.Timestamp(start_day) if start_day
                    else returns_df.index[split_info['test_start']])
    custom_end   = (pd.Timestamp(end_day) if end_day
                    else returns_df.index[split_info['test_end']])

    if custom_start > custom_end:
        raise ValueError(
            f"--start_day ({custom_start.date()}) must be <= "
            f"--end_day ({custom_end.date()})")
    if custom_start < returns_df.index[lookback_days]:
        raise ValueError(
            f"--start_day ({custom_start.date()}) is too early: "
            f"need at least {lookback_days} days of history before it.")

    infer_start_idx = returns_df.index.searchsorted(custom_start)
    infer_end_idx   = returns_df.index.searchsorted(custom_end, side='right') - 1

    if infer_start_idx > infer_end_idx:
        raise ValueError(
            f"No trading days found in [{custom_start.date()}, {custom_end.date()}]. "
            "Check --start_day / --end_day.")
    return infer_start_idx, infer_end_idx


def _inference_one_seed(model: DFLMVOModel, test_loader, device: str) -> Dict:
    model.eval()
    model.reset_w_prev()
    mu_hat_list, mu_star_list, w_hat_list, dates_list = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='  inference', leave=False):
            X       = batch['X'].to(device)
            mu_star = batch['mu_star'].to(device)
            Sigma   = batch['Sigma'].to(device)
            idx     = batch['idx']
            out     = model(X, Sigma, mu_star)
            w_hat   = out['w_hat'].cpu().numpy()
            w_hat               = np.clip(w_hat, 0, None)
            w_hat[w_hat < 1e-4] = 0
            row_sums            = w_hat.sum(axis=1, keepdims=True)
            w_hat               = w_hat / np.where(row_sums > 0, row_sums, 1)
            mu_hat_list.append(out['mu_hat'].cpu().numpy())
            mu_star_list.append(mu_star.cpu().numpy())
            w_hat_list.append(w_hat)
            dates_list.extend(idx.numpy().tolist())
    return {
        'mu_hat':  np.concatenate(mu_hat_list,  axis=0),
        'mu_star': np.concatenate(mu_star_list, axis=0),
        'w_hat':   np.concatenate(w_hat_list,   axis=0),
        'dates':   dates_list,
    }


# ─────────────────────────────────────────────────────────────
# Portfolio returns
# ─────────────────────────────────────────────────────────────

def _build_rebalance_mask(date_labels: pd.DatetimeIndex, freq: str) -> np.ndarray:
    """
    依 freq 產生每天是否為「調整日」的布林遮罩：
      'daily'      → 全部 True（原始做法，每天都調整）
      'weekly'     → 每週第一個交易日 True
      'biweekly'   → 每兩週第一個交易日 True
      'monthly'    → 每月第一個交易日 True
      'quarterly'  → 每季第一個交易日 True
      'semiannual' → 每半年（1-6月 / 7-12月）第一個交易日 True
    非調整日沿用前一天持倉、隨當天報酬 drift，不交易。
    """
    T = len(date_labels)
    if freq == 'daily':
        return np.ones(T, dtype=bool)

    dt = pd.Series(date_labels)
    if freq == 'weekly':
        key = dt.dt.to_period('W')
    elif freq == 'biweekly':
        # 以固定的週一（1970-01-05）為起點、每 14 天一組，跨年也連續
        # （ISO year*53+week 在只有 52 週的年份交界會讓兩組各只剩一週）
        monday = dt - pd.to_timedelta(dt.dt.weekday, unit='D')
        key    = (monday - pd.Timestamp('1970-01-05')).dt.days // 14
    elif freq == 'monthly':
        key = dt.dt.to_period('M')
    elif freq == 'quarterly':
        key = dt.dt.to_period('Q')
    elif freq == 'semiannual':
        key = dt.dt.year * 2 + (dt.dt.month > 6).astype(int)
    else:
        raise ValueError(
            "weight_rebalance_freq 必須是 'daily'/'weekly'/'biweekly'/'monthly'/"
            f"'quarterly'/'semiannual'，收到：{freq!r}")

    mask    = (key != key.shift(1)).values
    mask[0] = True  # 第一天一定要建倉
    return mask


def _build_returns_from_w(
    w_hat:             np.ndarray,
    mu_star_arr:       np.ndarray,
    date_labels:       pd.DatetimeIndex,
    risk_free_returns: pd.Series,
    stock_weight:      float,
    transaction_cost:  float,
    rebalance_freq:    str = 'daily',
) -> Tuple[pd.Series, np.ndarray, np.ndarray]:
    """
    drift-adjusted w_prev，計算每日淨報酬。

    rebalance_freq='daily'（預設）時完全等同原本做法：每天都把持倉調整成 w_hat[t]。
    'weekly'/'monthly' 時，只有調整日才交易到 w_hat[t]，其餘日子沿用前一天持倉
    隨當天報酬 drift（不交易，換手率為 0）。

    Returns
    -------
    port_series   : 每日淨報酬
    turnover_cost : 每日交易成本
    w_actual      : 實際持倉權重時序（調整日=w_hat[t]，非調整日=drift後的前一天持倉）
    """
    bw = 1.0 - stock_weight
    T  = len(w_hat)
    rebalance_mask = _build_rebalance_mask(date_labels, rebalance_freq)

    w_actual     = np.zeros_like(w_hat)
    w_drift_hist = np.zeros_like(w_hat)  # [0] 保持全 0，代表第一天從空手建倉（沿用原本做法）
    w_actual[0]  = w_hat[0]
    for t in range(1, T):
        w_prev   = w_actual[t - 1]
        r_t      = mu_star_arr[t - 1]
        w_grown  = w_prev * (1.0 + r_t)
        port_val = w_grown.sum()
        w_drift  = w_grown / port_val if port_val > 1e-8 else w_prev
        w_drift_hist[t] = w_drift
        w_actual[t]     = w_hat[t] if rebalance_mask[t] else w_drift

    turnover      = np.abs(w_actual - w_drift_hist).sum(axis=1)
    turnover_cost = transaction_cost * turnover * stock_weight
    risk_free_ret_aligned = risk_free_returns.reindex(date_labels).fillna(0).values
    stock_port_ret   = (w_actual * mu_star_arr).sum(axis=1)
    portfolio_ret    = (stock_port_ret * stock_weight
                        + risk_free_ret_aligned * bw
                        - turnover_cost)
    return pd.Series(portfolio_ret, index=date_labels), turnover_cost, w_actual


def _build_returns_for_seed(
    raw:               Dict,
    returns_df:        pd.DataFrame,
    risk_free_returns: pd.Series,
    stock_weight:      float,
    transaction_cost:  float,
    rebalance_freq:    str = 'daily',
) -> Tuple[pd.Series, np.ndarray, pd.DatetimeIndex, np.ndarray]:
    date_labels  = returns_df.index[raw['dates']]
    port_ret, tc, w_actual = _build_returns_from_w(
        w_hat=raw['w_hat'], mu_star_arr=raw['mu_star'],
        date_labels=date_labels, risk_free_returns=risk_free_returns,
        stock_weight=stock_weight, transaction_cost=transaction_cost,
        rebalance_freq=rebalance_freq,
    )
    return port_ret, tc, date_labels, w_actual


# ─────────────────────────────────────────────────────────────
# Multi-year inference
# ─────────────────────────────────────────────────────────────

def _infer_multi_year(
    args,
    price_df:          pd.DataFrame,
    risk_free_returns: pd.Series,
    years:             List[int],
    device:            str,
    out_dir:           Optional[Path],
    group_assets:      Optional[str]   = None,
    group_min_weight:  Optional[float] = None,
) -> Tuple[Dict[str, pd.Series], Dict[int, np.ndarray], Dict[int, np.ndarray],
           np.ndarray, pd.DatetimeIndex, List[str], List[Dict], str]:
    """
    跨年度 inference：依序使用各 testYear 的模型，拼接成連續時間序列。
    - 年度邊界以 split_info 的 test_end 為準
    - w_prev（換手率）全程連續（router 的 rolling buffer 也因此可跨年連續）
    - group_assets / group_min_weight：MVO 裡這幾檔標的合計權重的硬性下限，
      兩者皆給才生效；None = 不設
    """
    asset_names:   Optional[List[str]]      = None
    group_indices: Optional[List[int]]      = None
    seed_w_acc:    Dict[int, list]          = {}
    seed_muh_acc:  Dict[int, list]          = {}
    mu_acc:        List[np.ndarray]         = []
    dt_acc:        List[pd.DatetimeIndex]   = []

    for year in years:
        ret_df, si = prepare_data(price_df, lookback_days=args.lookback_days, test_year=year)
        ts = ret_df.index[si['test_start']]
        te = ret_df.index[si['test_end']]
        print(f"\n─── Year {year}  [{ts.date()} → {te.date()}] ───")

        if asset_names is None:
            asset_names   = [n.replace('.csv', '') for n in si['asset_names']]
            group_indices = _resolve_group_indices(asset_names, group_assets)

        _, _, loader = create_dataloaders(
            ret_df, si, lookback_days=args.lookback_days, batch_size=1)

        year_w:   Dict[int, np.ndarray]      = {}
        year_muh: Dict[int, np.ndarray]      = {}
        year_mu:  Optional[np.ndarray]       = None
        year_dt:  Optional[pd.DatetimeIndex] = None

        for seed in args.seeds:
            ckpt = (Path(args.base_checkpoint_dir) / args.model_type
                    / f"testYear{year}" / f"seed{seed}" / args.checkpoint_type)
            if not ckpt.exists():
                print(f"  ✗ seed={seed}: 無 testYear{year} checkpoint，跳過")
                continue
            model = _load_model(ckpt, device, group_indices=group_indices,
                                 group_min_weight=group_min_weight)
            raw   = _inference_one_seed(model, loader, device)
            year_w[seed]   = raw['w_hat']
            year_muh[seed] = raw['mu_hat']
            if year_mu is None:
                year_mu = raw['mu_star']
                year_dt = ret_df.index[raw['dates']]

        if not year_w:
            print(f"  [WARN] year {year} 無有效 seed，跳過")
            continue

        for s, w in year_w.items():
            seed_w_acc.setdefault(s, []).append(w)
        for s, mh in year_muh.items():
            seed_muh_acc.setdefault(s, []).append(mh)
        mu_acc.append(year_mu)
        dt_acc.append(year_dt)

    n_seg = len(dt_acc)
    valid_seeds = sorted(s for s, lst in seed_w_acc.items() if len(lst) == n_seg)
    if not valid_seeds:
        raise ValueError("沒有任何 seed 在所有所需年度都有 checkpoint。")

    seed_w_hats        = {s: np.vstack(seed_w_acc[s])  for s in valid_seeds}
    seed_mu_hats_all   = {s: np.vstack(seed_muh_acc[s]) for s in valid_seeds}
    shared_mu_star     = np.vstack(mu_acc)
    shared_date_labels = pd.DatetimeIndex(
        np.concatenate([d.values for d in dt_acc]))

    seed_port_rets: Dict[str, pd.Series] = {}
    seed_metrics:   List[Dict]           = []
    for seed in valid_seeds:
        port_ret, tc, w_actual = _build_returns_from_w(
            w_hat             = seed_w_hats[seed],
            mu_star_arr       = shared_mu_star,
            date_labels       = shared_date_labels,
            risk_free_returns = risk_free_returns,
            stock_weight      = args.stock_weight,
            transaction_cost  = args.transaction_cost,
            rebalance_freq    = args.weight_rebalance_freq,
        )
        label = f'seed{seed}'
        seed_port_rets[label] = port_ret
        if out_dir is not None:
            _save_subdir(
                subdir        = out_dir / label,
                port_ret      = port_ret,
                turnover_cost = tc,
                date_labels   = shared_date_labels,
                asset_names   = asset_names,
                stock_weight  = args.stock_weight,
                title         = label,
                w_hat         = w_actual,
                mu_hat        = seed_mu_hats_all[seed],
            )
        m = _metrics(port_ret, label=label, risk_free=risk_free_returns)
        seed_metrics.append(m)
        print(f"  [seed={seed}] Sharpe={m['sharpe']:.3f} | "
              f"AnnRet={m['ann_ret']:.2%} | MaxDD={m['max_dd']:.2%}")

    title_suffix = f"Multi-Year {years[0]}–{years[-1]}"
    return (seed_port_rets, seed_w_hats, seed_mu_hats_all, shared_mu_star,
            shared_date_labels, asset_names, seed_metrics, title_suffix)


# ─────────────────────────────────────────────────────────────
# Daily return distribution
# ─────────────────────────────────────────────────────────────

def _historical_var_cvar(r: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """
    歷史模擬法計算 VaR / CVaR（回傳正數，代表虧損幅度）。

    VaR(confidence)  = -左尾 (1-confidence) 分位數
    CVaR(confidence)  = -左尾分位數以下所有樣本的平均（Expected Shortfall）
    """
    alpha       = 1.0 - confidence
    var_thresh  = np.percentile(r, alpha * 100)      # 左尾分位數（負值）
    tail        = r[r <= var_thresh]
    cvar_thresh = tail.mean() if len(tail) > 0 else var_thresh
    return -var_thresh, -cvar_thresh


def plot_return_distribution(
    port_ret:       pd.Series,
    save_path:      str,
    title:          str,
    bins:           int   = 50,
    var_confidence: float = 0.95,
) -> None:
    """畫每日報酬的分布圖（直方圖 + KDE），並標注 mean/std/skew/kurtosis/VaR/CVaR 等統計量。"""
    r = port_ret.dropna().values
    if len(r) < 2:
        print(f"[plot_return_distribution] 樣本數不足，跳過 {save_path}")
        return

    mean_r, std_r, median_r = r.mean(), r.std(), np.median(r)
    skew_r, kurt_r = skew(r), kurtosis(r)  # kurtosis 為超額峰度，常態=0
    var_r, cvar_r  = _historical_var_cvar(r, var_confidence)
    conf_pct       = int(round(var_confidence * 100))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(r, bins=bins, density=True, color='steelblue',
            alpha=0.6, edgecolor='white', label='Daily return')

    # 左尾（虧損超過 VaR 的區域）淡紅色網底標示
    ax.axvspan(r.min(), -var_r, color='#E84040', alpha=0.12, zorder=0)

    if std_r > 1e-12:
        try:
            kde    = gaussian_kde(r)
            x_grid = np.linspace(r.min(), r.max(), 300)
            ax.plot(x_grid, kde(x_grid), color='#E84040',
                    linewidth=1.8, label='KDE')
        except Exception:
            pass

    ax.axvline(mean_r, color='black', linestyle='--', linewidth=1.2,
               label=f'Mean = {mean_r:.4%}')
    ax.axvline(median_r, color='gray', linestyle=':', linewidth=1.2,
               label=f'Median = {median_r:.4%}')
    ax.axvline(-var_r, color='#B87A10', linestyle='-.', linewidth=1.4,
               label=f'VaR{conf_pct} = -{var_r:.4%}')
    ax.axvline(-cvar_r, color='#7F1D1D', linestyle='-.', linewidth=1.6,
               label=f'CVaR{conf_pct} = -{cvar_r:.4%}')
    ax.axvline(0, color='gray', linewidth=0.7)

    stats_text = (
        f"N       = {len(r)}\n"
        f"Std     = {std_r:.4%}\n"
        f"Skew    = {skew_r:.3f}\n"
        f"Kurt    = {kurt_r:.3f}\n"
        f"Min     = {r.min():.4%}\n"
        f"Max     = {r.max():.4%}\n"
        f"VaR{conf_pct}   = {var_r:.4%}\n"
        f"CVaR{conf_pct}  = {cvar_r:.4%}"
    )
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            fontsize=9, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_title(f'Daily Return Distribution — {title}')
    ax.set_xlabel('Daily Return')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Per-strategy output folder
# ─────────────────────────────────────────────────────────────

def _save_subdir(
    subdir:        Path,
    port_ret:      pd.Series,
    turnover_cost: np.ndarray,
    date_labels:   pd.DatetimeIndex,
    asset_names:   List[str],
    stock_weight:  float,
    title:         str,
    w_hat:         np.ndarray = None,
    mu_hat:        np.ndarray = None,
) -> None:
    subdir.mkdir(parents=True, exist_ok=True)
    port_ret.to_csv(subdir / 'daily_returns.csv', header=True)

    if mu_hat is not None:
        pd.DataFrame(mu_hat, index=date_labels, columns=asset_names).to_csv(
            subdir / 'expected_return_history.csv')

    if w_hat is not None:
        bw  = 1.0 - stock_weight
        df  = pd.DataFrame(w_hat * stock_weight,
                           index=date_labels, columns=asset_names)
        df['Risk_Free'] = bw
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.stackplot(df.index, [df[c].values for c in df.columns],
                     labels=df.columns, alpha=0.7)
        ax.set_title(f'Weight distribution — {title}')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Weight')
        ax.legend(loc='upper right', fontsize=6, ncol=5, framealpha=0.5)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(subdir / 'weights_stacked.png', dpi=150)
        plt.close(fig)
        pd.DataFrame(w_hat, index=date_labels, columns=asset_names).to_csv(
            subdir / 'weights_history.csv')

        # ── Group weight distribution：全球股票指數 / 債券指數 / 無風險資產 ──
        bond_cols  = [c for c in asset_names if c in _BOND_TICKERS]
        stock_cols = [c for c in asset_names if c not in _BOND_TICKERS]
        group_df = pd.DataFrame({
            'Global Equity (12)': df[stock_cols].sum(axis=1),
            'Bond Index (8)':     df[bond_cols].sum(axis=1),
            'Risk-Free':          df['Risk_Free'],
        }, index=df.index)
        group_cols   = list(_GROUP_COLORS.keys())
        group_colors = [_GROUP_COLORS[c] for c in group_cols]
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.stackplot(group_df.index,
                     [group_df[c].values for c in group_cols],
                     labels=group_cols, colors=group_colors, alpha=0.85)
        ax.set_title(f'Group Weight Distribution (Equity / Bond / Risk-Free) — {title}')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Weight')
        ax.legend(loc='upper right', fontsize=8, framealpha=0.6)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(subdir / 'group_weights_stacked.png', dpi=150)
        plt.close(fig)
        group_df.to_csv(subdir / 'group_weights_history.csv')

    tc_series = pd.Series(turnover_cost, index=date_labels)
    cost_m    = calculate_cost_to_alpha_ratio(port_ret, tc_series)
    pd.DataFrame([cost_m]).to_csv(subdir / 'cost_metrics.csv', index=False)

    if w_hat is not None:
        w_prev_s     = np.zeros_like(w_hat)
        w_prev_s[1:] = w_hat[:-1]
        daily_to     = np.abs(w_hat - w_prev_s).sum(axis=1)
        pd.DataFrame({
            'date':           date_labels,
            'daily_turnover': daily_to,
            'daily_tc':       turnover_cost,
            'avg_turnover':   daily_to.mean(),
            'total_turnover': daily_to.sum(),
        }).to_csv(subdir / 'turnover_analysis.csv', index=False)
        print(f"         avg_turnover={daily_to.mean():.4f}  "
              f"total={daily_to.sum():.2f}")

    print(f"  [OK] {subdir.name}/  "
          f"net={cost_m['total_net_return']:.4f}  "
          f"cost_ratio={cost_m['cost_to_alpha_ratio']*100:.2f}%")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                             gridspec_kw={'height_ratios': [2, 1]})
    cum = (1 + port_ret.dropna()).cumprod()
    axes[0].plot(cum, color='steelblue', linewidth=1.2)
    axes[0].axhline(1.0, color='gray', linestyle='--', linewidth=0.7)
    axes[0].set_title(f'Cumulative Return — {title}')
    axes[0].set_ylabel('Portfolio Value (start = 1)')
    axes[0].grid(True, alpha=0.3)
    colors = ['#E84040' if r < 0 else '#4CAF50' for r in port_ret.dropna().values]
    axes[1].bar(port_ret.dropna().index, port_ret.dropna().values,
                color=colors, width=1.0, alpha=0.7)
    axes[1].axhline(0, color='gray', linestyle='--', linewidth=0.7)
    axes[1].set_title('Daily Return')
    axes[1].set_ylabel('Return')
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(subdir / 'daily_returns_plot.png', dpi=150)
    plt.close(fig)

    plot_return_distribution(
        port_ret  = port_ret,
        save_path = str(subdir / 'daily_returns_distribution.png'),
        title     = title,
    )


# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────

def _metrics(ret: pd.Series, label: str, risk_free: Optional[pd.Series] = None) -> Dict:
    r = ret.dropna()
    return {
        'label':     label,
        'total_ret': Total_Return(r),
        'ann_ret':   Annualized_Return(r),
        'ann_vol':   Annualized_Volatility(r),
        'sharpe':    Sharpe_Ratio(r, risk_free),
        'sortino':   Sortino_Ratio(r, risk_free),
        'max_dd':    Max_Drawdown(r),
    }


def _mean_std_metrics(seed_metrics: List[Dict], label: str) -> Dict:
    """
    把多個 seed 各自算好的 metrics，逐欄位取 cross-seed (mean, std)。
    每個欄位存成 (mean, std) tuple，供 _print_table 與各腳本的 CSV 存檔函式
    格式化成 "mean ± std" 字串顯示。
    """
    return {
        'label': label,
        **{k: (float(np.mean([m[k] for m in seed_metrics])),
               float(np.std([m[k] for m in seed_metrics])))
           for k in _METRIC_KEYS},
    }


def _fmt_metric(v, pct: bool = True) -> str:
    """把單一數值或 (mean, std) tuple 格式化成字串，供 console 表格顯示。"""
    if isinstance(v, tuple):
        m, s = v
        return f"{m:.2%}±{s:.2%}" if pct else f"{m:.3f}±{s:.3f}"
    return f"{v:.2%}" if pct else f"{v:.3f}"


def _print_table(rows: List[Dict]):
    hdr = (f"{'Strategy':<32} {'TotalRet':>18} {'AnnRet':>18} {'AnnVol':>18} "
           f"{'Sharpe':>16} {'Sortino':>16} {'MaxDD':>18}")
    print('\n' + '─' * len(hdr))
    print(hdr)
    print('─' * len(hdr))
    for r in rows:
        print(
            f"{r['label']:<32} "
            f"{_fmt_metric(r['total_ret']):>18} "
            f"{_fmt_metric(r['ann_ret']):>18} "
            f"{_fmt_metric(r['ann_vol']):>18} "
            f"{_fmt_metric(r['sharpe'], pct=False):>16} "
            f"{_fmt_metric(r['sortino'], pct=False):>16} "
            f"{_fmt_metric(r['max_dd']):>18}"
        )
    print('─' * len(hdr) + '\n')


def _per_year_metrics(
    seed_port_rets:    Dict[str, pd.Series],
    strategy_rets:     Dict[str, pd.Series],
    risk_free_returns: pd.Series,
) -> List[Dict]:
    """
    跨年度模式：把整段連續序列依日曆年切開，逐年計算各策略指標。
    各年是同一條連續序列的切片（持倉、router buffer 皆跨年連續），不是每年重跑；
    Experts 一樣是當年各 seed 指標的 cross-seed mean ± std。
    每列多一個 'year' 欄位，列順序同 performance_summary.csv。
    """
    years = sorted(pd.DataFrame(seed_port_rets).index.year.unique())
    rows  = []
    for y in years:
        def _in_year(s: pd.Series) -> pd.Series:
            return s[s.index.year == y].dropna()

        seed_m = [_metrics(_in_year(r), label=k, risk_free=risk_free_returns)
                  for k, r in seed_port_rets.items()]
        year_rows = seed_m + [_mean_std_metrics(seed_m, label='Experts')]
        for k, r in strategy_rets.items():
            r_y = _in_year(r)
            if len(r_y) < 2:   # 例如 baseline 沒涵蓋到這一年
                continue
            year_rows.append(_metrics(r_y, label=k, risk_free=risk_free_returns))

        print(f"\n[Per-year performance | {y}]")
        _print_table(year_rows)
        rows.extend({'year': y, **r} for r in year_rows)
    return rows


# ─────────────────────────────────────────────────────────────
# Aggregate plots
# ─────────────────────────────────────────────────────────────

def plot_cumulative(
    port_dict:      Dict[str, pd.Series],
    save_path:      str,
    title_suffix:   str,
    ensemble_label: str = 'Ensemble',
) -> None:
    """所有策略的累積報酬比較圖。ensemble_label 為等權 ensemble 那條線的標籤（畫成藍色虛線）。"""
    SEED_COLOR    = '#90B4D4'
    EXPERTS_COLOR = '#E84040'
    EQUAL_COLOR   = 'steelblue'

    fig, ax   = plt.subplots(figsize=(14, 6))
    seed_line = None

    for label, ret in port_dict.items():
        cum = (1 + ret.dropna()).cumprod()
        if label.startswith('seed'):
            line, = ax.plot(cum, color=SEED_COLOR, linewidth=0.8, alpha=0.45)
            if seed_line is None:
                seed_line = line
        elif label == 'Experts':
            ax.plot(cum, label=label, color=EXPERTS_COLOR,
                    linewidth=2.0, zorder=5)
        elif label == ensemble_label:
            ax.plot(cum, label=label, color=EQUAL_COLOR,
                    linewidth=2.0, linestyle='--', zorder=5)
        else:
            ax.plot(cum, label=label, color='#2C2C2C',
                    linewidth=1.5, zorder=4)

    if seed_line is not None:
        proxy = Line2D([0], [0], color=SEED_COLOR, linewidth=0.8, alpha=0.7)
        handles, labels_leg = ax.get_legend_handles_labels()
        ax.legend([proxy] + handles,
                  ['Individual seeds'] + labels_leg,
                  fontsize=8, ncol=2)

    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.7)
    ax.set_title(f'Cumulative Returns — {title_suffix}')
    ax.set_ylabel('Portfolio value (start = 1)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_seed_band(
    seed_rets:    Dict[str, pd.Series],
    save_path:    str,
    title_suffix: str,
) -> None:
    aligned = pd.DataFrame(seed_rets).dropna()
    cum     = (1 + aligned).cumprod()
    mean    = cum.mean(axis=1)
    std     = cum.std(axis=1)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(mean.index, mean - std, mean + std,
                    alpha=0.25, color='steelblue', label='±1 std across seeds')
    ax.plot(mean, color='steelblue', linewidth=1.5, label='Mean across seeds')
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.7)
    ax.set_title(f'Seed robustness band — {title_suffix}')
    ax.set_ylabel('Portfolio value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")
