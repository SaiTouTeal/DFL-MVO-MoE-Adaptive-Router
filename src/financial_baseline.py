import numpy as np
import pandas as pd
from utils import *
import os
import cvxpy as cp
import argparse


def solve_weights(mu, Sigma, n, method):
    if method == "closed-form":
        w_raw = np.linalg.solve(Sigma, mu)
        return w_raw / np.sum(w_raw)
    elif method == "constrain":
        w = cp.Variable(n)
        prob = cp.Problem(
            cp.Maximize(mu @ w - 1.0 * cp.quad_form(w, Sigma)),
            [cp.sum(w) == 1, w >= 0]
        )
        prob.solve()
        return w.value


def _rebalance_calendar_freq(rebalance_frequency: float) -> str:
    """
    rebalance_frequency 單位是「月」：
        >= 1 的整數 → 每 n 個月的月初（'nMS'）
        0.5 → 每兩週（'2W-MON'），0.25 → 每週（'W-MON'），以每月 4 週換算
    """
    if rebalance_frequency >= 1:
        if rebalance_frequency != int(rebalance_frequency):
            raise ValueError(f"rebalance_frequency >= 1 時必須是整數月份，收到：{rebalance_frequency}")
        return f'{int(rebalance_frequency)}MS'
    n_weeks = rebalance_frequency * 4
    if n_weeks <= 0 or n_weeks != int(n_weeks):
        raise ValueError(f"rebalance_frequency < 1 時必須是 0.25 的倍數（以週為單位），收到：{rebalance_frequency}")
    return f'{int(n_weeks)}W-MON'


def _build_rebalance_dates(index: pd.DatetimeIndex,
                            Start_Day: str,
                            end_day: str,
                            rebalance_frequency: float) -> pd.DatetimeIndex:
    """
    建立再平衡日期序列（頻率換算見 _rebalance_calendar_freq）。

    核心修正：
        pd.date_range(start=Start_Day, freq='MS') 的第一個日期是
        「Start_Day 所在月份的月初」或「下一個月的月初」，
        如果 Start_Day 不是月初（例如 2022-01-03），
        第一個 MS 日期就是 2022-02-01，導致 1 月份沒有 rebalance，
        portfolio_returns 那段是 NaN，最後 dropna 後從 2 月才開始。

        修正方式：強制把「Start_Day 的第一個實際交易日」加入
        rebalance_dates 的開頭，確保從第一天就開始持有。
    """
    start_ts = pd.Timestamp(Start_Day)
    end_ts   = pd.Timestamp(end_day)

    # 找 Start_Day 對應的第一個實際交易日
    loc          = index.get_indexer([start_ts], method='bfill')[0]
    actual_start = index[loc]

    # 定期再平衡日（月度從下個月 MS 開始；週度從下個週一開始）
    calendar = pd.date_range(start=Start_Day, end=index[-1],
                             freq=_rebalance_calendar_freq(rebalance_frequency))
    monthly_dates = index[index.get_indexer(calendar, method='bfill')]
    monthly_dates = monthly_dates[monthly_dates <= end_ts]

    # 把 actual_start 加到最前面，去重後排序
    all_dates = pd.DatetimeIndex([actual_start]).append(monthly_dates).unique().sort_values()
    all_dates  = all_dates[all_dates <= end_ts]

    return all_dates


def _blend_with_risk_free(pool_returns: pd.Series,
                           risk_free: pd.Series,
                           stock_pool_weight: float,
                           Start_Day: str,
                           end_day: str,
                           rebalance_frequency: int,
                           transaction_cost: float) -> pd.Series:
    """
    將資產池報酬（pool_returns）與無風險利率（risk_free）依
    stock_pool_weight : 1 - stock_pool_weight 定期再平衡混合，
    邏輯與 single_asset_portfolio 相同（含 drift 與轉換成本）。
    """
    risk_free_ret = risk_free[(risk_free.index >= pd.Timestamp(Start_Day)) &
                              (risk_free.index <= pd.Timestamp(end_day))]

    combined = pd.concat([pool_returns, risk_free_ret], axis=1, join='inner').dropna()
    combined.columns = ['pool', 'risk_free']

    target_weights = np.array([stock_pool_weight, 1.0 - stock_pool_weight])

    rebalance_dates = _build_rebalance_dates(
        combined.index, Start_Day, end_day, rebalance_frequency
    )

    blended_returns = pd.Series(index=combined.index, dtype=float)
    prev_weights    = None

    for i, date in enumerate(rebalance_dates):
        if i < len(rebalance_dates) - 1:
            next_date  = rebalance_dates[i + 1]
            period_idx = combined.loc[date:next_date].index[:-1]
        else:
            period_idx = combined.loc[date:end_day].index

        if len(period_idx) == 0:
            continue

        period_rets = combined.loc[period_idx].dot(target_weights).copy()

        if prev_weights is not None:
            prev_date       = rebalance_dates[i - 1]
            drift_period    = combined.loc[prev_date:date]
            growth_factors  = (1 + drift_period).prod().values
            drifted         = prev_weights * growth_factors
            drifted_weights = drifted / drifted.sum()
            tc_amount       = transaction_cost * np.abs(
                                  target_weights - drifted_weights).sum()
        else:
            tc_amount = transaction_cost * np.abs(target_weights).sum()

        period_rets.iloc[0] -= tc_amount
        blended_returns.loc[period_idx] = period_rets
        prev_weights = target_weights.copy()

    return blended_returns.dropna()


def _to_portfolio_weights(pool_weights: pd.DataFrame,
                          stock_pool_weight: float) -> pd.DataFrame:
    """
    把資產池內部權重（每列加總=1）換算成整體投組視角：
    各標的 * stock_pool_weight，另加 Risk_Free = 1 - stock_pool_weight。
    欄位名稱去掉 .csv 副檔名。
    """
    df = pool_weights.astype(float).dropna(how='all') * stock_pool_weight
    df.columns = [str(c).replace('.csv', '') for c in df.columns]
    df['Risk_Free'] = 1.0 - stock_pool_weight
    df.index.name = 'Dates'
    return df


# ──────────────────────────────────────────────────────────────
# Equal Weight Portfolio
# ──────────────────────────────────────────────────────────────

def equal_weight_portfolio(Start_Day: str = "2020-01-01",
                           end_day: str = "2025-10-09",
                           rebalance_frequency: int = 1,
                           transaction_cost: float = 0.003,
                           dataset_dir: str = "./dataset/market_index",
                           stock_pool_weight: float = 0.6,
                           risk_free=None,
                           return_weights: bool = False):
    """
    等權重投資組合，定期再平衡並扣除交易成本。
    return_weights=True 時，回傳 tuple 最後多一個各再平衡日的目標權重 DataFrame
    （整體投組視角，含 Risk_Free 欄）。
    確保 portfolio_returns 從 Start_Day 的第一個實際交易日開始。

    資產池內部（預設 market_index，目前含 20 檔，已內含 G0Q0 債券指數）採等權重，
    再以 stock_pool_weight : 1 - stock_pool_weight（預設 6:4）與 risk_free
    定期再平衡混合。
    """
    if risk_free is None:
        raise ValueError("equal_weight_portfolio 需要傳入 risk_free（用於 6:4 無風險資產混合）")

    all_data = []
    files    = sorted([f for f in os.listdir(dataset_dir) if f.endswith(".csv")])
    n_files  = len(files)

    for file_name in files:
        df  = pd.read_csv(os.path.join(dataset_dir, file_name))
        df['Dates'] = pd.to_datetime(df['Dates'])
        df  = df[df['Dates'] >= "2000-01-01"].set_index('Dates')
        # 先對完整歷史算 pct_change，再截日期
        ret = df['PX_LAST'].pct_change().dropna()
        ret = ret[(ret.index >= pd.Timestamp(Start_Day)) &
                  (ret.index <= pd.Timestamp(end_day))]
        ret.name = file_name
        all_data.append(ret)

    returns = pd.concat(all_data, axis=1).fillna(0)
    returns = returns[(returns.index >= pd.Timestamp(Start_Day)) &
                      (returns.index <= pd.Timestamp(end_day))]

    target_weights = np.array([1 / n_files] * n_files)

    # ── 使用修正後的 rebalance_dates，確保第一天就有持倉 ──
    rebalance_dates = _build_rebalance_dates(
        returns.index, Start_Day, end_day, rebalance_frequency
    )

    portfolio_returns = pd.Series(index=returns.index, dtype=float)
    prev_weights      = None

    for i, date in enumerate(rebalance_dates):
        if i < len(rebalance_dates) - 1:
            next_date  = rebalance_dates[i + 1]
            period_idx = returns.loc[date:next_date].index[:-1]
        else:
            period_idx = returns.loc[date:end_day].index

        if len(period_idx) == 0:
            continue

        period_rets = returns.loc[period_idx].dot(target_weights).copy()

        if prev_weights is not None:
            prev_date       = rebalance_dates[i - 1]
            drift_period    = returns.loc[prev_date:date]
            growth_factors  = (1 + drift_period).prod().values
            drifted         = prev_weights * growth_factors
            drifted_weights = drifted / drifted.sum()
            tc_amount       = transaction_cost * np.abs(
                                  target_weights - drifted_weights).sum()
        else:
            tc_amount = transaction_cost * np.abs(target_weights).sum()

        period_rets.iloc[0] -= tc_amount
        portfolio_returns.loc[period_idx] = period_rets
        prev_weights = target_weights.copy()

    portfolio_returns = portfolio_returns[
        portfolio_returns.index >= pd.Timestamp(Start_Day)
    ].dropna()

    # ── 與 risk_free 依 stock_pool_weight : 1 - stock_pool_weight（預設 6:4）混合 ──
    portfolio_returns = _blend_with_risk_free(
        portfolio_returns, risk_free, stock_pool_weight,
        Start_Day, end_day, rebalance_frequency, transaction_cost
    )

    total_return      = Total_Return(portfolio_returns)
    annualized_return = Annualized_Return(portfolio_returns)
    annualized_vol    = Annualized_Volatility(portfolio_returns)
    sharpe_ratio      = Sharpe_Ratio(portfolio_returns, risk_free)
    sortino_ratio     = Sortino_Ratio(portfolio_returns, risk_free)
    mdd               = Max_Drawdown(portfolio_returns)

    os.makedirs('./baselines_results/equal_weight', exist_ok=True)
    portfolio_returns.to_csv(
        './baselines_results/equal_weight/portfolio_returns.csv')
    figure_saving(portfolio_returns,
                  f'Equal Weight Portfolio Return (pool {stock_pool_weight:.0%} / rf {1 - stock_pool_weight:.0%})',
                  './baselines_results/equal_weight/portfolio_returns.png')

    result = (portfolio_returns, total_return, annualized_return,
              annualized_vol, sharpe_ratio, sortino_ratio, mdd)
    if return_weights:
        pool_w = pd.DataFrame([target_weights] * len(rebalance_dates),
                              index=rebalance_dates, columns=returns.columns)
        result += (_to_portfolio_weights(pool_w, stock_pool_weight),)
    return result


# ──────────────────────────────────────────────────────────────
# Auto Reweight Portfolio
# ──────────────────────────────────────────────────────────────

def auto_reweight_portfolio(rebalance_frequency: int = 6,
                            Start_Day: str = "2020-01-01",
                            end_day: str = "2025-10-09",
                            lookback_window: int = 200,
                            return_type: str = "daily",
                            transaction_cost: float = 0.002,
                            optimization_method: str = "constrain",
                            stocks_dir: str = "./dataset/market_index",
                            stock_pool_weight: float = 0.6,
                            risk_free=None,
                            return_weights: bool = False):
    """
    依照 lookback window 的資料，每個 period 自動計算與更新 weight。
    return_weights=True 時，回傳 tuple 最後多一個各再平衡日的 MVO 權重 DataFrame
    （整體投組視角，含 Risk_Free 欄）。
    回傳前過濾到 [Start_Day, end_day]，避免回傳整段歷史資料。

    資產池內部（預設 market_index，目前含 20 檔，已內含 G0Q0 債券指數）
    對整個資產池做一次 MVO 最佳化，再以 stock_pool_weight : 1 - stock_pool_weight
    （預設 6:4）與 risk_free 定期再平衡混合。
    """
    if risk_free is None:
        raise ValueError("auto_reweight_portfolio 需要傳入 risk_free（用於 6:4 無風險資產混合）")

    block_size = {"daily": 1, "weekly": 5, "monthly": 20}.get(return_type, 1)

    portfolio_start = pd.Timestamp(Start_Day)

    stock_tickers = sorted(os.listdir(stocks_dir))

    dfs_stock = [pd.read_csv(f'{stocks_dir}/{t}',
                             parse_dates=['Dates'], index_col='Dates')
                 for t in stock_tickers]

    prices_stock = pd.concat([df['PX_LAST'] for df in dfs_stock], axis=1)
    prices_stock.columns = stock_tickers

    prices_stock = prices_stock.sort_index()[prices_stock.sort_index().index <= end_day]

    rets_stock = prices_stock.pct_change(fill_method=None).dropna()

    start_loc    = rets_stock.index.searchsorted(portfolio_start)
    actual_start = rets_stock.index[start_loc]

    if start_loc < lookback_window:
        raise ValueError(
            f"Start_Day {Start_Day} 之前只有 {start_loc} 個交易日，"
            f"不足 lookback_window={lookback_window}"
        )

    # ── 使用修正後的 rebalance_dates ──
    rebalance_dates = _build_rebalance_dates(
        rets_stock.index, Start_Day, end_day, rebalance_frequency
    )

    all_rets    = rets_stock
    all_tickers = stock_tickers

    weights_history   = pd.DataFrame(index=rebalance_dates,
                                     columns=all_tickers)
    portfolio_returns = pd.Series(index=rets_stock.index, dtype=float)
    prev_weights      = None

    for i, date in enumerate(rebalance_dates):
        date_loc        = rets_stock.index.get_loc(date)
        start_stock_idx = date_loc - lookback_window

        if start_stock_idx < 0:
            continue

        sample_rets_stock = rets_stock.iloc[start_stock_idx:date_loc]
        sample_rets_stock = ((1 + sample_rets_stock).groupby(
            np.arange(len(sample_rets_stock)) // block_size).prod() - 1) * 100

        mu_s    = sample_rets_stock.mean().values
        Sigma_s = sample_rets_stock.cov().values

        ws_opt = solve_weights(mu_s, Sigma_s, len(stock_tickers), optimization_method)

        if ws_opt is None:
            print(f"[WARN] Optimization failed at {date}, using equal weights")
            ws_opt = np.ones(len(stock_tickers)) / len(stock_tickers)

        new_weights = ws_opt
        weights_history.loc[date] = new_weights

        if i < len(rebalance_dates) - 1:
            next_date  = rebalance_dates[i + 1]
            period_idx = all_rets.loc[date:next_date].index[:-1]
        else:
            period_idx = all_rets.loc[date:end_day].index

        period_rets_raw = all_rets.loc[period_idx].dot(new_weights)

        if prev_weights is not None:
            prev_date       = rebalance_dates[i - 1]
            drift_period    = all_rets.loc[prev_date:date]
            growth_factors  = (1 + drift_period).prod().values
            drifted         = prev_weights * growth_factors
            drifted_weights = drifted / drifted.sum()
            tc_amount       = transaction_cost * np.abs(
                                  new_weights - drifted_weights).sum()
        else:
            tc_amount = transaction_cost * np.abs(new_weights).sum()

        period_rets         = period_rets_raw.copy()
        period_rets.iloc[0] -= tc_amount
        portfolio_returns.loc[period_idx] = period_rets
        prev_weights = new_weights

    # ── 只回傳 [Start_Day, end_day] 有值的部分 ──
    portfolio_returns = portfolio_returns[
        (portfolio_returns.index >= pd.Timestamp(Start_Day)) &
        (portfolio_returns.index <= pd.Timestamp(end_day))
    ].dropna()

    # ── 與 risk_free 依 stock_pool_weight : 1 - stock_pool_weight（預設 6:4）混合 ──
    portfolio_returns = _blend_with_risk_free(
        portfolio_returns, risk_free, stock_pool_weight,
        Start_Day, end_day, rebalance_frequency, transaction_cost
    )

    total_return      = Total_Return(portfolio_returns)
    annualized_return = Annualized_Return(portfolio_returns)
    annualized_vol    = Annualized_Volatility(portfolio_returns)
    sharpe_ratio      = Sharpe_Ratio(portfolio_returns, risk_free)
    sortino_ratio     = Sortino_Ratio(portfolio_returns, risk_free)
    mdd               = Max_Drawdown(portfolio_returns)

    os.makedirs('./baselines_results/auto_reweight', exist_ok=True)
    weights_history.to_csv(
        f'./baselines_results/auto_reweight/weights_history'
        f'_rf{rebalance_frequency:g}_lb{lookback_window}.csv')
    portfolio_returns.to_csv(
        f'./baselines_results/auto_reweight/portfolio_returns'
        f'_rf{rebalance_frequency:g}_lb{lookback_window}.csv')
    figure_saving(portfolio_returns,
                  f'Auto Reweight Portfolio Return (pool {stock_pool_weight:.0%} / rf {1 - stock_pool_weight:.0%})',
                  f'./baselines_results/auto_reweight/portfolio'
                  f'_rf{rebalance_frequency:g}_lb{lookback_window}_returns.png')

    result = (portfolio_returns, total_return, annualized_return,
              annualized_vol, sharpe_ratio, sortino_ratio, mdd)
    if return_weights:
        result += (_to_portfolio_weights(weights_history, stock_pool_weight),)
    return result


# ──────────────────────────────────────────────────────────────
# Single Asset Portfolio
# ──────────────────────────────────────────────────────────────

def single_asset_portfolio(StockPoolWeight: float,
                           stock_file: str = "MXWO.csv",
                           Start_Day: str = "2020-01-01",
                           end_day: str = "2025-10-09",
                           rebalance_frequency: int = 1,
                           transaction_cost: float = 0.003,
                           risk_free=None,
                           return_weights: bool = False):
    """
    單一股票標的 + 無風險資產（3個月期零息公債）的 baseline。
    return_weights=True 時，回傳 tuple 最後多一個各再平衡日的目標權重 DataFrame
    （欄位：該標的、Risk_Free）。
    確保 portfolio_returns 從 Start_Day 的第一個實際交易日開始。
    """
    if risk_free is None:
        raise ValueError("single_asset_portfolio 需要傳入 risk_free（load_risk_free_rate 載入的日頻無風險利率序列）")

    bw = 1.0 - StockPoolWeight

    stock_path = os.path.join("./dataset/market_index", stock_file)
    if not os.path.exists(stock_path):
        raise FileNotFoundError(f"找不到股票檔案: {stock_path}")
    stock_ret = pd.read_csv(stock_path, parse_dates=['Dates'],
                            index_col='Dates').sort_index()['PX_LAST'].pct_change().dropna()
    stock_ret = stock_ret[(stock_ret.index >= pd.Timestamp(Start_Day)) &
                          (stock_ret.index <= pd.Timestamp(end_day))]

    risk_free_ret = risk_free[(risk_free.index >= pd.Timestamp(Start_Day)) &
                              (risk_free.index <= pd.Timestamp(end_day))]

    combined = pd.concat([stock_ret, risk_free_ret], axis=1, join='inner').dropna()
    combined.columns = ['stock', 'risk_free']

    target_weights = np.array([StockPoolWeight, bw])

    # ── 使用修正後的 rebalance_dates，確保第一天就有持倉 ──
    rebalance_dates = _build_rebalance_dates(
        combined.index, Start_Day, end_day, rebalance_frequency
    )

    portfolio_returns = pd.Series(index=combined.index, dtype=float)
    prev_weights      = None

    for i, date in enumerate(rebalance_dates):
        if i < len(rebalance_dates) - 1:
            next_date  = rebalance_dates[i + 1]
            period_idx = combined.loc[date:next_date].index[:-1]
        else:
            period_idx = combined.loc[date:end_day].index

        if len(period_idx) == 0:
            continue

        period_rets = combined.loc[period_idx].dot(target_weights).copy()

        if prev_weights is not None:
            prev_date       = rebalance_dates[i - 1]
            drift_period    = combined.loc[prev_date:date]
            growth_factors  = (1 + drift_period).prod().values
            drifted         = prev_weights * growth_factors
            drifted_weights = drifted / drifted.sum()
            tc_amount       = transaction_cost * np.abs(
                                  target_weights - drifted_weights).sum()
        else:
            tc_amount = transaction_cost * np.abs(target_weights).sum()

        period_rets.iloc[0] -= tc_amount
        portfolio_returns.loc[period_idx] = period_rets
        prev_weights = target_weights.copy()

    portfolio_returns = portfolio_returns[
        portfolio_returns.index >= pd.Timestamp(Start_Day)
    ].dropna()

    total_return      = Total_Return(portfolio_returns)
    annualized_return = Annualized_Return(portfolio_returns)
    annualized_vol    = Annualized_Volatility(portfolio_returns)
    sharpe_ratio      = Sharpe_Ratio(portfolio_returns, risk_free)
    sortino_ratio     = Sortino_Ratio(portfolio_returns, risk_free)
    mdd               = Max_Drawdown(portfolio_returns)

    stock_name = stock_file.replace('.csv', '')
    save_dir   = f'./baselines_results/single_asset/{stock_name}__riskfree'
    os.makedirs(save_dir, exist_ok=True)
    portfolio_returns.to_csv(
        f'{save_dir}/portfolio_sw{StockPoolWeight}_returns.csv')
    figure_saving(portfolio_returns, f'Single Asset ({stock_name})',
                  f'{save_dir}/portfolio_sw{StockPoolWeight}_returns.png')

    result = (portfolio_returns, total_return, annualized_return,
              annualized_vol, sharpe_ratio, sortino_ratio, mdd)
    if return_weights:
        pool_w = pd.DataFrame(1.0, index=rebalance_dates, columns=[stock_name])
        result += (_to_portfolio_weights(pool_w, StockPoolWeight),)
    return result


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="financial baselines")
    parser.add_argument("--sw", "--stock_weight", type=float,
                        dest="stock_weight", default=1.0)
    parser.add_argument("--start_day",  type=str, default="2020-01-01")
    parser.add_argument("--end_day",    type=str, default="2025-10-09")
    parser.add_argument("--rf", "--rebalance_freq", type=float, default=6,
                        choices=[0.25, 0.5, 1, 3, 6, 12], dest="rebalance_frequency")
    parser.add_argument("--lb", "--lookback_window", type=int, default=200,
                        choices=[25, 50, 100, 125, 150, 200],
                        dest="lookback_window")
    parser.add_argument("--return_type",  type=str, default="daily",
                        choices=["daily", "weekly", "monthly"])
    parser.add_argument("--MPT_type",     type=str, default="constrain",
                        choices=["constrain", "closed-form"])
    parser.add_argument('--single_stock_file', type=str, default='MXWO.csv')
    parser.add_argument('--pool_weight', type=float, default=0.6,
                        dest='pool_weight',
                        help='Equal Weight 與 MPT 的資產池權重（其餘 1 - pool_weight 為 risk_free），預設 0.6（6:4）')
    parser.add_argument('--transaction_cost',  type=float, default=0.003)
    parser.add_argument('--risk_free_path', type=str,
                        default='./dataset/risk_free_index/US zero coupon bond-rf.csv',
                        help='無風險利率檔（Bloomberg 零息公債殖利率匯出檔）：'
                             '同時作為 Sharpe/Sortino 的超額報酬基準，'
                             '以及三種方法（equal_weight/auto_reweight/single_asset）中無風險資產部位')

    args = parser.parse_args()

    risk_free = load_risk_free_rate(args.risk_free_path)

    ew_ret, *ew_stats = equal_weight_portfolio(
        Start_Day         = args.start_day,
        end_day           = args.end_day,
        transaction_cost  = args.transaction_cost,
        stock_pool_weight = args.pool_weight,
        risk_free         = risk_free,
    )

    mpt_ret, *mpt_stats = auto_reweight_portfolio(
        rebalance_frequency = args.rebalance_frequency,
        Start_Day           = args.start_day,
        end_day             = args.end_day,
        lookback_window     = args.lookback_window,
        return_type         = args.return_type,
        optimization_method = args.MPT_type,
        transaction_cost    = args.transaction_cost,
        stock_pool_weight   = args.pool_weight,
        risk_free            = risk_free,
    )

    single_ret, *single_stats = single_asset_portfolio(
        StockPoolWeight  = args.stock_weight,
        stock_file       = args.single_stock_file,
        Start_Day        = args.start_day,
        end_day          = args.end_day,
        transaction_cost = args.transaction_cost,
        risk_free        = risk_free,
    )

    stock_label = args.single_stock_file.replace('.csv', '')
    metrics_idx = ["Total Return", "Annualized Return", "Annualized Volatility",
                   "Sharpe Ratio", "Sortino Ratio", "Max Drawdown"]

    summary = pd.DataFrame({
        "Equal Weight":         ew_stats,
        "MPT":                  mpt_stats,
        f"Single ({stock_label})": single_stats,
    }, index=metrics_idx)

    print(summary)
    summary.to_csv(
        f'./baselines_results/portfolio'
        f'_rf{args.rebalance_frequency:g}_lb{args.lookback_window}_summary.csv'
    )

    plt.figure(figsize=(12, 6))
    figure_plotting(ew_ret,     'Equal Weight')
    figure_plotting(mpt_ret,    'MPT')
    figure_plotting(single_ret, f'Single ({stock_label})')
    plt.title('Portfolio Performance Comparison')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.savefig(
        f"./baselines_results/portfolio"
        f"_rf{args.rebalance_frequency:g}_lb{args.lookback_window}"
        f"_baseline_comparison.png",
        dpi=300, bbox_inches="tight"
    )
    plt.close()