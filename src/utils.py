import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def Total_Return(portfolio_ret):
    total_return = (1 + portfolio_ret).prod() - 1
    # print(f"Total Return: {total_return*100:.4f}%")
    return round(total_return, 4)

def Annualized_Return(portfolio_ret):
    total_return = (1 + portfolio_ret).prod() - 1
    n_years = (portfolio_ret.index[-1] - portfolio_ret.index[0]).days / 365.25
    annualized_return = (1 + total_return) ** (1/n_years) - 1
    # print(f"Annualized Return: {annualized_return*100:.4f}%")
    return round(annualized_return, 4)

def Annualized_Volatility(portfolio_ret):
    annualized_vol = np.std(portfolio_ret) * np.sqrt(252)
    # print(f"Annualized Volatility: {annualized_vol*100:.4f}%")
    return round(annualized_vol, 4)

def _excess_return(portfolio_ret, risk_free=None):
    """扣掉無風險利率，得到超額報酬；risk_free=None 時視為 0（原本做法）。"""
    if risk_free is None:
        return portfolio_ret
    rf_aligned = risk_free.reindex(portfolio_ret.index).ffill().bfill().fillna(0)
    return portfolio_ret - rf_aligned

def Sharpe_Ratio(portfolio_ret, risk_free=None):
    excess_ret = _excess_return(portfolio_ret, risk_free)
    sharpe_ratio = (excess_ret.mean() / excess_ret.std()) * np.sqrt(252)
    # print(f"Sharpe Ratio:{sharpe_ratio:.4f}")
    return round(sharpe_ratio, 4)

def Sortino_Ratio(portfolio_ret, risk_free=None):
    excess_ret = _excess_return(portfolio_ret, risk_free)
    downside_ret = excess_ret[excess_ret < 0]
    sortino_ratio = (excess_ret.mean() / downside_ret.std()) * np.sqrt(252)
    # print(f"Sortino Ratio:{sortino_ratio:.4f}")
    return round(sortino_ratio, 4)

def load_risk_free_rate(csv_path, maturity_col=1):
    """
    載入 Bloomberg 匯出的零息公債殖利率檔（US zero coupon bond-rf.csv），
    換算成日頻無風險利率（小數，可直接跟每日報酬相減）。

    maturity_col: 天期欄位在資料區的位置（1-indexed，預設 1 = 3 個月期 I02503M Index，
                   Sharpe/Sortino 最常用的無風險利率慣例天期）。
    """
    df = pd.read_csv(csv_path, skiprows=6, header=None,
                      na_values=['#N/A N/A', '#NAME?'])
    dates = pd.to_datetime(df[0], errors='coerce')
    annual_yield_pct = pd.to_numeric(df[maturity_col], errors='coerce')
    rf = pd.Series(annual_yield_pct.values, index=dates)
    rf = rf[rf.index.notna() & rf.notna()].sort_index()
    daily_rf = (1 + rf / 100) ** (1 / 252) - 1
    return daily_rf

def load_price_index_returns(csv_path):
    """
    載入單一價格指數檔（如 G0Q0_Index.csv），回傳日報酬（小數）。
    用於 risk_free_source='g0q0' 時，把 G0Q0 公債指數的日報酬當作無風險利率的替代來源。
    """
    df = pd.read_csv(csv_path, parse_dates=['Dates'], index_col='Dates').sort_index()
    return df['PX_LAST'].pct_change().dropna()

def Max_Drawdown(portfolio_ret):
    cum_value = (1 + portfolio_ret).cumprod()
    rolling_max = cum_value.cummax()
    mdd = ((cum_value - rolling_max) / rolling_max).min()
    # print(f"Max Drawdown:{mdd:.4f}")
    return round(mdd, 4)

def figure_saving(portfolio_ret, portfolio_type, save_path):
    cum_value = (1 + portfolio_ret).cumprod()
    plt.figure(figsize=(12,6))
    plt.plot(cum_value, label = portfolio_type)
    plt.title('Portfolio Performance')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return 0

def figure_plotting(portfolio_ret, label):
    cum_value = (1 + portfolio_ret).cumprod()
    plt.plot(cum_value, label=label)

    return 0

# def calculate_cost_to_alpha_ratio(portfolio_net_ret, turnover_costs):
#     """
#     計算成本損耗比與相關穩健性指標
#     Args:
#         portfolio_net_ret (pd.Series): 扣除成本後的每日淨報酬
#         turnover_costs (pd.Series): 每日產生的交易成本數值 (非負值)
#     """
#     # total_net_return = portfolio_net_ret.sum()
#     # total_cost = turnover_costs.sum()
    
#     # 原始收益 (Gross Return) = 淨收益 + 總成本
#     # gross_return = total_net_return + total_cost
#     daily_gross = portfolio_net_ret + turnover_costs
#     total_net_return   = (1 + portfolio_net_ret).prod() - 1
#     total_gross_return = (1 + daily_gross).prod() - 1
#     total_cost         = turnover_costs.sum()          # 成本本身直接加總即可

#     # 成本損耗比 (Ratio)
#     # 若 gross_return 為負或接近 0，此比例會失去意義，需做安全性檢查
#     if total_gross_return > 1e-8:
#         ratio = total_cost / total_gross_return
#     else:
#         ratio = float('inf') # 代表成本遠超收益，或根本沒賺錢
        
#     return {
#         'total_net_return':     round(total_net_return,   4),
#         'total_gross_return':   round(total_gross_return, 4),
#         'total_cost':           round(total_cost,         6),
#         'cost_to_alpha_ratio':  round(ratio, 4) if not np.isnan(ratio) else float('nan')
#     }

def calculate_cost_to_alpha_ratio(portfolio_net_ret, turnover_costs):
    
    daily_gross = portfolio_net_ret + turnover_costs

    # ── 複利口徑（和 total_net/gross_return 一致）──────────────
    total_net_return   = (1 + portfolio_net_ret).prod() - 1
    total_gross_return = (1 + daily_gross).prod() - 1
    
    # cost 也用複利差額，而不是直接加總
    # gross 複利 - net 複利 = 真正被成本侵蝕的複利報酬
    total_cost_compounded = total_gross_return - total_net_return

    # ── 加總口徑（直觀，但忽略複利效應）──────────────────────────
    total_cost_simple = turnover_costs.sum()

    if total_gross_return > 1e-8:
        ratio = total_cost_compounded / total_gross_return
    else:
        ratio = float('inf')

    return {
        'total_net_return':        round(total_net_return,        4),
        'total_gross_return':      round(total_gross_return,      4),
        'total_cost_compounded':   round(total_cost_compounded,   6),  # 複利口徑
        'total_cost_simple':       round(total_cost_simple,       6),  # 加總口徑（供參考）
        'cost_to_alpha_ratio':     round(ratio, 4)
            if not np.isnan(ratio) else float('nan')
    }