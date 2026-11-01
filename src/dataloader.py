"""
dataloader.py
數據載入和預處理模組
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict
import os
import argparse
from sklearn.covariance import LedoitWolf


class FinancialDataset(Dataset):
    """
    金融時間序列數據集，使用滾動窗口方法
    
    每個樣本包含：
    - X: 過去60天的收益率 (lookback window)
    - mu_star: 下一天的實際收益率 (ground truth)
    - Sigma: 從lookback window計算的協方差矩陣
    """
    def __init__(self, returns_df: pd.DataFrame, lookback_days: int = 60,
                 start_idx: int = 0, end_idx: int = None):
        """
        Args:
            returns_df: DataFrame，shape為(T, N)，T=時間，N=資產數量
            lookback_days: 歷史回溯天數
            start_idx: 此數據集的起始索引
            end_idx: 此數據集的結束索引
        """
        self.returns = returns_df.values.astype(np.float32)
        self.lookback_days = lookback_days
        self.n_assets = self.returns.shape[1]
        
        # 確定有效樣本範圍
        if end_idx is None:
            end_idx = len(self.returns)
        
        # 需要lookback_days的歷史數據 + 1天的未來數據
        self.start_idx = max(lookback_days, start_idx)
        self.end_idx = min(len(self.returns) - 1, end_idx)
        
        # 預先計算所有有效索引
        self.valid_indices = list(range(self.start_idx, self.end_idx))
        
        print(f"Dataset created: {len(self.valid_indices)} samples, "
              f"{self.n_assets} assets")
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        返回一個包含歷史收益率、協方差矩陣和未來收益率的樣本
        """
        actual_idx = self.valid_indices[idx]
        
        # 歷史收益率（過去60天）
        start = actual_idx - self.lookback_days
        end = actual_idx
        X = self.returns[start:end, :]  # (60, n_assets)

        # X_mean = X.mean(axis=1, keepdims=True)    # (lookback, 1)
        # X_std  = X.std(axis=1, keepdims=True) + 1e-8
        # X_demeaned = (X - X_mean)/X_std            # (lookback, n_assets)
        
        # Ground truth: 下一天的收益率
        mu_star = self.returns[end, :]  # (n_assets,)
        
        # 從歷史數據計算協方差矩陣
        # 添加小的正則化項確保正定性
        # Sigma = np.cov(X.T) + np.eye(self.n_assets) * 1e-4

        # 使用shrinkage method estismator LedoitWolf
        Sigma = LedoitWolf().fit(X).covariance_ 

        

        
        # print(f"Index: {actual_idx} | X_last_day_return: {X[-1, 0]} | mu_star_return: {mu_star[0]}")
        
        return {
            'X': torch.tensor(X, dtype=torch.float32),
            'mu_star': torch.tensor(mu_star, dtype=torch.float32),
            'Sigma': torch.tensor(Sigma, dtype=torch.float32),
            'idx': actual_idx
        }

def prepare_data(price_df: pd.DataFrame, 
                train_ratio: float = 0.7,
                val_ratio: float = 0.15,
                test_ratio: float = 0.15,
                lookback_days: int = 60,
                test_year: int = None) -> Tuple[pd.DataFrame, Dict]:

    returns_df = price_df.pct_change().dropna()

    risk_free_cols = [col for col in returns_df.columns 
                     if 'risk_free' in col.lower() or col.upper() == 'RF']
    if risk_free_cols:
        print(f"Removing risk-free asset columns: {risk_free_cols}")
        returns_df = returns_df.drop(columns=risk_free_cols)

    total_samples = len(returns_df)
    usable_start  = lookback_days

    if test_year is not None:
        # ── 指定 test year 模式 ──
        # test set = 指定年份整年
        # val  set = 指定年份的前一年
        # train set = val set 之前的所有資料（從 usable_start 開始）

        test_start_date = pd.Timestamp(f'{test_year}-01-01')
        test_end_date   = pd.Timestamp(f'{test_year}-12-31')
        val_start_date  = pd.Timestamp(f'{test_year - 1}-01-01')
        val_end_date    = pd.Timestamp(f'{test_year - 1}-12-31')

        # 找對應的整數 index
        # 使用 searchsorted 找最近的交易日
        test_start_idx = returns_df.index.searchsorted(test_start_date)
        test_end_idx   = returns_df.index.searchsorted(test_end_date, side='right') - 1
        val_start_idx  = returns_df.index.searchsorted(val_start_date)
        val_end_idx    = returns_df.index.searchsorted(val_end_date, side='right') - 1

        # train 結束於 val 開始之前
        train_end_idx = val_start_idx

        # 確保 train 起點有足夠的 lookback
        train_start_idx = usable_start

        split_info = {
            'train_start': train_start_idx,
            'train_end':   train_end_idx,
            'val_start':   val_start_idx,
            'val_end':     val_end_idx,
            'test_start':  test_start_idx,
            'test_end':    test_end_idx,
            'total_samples': total_samples,
            'n_assets':      returns_df.shape[1],
            'asset_names':   list(returns_df.columns),
            'lookback_days': lookback_days,
            'test_year':     test_year
        }

        print(f"\n{'='*60}")
        print(f"Data Preparation Summary (test_year={test_year})")
        print(f"{'='*60}")
        print(f"Date range: {returns_df.index[0]} to {returns_df.index[-1]}")
        print(f"Total samples: {total_samples}")
        print(f"Number of risky assets: {split_info['n_assets']}")
        print(f"\nData Split:")
        print(f"  Training:   idx {train_start_idx} to {train_end_idx} "
              f"({train_end_idx - train_start_idx} samples) "
              f"[{returns_df.index[train_start_idx].date()} → {returns_df.index[train_end_idx-1].date()}]")
        print(f"  Validation: idx {val_start_idx} to {val_end_idx} "
              f"({val_end_idx - val_start_idx} samples) "
              f"[{returns_df.index[val_start_idx].date()} → {returns_df.index[val_end_idx].date()}]")
        print(f"  Testing:    idx {test_start_idx} to {test_end_idx} "
              f"({test_end_idx - test_start_idx} samples) "
              f"[{returns_df.index[test_start_idx].date()} → {returns_df.index[test_end_idx].date()}]")
        print(f"{'='*60}\n")

    else:
        # ── 原始比例切割模式 ──
        usable_samples = total_samples - usable_start - 1

        train_end = usable_start + int(usable_samples * train_ratio)
        val_end   = train_end   + int(usable_samples * val_ratio)
        test_end  = total_samples - 1

        split_info = {
            'train_start': usable_start,
            'train_end':   train_end,
            'val_start':   train_end,
            'val_end':     val_end,
            'test_start':  val_end,
            'test_end':    test_end,
            'total_samples': total_samples,
            'n_assets':      returns_df.shape[1],
            'asset_names':   list(returns_df.columns),
            'lookback_days': lookback_days,
            'test_year':     None
        }

        print(f"\n{'='*60}")
        print(f"Data Preparation Summary")
        print(f"{'='*60}")
        print(f"Date range: {returns_df.index[0]} to {returns_df.index[-1]}")
        print(f"Total samples: {total_samples}")
        print(f"Number of risky assets: {split_info['n_assets']}")
        print(f"\nData Split:")
        print(f"  Training:   {usable_start} to {train_end} "
              f"({train_end - usable_start} samples, {train_ratio*100:.0f}%)")
        print(f"  Validation: {train_end} to {val_end} "
              f"({val_end - train_end} samples, {val_ratio*100:.0f}%)")
        print(f"  Testing:    {val_end} to {test_end} "
              f"({test_end - val_end} samples, {test_ratio*100:.0f}%)")
        print(f"{'='*60}\n")

    return returns_df, split_info

def create_dataloaders(returns_df: pd.DataFrame,
                      split_info: Dict,
                      lookback_days: int = 60,
                      batch_size: int = 16,
                      num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    創建訓練、驗證和測試集的DataLoader
    
    Args:
        returns_df: 收益率DataFrame
        split_info: 從prepare_data返回的切分信息字典
        lookback_days: 回溯窗口大小
        batch_size: 批次大小
        num_workers: 數據載入的工作進程數（Windows建議設為0）
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # 創建數據集
    train_dataset = FinancialDataset(
        returns_df=returns_df,
        lookback_days=lookback_days,
        start_idx=split_info['train_start'],
        end_idx=split_info['train_end']
    )
    
    val_dataset = FinancialDataset(
        returns_df=returns_df,
        lookback_days=lookback_days,
        start_idx=split_info['val_start'],
        end_idx=split_info['val_end']
    )
    
    test_dataset = FinancialDataset(
        returns_df=returns_df,
        lookback_days=lookback_days,
        start_idx=split_info['test_start'],
        end_idx=split_info['test_end']
    )
    
    # 創建dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # 訓練時不打亂
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # 驗證時不打亂
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # 測試時不打亂
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"DataLoaders created:")
    print(f"  Train:      {len(train_loader)} batches "
          f"({len(train_dataset)} samples)")
    print(f"  Validation: {len(val_loader)} batches "
          f"({len(val_dataset)} samples)")
    print(f"  Test:       {len(test_loader)} batches "
          f"({len(test_dataset)} samples)")
    print(f"  Batch size: {batch_size}\n")
    
    return train_loader, val_loader, test_loader


def load_price_data(file_path: str = "./dataset/market_index/", date_column: str = None) -> pd.DataFrame:
    """
    從CSV文件載入價格數據
    
    預期格式：
    - 第一列（或指定列）: 日期
    - 其他列: 資產價格
    
    Args:
        file_path: CSV文件路徑
        date_column: 日期列名稱（如果為None，使用第一列）
    
    Returns:
        price_df: 包含DatetimeIndex和資產價格的DataFrame
    """
    # 依檔名排序，確保資產欄位順序跨作業系統一致（os.listdir 不保證順序），
    # 模型輸出的第 i 維才會對應到訓練時同一檔資產
    stock_tickers = sorted(f for f in os.listdir(file_path) if f.endswith('.csv'))
    dfs_stock = [pd.read_csv(os.path.join(file_path, t), parse_dates=['Dates'], \
                             index_col='Dates') for t in stock_tickers]
    df = pd.concat([df['PX_LAST'] for df in dfs_stock], axis=1)
    df.columns = stock_tickers
    
    # 確保datetime索引
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # 按日期排序
    df = df.sort_index()
    
    # 移除任何NaN值
    initial_len = len(df)
    df = df.dropna()
    if len(df) < initial_len:
        print(f"Warning: Removed {initial_len - len(df)} rows with NaN values")
    
    print(f"\n{'='*60}")
    print(f"Price Data Loaded")
    print(f"{'='*60}")
    print(f"File: {file_path}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Number of trading days: {len(df)}")
    print(f"Number of assets: {df.shape[1]}")
    print(f"\nAsset columns: {list(df.columns)}")
    print(f"{'='*60}\n")
    
    return df


# 測試和使用範例
if __name__ == "__main__":
    """
    使用範例
    """
    
    # 方法1: 載入你的真實數據
    price_df = load_price_data(f'./dataset/market_index/')

    # 方法2: 創建示範數據（用於測試）
    # print("Creating synthetic data for demonstration...\n")
    # dates = pd.date_range('2017-01-01', '2025-12-31', freq='D')
    # n_assets = 19
    # np.random.seed(42)
    
    # # 模擬帶有相關結構的價格
    # returns = np.random.randn(len(dates)-1, n_assets) * 0.01
    # prices = np.zeros((len(dates), n_assets))
    # prices[0] = 100
    
    # for i in range(1, len(dates)):
    #     prices[i] = prices[i-1] * (1 + returns[i-1])
    
    # price_df = pd.DataFrame(
    #     prices, 
    #     index=dates, 
    #     columns=[f'Asset_{i}' for i in range(n_assets)]
    # )
    
    # 準備數據
    returns_df, split_info = prepare_data(
        price_df=price_df,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        lookback_days=60
    )
    """
    這一段可以觀察training set, validation set, testing set等資料切割之間的開始與結束日期、mean與std等指標
    print(f"Train: {returns_df.index[split_info['train_start']]} "
      f"→ {returns_df.index[split_info['train_end']-1]}")
    print(f"Val:   {returns_df.index[split_info['val_start']]} "
        f"→ {returns_df.index[split_info['val_end']-1]}")
    print(f"Test:  {returns_df.index[split_info['test_start']]} "
        f"→ {returns_df.index[split_info['test_end']]}")
    

    train_returns = returns_df.iloc[split_info['train_start']:split_info['train_end']]
    val_returns   = returns_df.iloc[split_info['val_start']:split_info['val_end']]
    test_returns   = returns_df.iloc[split_info['test_start']:split_info['test_end']]

    print("Train returns stats:")
    print(f"  Mean:  {train_returns.mean().mean():.6f}")
    print(f"  Std:   {train_returns.std().mean():.6f}")

    print("Val returns stats:")
    print(f"  Mean:  {val_returns.mean().mean():.6f}")
    print(f"  Std:   {val_returns.std().mean():.6f}")

    print("Test returns stats:")
    print(f"  Mean:  {test_returns.mean().mean():.6f}")
    print(f"  Std:   {test_returns.std().mean():.6f}")
    """
    # 創建dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        returns_df=returns_df,
        split_info=split_info,
        lookback_days=60,
        batch_size=16,
        num_workers=0  # Windows使用0，Linux/Mac可以使用4
    )
    
    # 測試一個batch
    print("Testing one batch from training set:")
    for batch in train_loader:
        print(f"\nBatch contents:")
        print(f"  X shape:        {batch['X'].shape}")  # (batch_size, 60, n_assets)
        print(f"  mu_star shape:  {batch['mu_star'].shape}")  # (batch_size, n_assets)
        print(f"  Sigma shape:    {batch['Sigma'].shape}")  # (batch_size, n_assets, n_assets)
        print(f"  Indices:        {batch['idx'][:5]}...")  # First 5 indices
        
        # 驗證數據範圍
        print(f"\n  X range:        [{batch['X'].min():.4f}, {batch['X'].max():.4f}]")
        print(f"  mu_star range:  [{batch['mu_star'].min():.4f}, {batch['mu_star'].max():.4f}]")
        print(f"  Sigma diagonal: {torch.diagonal(batch['Sigma'][0]).mean():.6f} (avg variance)")
        
        break
    
    print("\n" + "="*60)
    print("DataLoader setup complete and tested successfully!")
    print("="*60)