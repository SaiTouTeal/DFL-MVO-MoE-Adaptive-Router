"""
multi_seeds_training.py
────────────────────────
依序用不同 random seed 訓練多個獨立的 DFLMVOModel。
每個 seed 完全獨立訓練，不共享梯度（與 multi_experts_training.py 的 joint training 不同）。

Checkpoint 結構:
    checkpoints/
    └── {model_type}/
        └── testYear{test_year}/
            └── seed{seed}/
                ├── checkpoint_best.pt
                ├── config.json
                └── training_history.png

Usage:
    python multi_seeds_training.py --model_type mlp --test_year 2022
    python multi_seeds_training.py --model_type lstm --test_year 2022 2023 --seeds 0 1 2 3 4
"""

import matplotlib
matplotlib.use('Agg')
import os
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm
from typing import Dict, Tuple
import matplotlib.pyplot as plt
from agents import DFLMVOModel
from dataloader import create_dataloaders, prepare_data, load_price_data
import argparse
import time
import sys
import random
import torch.nn as nn

# ──────────────────────────────────────────────────────────────
# Logger / Seed / EarlyStopping / ModelCheckpoint
# (與 training.py 完全一致，直接複製以保持獨立可執行)
# ──────────────────────────────────────────────────────────────

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


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[OK] Random seed set to {seed}")


class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 1e-6):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter   = 0
        return self.early_stop


class ModelCheckpoint:
    def __init__(self, save_dir: str, metric: str = 'loss'):
        self.save_dir    = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.metric      = metric
        self.best_metric = float('inf')
        self.best_epoch  = 0

    def save(self, model, optimizer, scheduler, epoch: int, metrics: Dict,
             is_best: bool = False) -> None:
        checkpoint = {
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': (scheduler.state_dict()
                                     if hasattr(scheduler, 'state_dict') else None),
            'metrics':              metrics,
        }

        # Save every 10 epochs after epoch 100
        if epoch % 10 == 0 and epoch >= 100:
            torch.save(checkpoint, self.save_dir / f'checkpoint_epoch_{epoch}.pt')
            print(f"[OK] Saved checkpoint at epoch {epoch}")

        if is_best:
            torch.save(checkpoint, self.save_dir / 'checkpoint_best.pt')
            self.best_metric = metrics['val_loss']
            self.best_epoch  = epoch
            print(f"[OK] Saved best model at epoch {epoch} "
                  f"with val_loss={metrics['val_loss']:.6f}")


# ──────────────────────────────────────────────────────────────
# Optimizer & Scheduler (與 training.py 完全一致)
# ──────────────────────────────────────────────────────────────

def _build_optimizer_and_scheduler(model, config: Dict, model_type: str, max_epochs: int):
    lr = config['learning_rate']
    wd = config['weight_decay']

    if model_type == 'mlp':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10)
        scheduler_type = 'plateau'

    elif model_type == 'lstm':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10)
        scheduler_type = 'plateau'

    elif model_type == 'transformer':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        scheduler_type = 'plateau'

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    print(f"[OK] model_type={model_type} → "
          f"optimizer={optimizer.__class__.__name__}, "
          f"scheduler={scheduler.__class__.__name__}")
    return optimizer, scheduler, scheduler_type


# ──────────────────────────────────────────────────────────────
# Train / Validate (與 training.py 完全一致)
# ──────────────────────────────────────────────────────────────

def train_one_epoch(model, train_loader, optimizer, device,
                    epoch: int, log_interval: int = 50) -> Dict:
    model.train()
    model.reset_w_prev()
    total_loss = total_mvo = total_mse = total_regret = 0
    n_batches  = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, batch in enumerate(pbar):
        X       = batch['X'].to(device)
        mu_star = batch['mu_star'].to(device)
        Sigma   = batch['Sigma'].to(device)

        optimizer.zero_grad()
        outputs = model(X, Sigma, mu_star)
        outputs['loss'].backward()

        if model.model_type == 'lstm':
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        elif model.model_type == 'transformer':
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        ld = outputs['loss_dict']
        total_loss   += ld['total_loss']
        total_mvo    += ld['mvo_loss']
        total_mse    += ld['mse_loss']
        total_regret += ld['regret']
        n_batches    += 1

        if batch_idx % log_interval == 0:
            pbar.set_postfix({
                'loss': f"{ld['total_loss']:.4f}",
                'mvo':  f"{ld['mvo_loss']:.4f}",
                'mse':  f"{ld['mse_loss']:.6f}",
            })

    return {
        'train_loss':     total_loss   / n_batches,
        'train_mvo_loss': total_mvo    / n_batches,
        'train_mse_loss': total_mse    / n_batches,
        'train_regret':   total_regret / n_batches,
    }


def validate(model, val_loader, device, epoch: int) -> Dict:
    model.eval()
    model.reset_w_prev()
    total_loss = total_mvo = total_mse = total_regret = 0
    n_batches  = 0
    all_mu_hat, all_mu_star, all_w_hat, all_w_star = [], [], [], []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        for batch in pbar:
            X       = batch['X'].to(device)
            mu_star = batch['mu_star'].to(device)
            Sigma   = batch['Sigma'].to(device)

            outputs = model(X, Sigma, mu_star)
            ld      = outputs['loss_dict']
            total_loss   += ld['total_loss']
            total_mvo    += ld['mvo_loss']
            total_mse    += ld['mse_loss']
            total_regret += ld['regret']
            n_batches    += 1

            all_mu_hat.append(outputs['mu_hat'].cpu())
            all_mu_star.append(mu_star.cpu())
            all_w_hat.append(outputs['w_hat'].cpu())
            all_w_star.append(outputs['w_star'].cpu())

            pbar.set_postfix({
                'loss': f"{ld['total_loss']:.4f}",
                'mvo':  f"{ld['mvo_loss']:.4f}",
                'mse':  f"{ld['mse_loss']:.6f}",
            })

    all_mu_hat  = torch.cat(all_mu_hat,  dim=0)
    all_mu_star = torch.cat(all_mu_star, dim=0)
    all_w_hat   = torch.cat(all_w_hat,   dim=0)

    metrics = {
        'val_loss':     total_loss   / n_batches,
        'val_mvo_loss': total_mvo    / n_batches,
        'val_mse_loss': total_mse    / n_batches,
        'val_regret':   total_regret / n_batches,
        'avg_portfolio_concentration': (
            (all_w_hat > 0.01).sum(dim=1).float().mean().item()),
        'prediction_correlation': torch.corrcoef(
            torch.stack([all_mu_hat.flatten(), all_mu_star.flatten()])
        )[0, 1].item(),
    }
    return metrics


# ──────────────────────────────────────────────────────────────
# Plot (與 training.py 完全一致)
# ──────────────────────────────────────────────────────────────

def plot_training_history(history_df: pd.DataFrame, save_dir: str):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    def _plot(ax, col_train, col_val, ylabel, title):
        ax.plot(history_df['epoch'], history_df[col_train],
                label='Train', marker='o')
        if col_val in history_df.columns:
            vd = history_df.dropna(subset=[col_val])
            ax.plot(vd['epoch'], vd[col_val], label='Val', marker='s')
        ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)

    _plot(axes[0, 0], 'train_loss',     'val_loss',     'Total Loss',  'Total Loss vs Epoch')
    _plot(axes[0, 1], 'train_mse_loss', 'val_mse_loss', 'MSE Loss',    'MSE Loss vs Epoch')
    _plot(axes[1, 0], 'train_regret',   'val_regret',   'Regret',      'Regret vs Epoch')

    if 'avg_portfolio_concentration' in history_df.columns:
        ax = axes[1, 1]
        vd = history_df.dropna(subset=['avg_portfolio_concentration'])
        ax.plot(vd['epoch'], vd['avg_portfolio_concentration'],
                marker='s', color='green')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Avg # Assets in Portfolio')
        ax.set_title('Portfolio Concentration vs Epoch')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(save_dir) / 'training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Training history plot saved to {save_dir}/training_history.png")


# ──────────────────────────────────────────────────────────────
# Single seed training
# ──────────────────────────────────────────────────────────────

def run_single_seed(seed: int, test_year: int, args, device: str) -> Dict:
    """完整訓練一個 seed，回傳結果 dict。"""
    set_seed(seed)

    model_type = args.model_type
    # ── checkpoint / runs / logs 路徑 ─────────────────────────
    save_dir = Path(args.base_checkpoint_dir) / model_type / f"testYear{test_year}"  / f"seed{seed}"
    runs_dir = Path(args.base_runs_dir)       / model_type / f"testYear{test_year}"  / f"seed{seed}"
    logs_dir = Path(args.base_logs_dir)       / model_type / f"testYear{test_year}"  / f"seed{seed}"
    for d in (save_dir, runs_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── per-seed logger ───────────────────────────────────────
    timestamp  = time.strftime("%Y%m%d_%H%M%S")
    log_path   = logs_dir / f"console_{timestamp}.log"
    old_stdout = sys.stdout
    sys.stdout = Logger(str(log_path))

    print(f"--- Training | model={model_type} | test_year={test_year} | "
          f"seed={seed} | device={device} ---")

    config = {
        'model_type':               model_type,
        'epochs':                   args.epochs,
        'learning_rate':            args.learning_rate,
        'weight_decay':             args.weight_decay,
        'batch_size':               args.batch_size,
        'val_interval':             args.val_interval,
        'log_interval':             50,
        'early_stopping_patience':  args.early_stopping_patience,
        'lookback_days':            args.lookback_days,
        'risk_aversion':            args.risk_aversion,
        'alpha':                    args.alpha,
        'max_weight':               args.max_weight,
        'test_year':                test_year,
        'transaction_cost':         args.transaction_cost,
        'seed':                     seed,
        'n_assets':                 None,   # filled after prepare_data
    }

    # ── data ─────────────────────────────────────────────────
    price_df = load_price_data(args.data_path)
    returns_df, split_info = prepare_data(
        price_df,
        lookback_days = args.lookback_days,
        test_year     = test_year,
    )
    train_loader, val_loader, _ = create_dataloaders(
        returns_df, split_info,
        lookback_days = args.lookback_days,
        batch_size    = args.batch_size,
    )
    config['n_assets'] = split_info['n_assets']
    print(f"  n_assets={split_info['n_assets']} | "
          f"train={split_info['train_end']-split_info['train_start']} days | "
          f"val={split_info['val_end']-split_info['val_start']} days")

    # ── model ─────────────────────────────────────────────────
    model = DFLMVOModel(
        n_assets         = split_info['n_assets'],
        lookback_days    = args.lookback_days,
        lambda_risk      = args.risk_aversion,
        max_weight       = args.max_weight,
        alpha            = args.alpha,
        transaction_cost = args.transaction_cost,
        device           = device,
        model_type       = model_type,
    ).to(device)

    # ── optimizer & scheduler ─────────────────────────────────
    optimizer, scheduler, scheduler_type = _build_optimizer_and_scheduler(
        model      = model,
        config     = config,
        model_type = model_type,
        max_epochs = args.epochs,
    )

    # ── training state ────────────────────────────────────────
    ckpt_manager  = ModelCheckpoint(save_dir=str(save_dir))
    early_stopper = EarlyStopping(patience=args.early_stopping_patience)
    writer        = SummaryWriter(log_dir=str(runs_dir / f"exp_{timestamp}"))

    best_val_loss = float('inf')
    history       = []

    print(f"\n{'='*70}")
    print(f"Starting training for {args.epochs} epochs")
    print(f"alpha={args.alpha} | risk_aversion={args.risk_aversion} | "
          f"max_weight={args.max_weight} | tc={args.transaction_cost}")
    print(f"{'='*70}\n")

    # ── epoch loop ────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):

        train_metrics = train_one_epoch(
            model        = model,
            train_loader = train_loader,
            optimizer    = optimizer,
            device       = device,
            epoch        = epoch,
            log_interval = 50,
        )

        writer.add_scalar('Train/Total_Loss', train_metrics['train_loss'],     epoch)
        writer.add_scalar('Train/MVO_Loss',   train_metrics['train_mvo_loss'], epoch)
        writer.add_scalar('Train/MSE_Loss',   train_metrics['train_mse_loss'], epoch)

        if epoch % args.val_interval == 0:
            val_metrics = validate(model, val_loader, device, epoch)

            writer.add_scalar('Valid/Total_Loss', val_metrics['val_loss'],     epoch)
            writer.add_scalar('Valid/MVO_Loss',   val_metrics['val_mvo_loss'], epoch)
            writer.add_scalar('Valid/MSE_Loss',   val_metrics['val_mse_loss'], epoch)

            if scheduler_type == 'plateau':
                scheduler.step(val_metrics['val_loss'])
            else:
                scheduler.step()

            epoch_metrics = {**train_metrics, **val_metrics, 'epoch': epoch}
            history.append(epoch_metrics)

            is_best = val_metrics['val_loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['val_loss']

            ckpt_manager.save(
                model     = model,
                optimizer = optimizer,
                scheduler = scheduler,
                epoch     = epoch,
                metrics   = epoch_metrics,
                is_best   = is_best,
            )

            print(f"\nEpoch {epoch}/{args.epochs} | "
                  f"train={train_metrics['train_loss']:.6f} | "
                  f"val={val_metrics['val_loss']:.6f} | "
                  f"best={best_val_loss:.6f}")
            print(f"  MSE  train={train_metrics['train_mse_loss']:.6f} | "
                  f"val={val_metrics['val_mse_loss']:.6f}")
            print(f"  Regret train={train_metrics['train_regret']:.6f} | "
                  f"val={val_metrics['val_regret']:.6f}")
            print(f"  Concentration={val_metrics['avg_portfolio_concentration']:.2f} | "
                  f"PredCorr={val_metrics['prediction_correlation']:.4f}")

            if early_stopper(val_metrics['val_loss']):
                print(f"\n⚠ Early stopping at epoch {epoch}")
                break

        else:
            history.append({**train_metrics, 'epoch': epoch})

    writer.close()

    # ── save history & config ─────────────────────────────────
    history_df = pd.DataFrame(history)
    history_df.to_csv(save_dir / 'training_history.csv', index=False)

    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    plot_training_history(history_df, str(save_dir))

    print(f"\n[seed={seed}] Done. best_val_loss={best_val_loss:.6f} | "
          f"saved to {save_dir}")

    sys.stdout = old_stdout

    return {
        'seed':          seed,
        'test_year':     test_year,
        'best_val_loss': best_val_loss,
        'save_dir':      str(save_dir),
    }


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sequential multi-seed training')

    # ── experiment axes ────────────────────────────────────────
    parser.add_argument('--model_type', type=str, default='mlp',
                        choices=['mlp', 'lstm', 'transformer'])
    parser.add_argument('--test_year',  type=int, nargs='+',
                        default=[2022], choices=[2022, 2023, 2024, 2025],
                        help='One or more test years (each runs independently)')
    parser.add_argument('--seeds',      type=int, nargs='+',
                        default=[0, 2, 4, 42, 181, 276, 370, 413, 475, 512, 518, 783, 788, 831, 1091, 1231, 1405, 1497, 1560, 2024],
                        help='Random seeds — one independent expert per seed')

    # ── data ──────────────────────────────────────────────────
    parser.add_argument('--data_path', type=str, default='./dataset/market_index/')

    # ── model hyperparameters ──────────────────────────────────
    parser.add_argument('--epochs',           type=int,   default=200)
    parser.add_argument('--batch_size',       type=int,   default=16)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay',     type=float, default=1e-4)
    parser.add_argument('--lookback_days', '-lb', type=int,   default=60)
    parser.add_argument('--alpha',            type=float, default=0.75,
                        choices=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument('--risk_aversion',    type=float, default=1.0)
    parser.add_argument('--max_weight',       type=float, default=0.2)
    parser.add_argument('--transaction_cost', type=float, default=0.002)
    parser.add_argument('--val_interval',     type=int,   default=5)
    parser.add_argument('--early_stopping_patience', type=int, default=10)

    # ── output dirs ────────────────────────────────────────────
    parser.add_argument('--base_checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--base_runs_dir',       type=str, default='./runs')
    parser.add_argument('--base_logs_dir',       type=str, default='./logs/training')

    # ── misc ───────────────────────────────────────────────────
    parser.add_argument('--device', type=str, default='cuda:0')

    args   = parser.parse_args()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    total_jobs = len(args.test_year) * len(args.seeds)
    print(f"\n{'='*70}")
    print(f"Sequential Multi-Seed Training")
    print(f"model_type={args.model_type} | "
          f"test_years={args.test_year} | seeds={args.seeds}")
    print(f"Total jobs: {total_jobs} "
          f"({len(args.test_year)} years × {len(args.seeds)} seeds)")
    print(f"{'='*70}\n")

    all_results = []
    t0 = time.time()

    for test_year in args.test_year:
        for seed in args.seeds:
            print(f"\n{'='*70}")
            print(f"  test_year={test_year} | seed={seed}")
            print(f"{'='*70}")

            result = run_single_seed(seed, test_year, args, device)
            all_results.append(result)

    elapsed = time.time() - t0

    # ── summary ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"All {total_jobs} jobs completed in {elapsed/60:.1f} min")
    print(f"{'='*70}")