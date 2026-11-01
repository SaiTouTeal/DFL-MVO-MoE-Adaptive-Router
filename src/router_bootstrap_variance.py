"""
router_bootstrap_variance.py
────────────────────────────────────────────────────────────────────────
背景：baseline 與各個 expert 都是用 20 個不同的 random seed 各自獨立訓練，
所以可以算出「跨 seed」的 mean ± std。但 router 是拿這 20 個訓練好的 expert
當作候選池、動態選出 top-k 個做等權平均，整個機制只會產生「一組」表現，
沒有天然的重複樣本可以算變異數。

做法（bootstrap resampling，對應與教授討論的方案一）：
    對每個 (model_type, test_year, router_mode) 組合，重複 B 次：
        1. 從基準的 20 個 seed 中「有放回」抽出 len(base_seeds) 次
        2. 取這次抽樣結果的「唯一 seed 集合」作為這個 replicate 的候選池
           （有放回抽樣造成的重複次數本身不使用，只用來決定某個 seed這次
           「有沒有出現」；避免同一顆 seed 在 router 的 top-k 選擇中因為
           出現兩次而被誤植成雙倍權重）
        3. 用跟正式跑法完全相同的 router 超參數（score_method /
           warmup_days / review_freq / rolling_window / combo_beta /
           top_k ...），呼叫 multi_experts_router_testing.py 的
           --no_save 模式，限制它只能看到這個候選池裡的 seed
        4. 解析 stdout 的 METRICS_JSON / ROUTER_PARAMS_JSON，記錄這個
           replicate 的 router 績效、ensemble（候選池內所有 seed 等權平均）
           績效，與實際用到的 top_k
    B 個 replicate 跑完後，計算 mean / std / 95% 分位數區間。

router_mode（對應 multi_experts_router_testing.py 的兩種 k 選法）：
    'static'              固定 --top_k（例如恆定用 k=3），每個 replicate 用同一個 k。
    'validation_selected' 加上 --grid_search，每個 replicate 各自在自己候選池的驗證集
                          （test_year 前一年）上，對 --gs_top_k 清單（預設候選
                          k={1,3,5,7,9,11}）做 grid search，選出該 replicate 的
                          最佳 k 才拿去跑測試集，等同 multi_experts_router_testing.py
                          的「validation-selected-k」跑法。grid search 完全只碰驗證集，
                          不會有 test-set data leakage（沿用原始腳本的保證）。
    'ensemble'            不看 router 演算法，只記錄候選池內「所有」seed 的 w_hat 等權平均
                          （一般 ensemble）的績效。彙總結果另存為 bootstrap_summary_ensemble.csv /
                          bootstrap_summary_ensemble_formatted.csv，並與 router 結果分開顯示。
                          例：python src/router_bootstrap_variance.py --router_modes ensemble --n_bootstrap 30
    兩種模式可以在同一次執行裡都跑（--router_modes static validation_selected），
    彙總表會分開列出，方便直接比較。

    注意：validation_selected 模式每個 replicate 會多做一次驗證窗（test_year 前一年）
    的完整推論，所以耗時大約是 static 模式的兩倍。

平均一個 replicate 抽 20 次（有放回）後，去重複大約會剩下 ~12-13 個
唯一 seed（bootstrap 的標準性質：約 63.2% 會被抽到至少一次），每次
replicate 的候選池大小都不太一樣，這就是變異的來源。

注意：這個腳本每個 replicate 都是重新啟動一個 subprocess、重新載入模型、
重新跑一次完整推論，所以速度取決於 model_type、seed 數量與 GPU。
正式要跑 200 次 x 4 個年度 x 3 個 model_type 之前，強烈建議先用
--n_bootstrap 20（甚至 --dry_run）跑一次估算耗時。

用法：
    # 先用小規模測試估算耗時
    python src/router_bootstrap_variance.py --model_types mlp --test_years 2022 \\
        --n_bootstrap 20

    # 正式跑（可分批對不同 model_type / test_year 各自下指令，方便平行/中斷續跑）
    python src/router_bootstrap_variance.py --model_types mlp lstm transformer \\
        --test_years 2022 2023 2024 2025 --n_bootstrap 200 --resume

輸出：
    {output_dir}/bootstrap_raw_{model_type}_{test_year}.csv   每個 replicate 的細節（跑一筆存一筆，可中斷續跑）
    {output_dir}/bootstrap_summary.csv                        依 model_type x test_year 彙總的 mean/std/95% CI
"""

import argparse
import itertools
import json
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

SCRIPT = Path(__file__).parent / 'multi_experts_router_testing.py'

BASE_SEEDS = [0, 2, 4, 42, 181, 276, 370, 413, 475,
              512, 518, 783, 788, 831, 1091, 1231,
              1405, 1497, 1560, 2024]

# METRICS_JSON 裡跟 router 有關的欄位（見 multi_experts_router_testing.py 的
# --no_save 分支），bootstrap 只關心這些。順序對齊 performance_summary.csv 的欄位順序：
# total_ret, ann_ret, ann_vol, sharpe, sortino, max_dd
ROUTER_METRIC_KEYS = ['router_ret', 'router_ann_ret', 'router_ann_vol',
                      'router_sharpe', 'router_sortino', 'router_max_dd']

# Ensemble（等權平均候選池內「所有」seed 的 w_hat）的指標，同樣來自 METRICS_JSON 的 eq_* 欄位。
# 只在 router_mode='ensemble' 時記錄與彙總；因為各 router_mode 共用同一組候選池序列，
# ensemble 與 static 的第 b 個 replicate 用的是同一個候選池，可直接配對比較
ENSEMBLE_METRIC_KEYS = ['eq_ret', 'eq_ann_ret', 'eq_ann_vol',
                        'eq_sharpe', 'eq_sortino', 'eq_max_dd']

# 百分比類指標（報酬/波動/回撤）用 "±%" 呈現；比率類指標（Sharpe/Sortino）用純小數呈現，
# 跟 multi_experts_router_testing.py 的 _scale_metrics_for_csv() 呈現慣例一致
PCT_METRIC_KEYS = {'router_ret', 'router_ann_ret', 'router_ann_vol', 'router_max_dd',
                   'eq_ret', 'eq_ann_ret', 'eq_ann_vol', 'eq_max_dd'}

METRIC_LABELS = {
    'eq_sharpe':      'Ensemble Sharpe',
    'eq_sortino':     'Ensemble Sortino',
    'eq_ret':         'Ensemble Total Return',
    'eq_ann_ret':     'Ensemble Ann. Return',
    'eq_ann_vol':     'Ensemble Ann. Vol',
    'eq_max_dd':      'Ensemble Max DD',
    'router_sharpe':  'Router Sharpe',
    'router_sortino': 'Router Sortino',
    'router_ret':     'Router Total Return',
    'router_ann_ret': 'Router Ann. Return',
    'router_ann_vol': 'Router Ann. Vol',
    'router_max_dd':  'Router Max DD',
    'used_top_k':     'Selected k',
}


def _fmt_mean_std(vals: np.ndarray, key: str) -> str:
    """把一個 metric 的 replicate 陣列格式化成 'mean ± std' 字串。
    百分比類指標（報酬/波動/回撤）乘 100 顯示（不附 % 符號）；Sharpe/Sortino 保留 3 位小數。"""
    mean = float(np.mean(vals))
    std  = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    if key in PCT_METRIC_KEYS:
        return f'{mean*100:.2f} ± {std*100:.2f}'
    return f'{mean:.3f} ± {std:.3f}'


def _metric_keys(router_mode: str) -> list:
    """ensemble 模式只看 eq_*；static / validation_selected 只看 router_*，兩者分開彙總。"""
    if router_mode == 'ensemble':
        return list(ENSEMBLE_METRIC_KEYS)
    return list(ROUTER_METRIC_KEYS)


def _format_summary_row(df: pd.DataFrame, model_type: str, test_year: int,
                        router_mode: str) -> dict:
    """把單一 (model_type, test_year, router_mode) 的所有 replicate，彙整成一列
    「每個指標都是 mean ± std」的人類可讀格式，方便直接貼進論文表格。"""
    keys = _metric_keys(router_mode)
    if router_mode == 'validation_selected':
        keys = keys + ['used_top_k']   # static 模式下 k 是常數，顯示 mean±std=0 沒意義，故只在 validation_selected 顯示
    row = {'model_type': model_type, 'test_year': test_year,
           'router_mode': router_mode, 'n_replicates': len(df)}
    for key in keys:
        if key not in df.columns:
            continue
        vals = df[key].dropna().to_numpy()
        if len(vals) == 0:
            continue
        row[METRIC_LABELS.get(key, key)] = _fmt_mean_std(vals, key)
    return row


def _combo_seed(bootstrap_seed: int, model_type: str, test_year: int) -> int:
    """每個 (model_type, test_year) 各自獨立、但跨執行可重現的 RNG seed。
    用 crc32 而不是內建 hash()，因為 Python 的字串 hash 預設有隨機化，
    不同次執行會不一樣，crc32 才能保證每次重跑抽到同一組 bootstrap 樣本。"""
    key = f'{model_type}_{test_year}'.encode()
    return (bootstrap_seed + zlib.crc32(key)) % (2**31)


def _bootstrap_pool(rng: np.random.RandomState, base_seeds: list, draw_size: int) -> list:
    """有放回抽 draw_size 次，回傳去重複後、由小到大排序的候選池。"""
    draw = rng.choice(base_seeds, size=draw_size, replace=True)
    return sorted(set(int(s) for s in draw))


def _run_router(model_type: str, test_year: int, pool: list, router_mode: str, args) -> dict:
    cmd = [
        sys.executable, str(SCRIPT),
        '--model_type', model_type,
        '--test_year', str(test_year),
        '--seeds', *[str(s) for s in pool],
        '--checkpoint_type',      args.checkpoint_type,
        '--base_checkpoint_dir',  args.base_checkpoint_dir,
        '--data_path',            args.data_path,
        '--transaction_cost',     str(args.transaction_cost),
        '--lookback_days',        str(args.lookback_days),
        '--warmup_days',          str(args.warmup_days),
        '--review_freq',          str(args.review_freq),
        '--top_k',                str(args.top_k),
        '--bottom_k',             str(args.bottom_k),
        '--rolling_window',       str(args.rolling_window),
        '--score_method',         args.score_method,
        '--combo_beta',           str(args.combo_beta),
        '--combo_alpha',          str(args.combo_alpha),
        '--device',               args.device,
        '--seed',                 str(args.torch_seed),
        '--no_save',
    ]

    if router_mode == 'validation_selected':
        cmd += [
            '--grid_search',
            '--grid_search_metric', args.grid_search_metric,
            '--gs_top_k', *[str(k) for k in args.gs_top_k],
        ]

    if args.dry_run:
        print(f'  [dry_run] mode={router_mode} pool={pool}')
        return {}

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f'  [ERROR] model={model_type} year={test_year} mode={router_mode} pool={pool}')
        print(result.stderr[-800:])
        return {}

    metrics, router_params = {}, {}
    for line in result.stdout.splitlines():
        if line.startswith('METRICS_JSON:'):
            try:
                metrics = json.loads(line[len('METRICS_JSON:'):])
            except Exception:
                pass
        elif line.startswith('ROUTER_PARAMS_JSON:'):
            try:
                router_params = json.loads(line[len('ROUTER_PARAMS_JSON:'):])
            except Exception:
                pass

    if not metrics:
        print(f'  [WARN] model={model_type} year={test_year} mode={router_mode} pool={pool} '
              f'沒有解析到 METRICS_JSON')
        return {}

    # ensemble 模式：只保留一般等權 ensemble 的 eq_* 指標，router 結果與 k 都不記錄
    if router_mode == 'ensemble':
        return {k: v for k, v in metrics.items() if k in ENSEMBLE_METRIC_KEYS}

    # validation_selected 模式下，這是該 replicate 在自己候選池上 grid search 選出的 k；
    # static 模式下，這就是固定的 args.top_k（每個 replicate 都一樣）
    if 'top_k' in router_params:
        metrics['used_top_k'] = router_params['top_k']
    return metrics


def _run_one_combo(model_type: str, test_year: int, router_mode: str,
                   args, out_dir: Path) -> pd.DataFrame:
    raw_path = (None if args.no_save else
                out_dir / f'bootstrap_raw_{model_type}_{test_year}_{router_mode}.csv')

    rows = []
    done_idx = set()
    if args.resume and raw_path is not None and raw_path.exists():
        prev = pd.read_csv(raw_path)
        rows = prev.to_dict('records')
        done_idx = set(int(i) for i in prev['replicate_idx'].tolist())
        print(f'[resume] {model_type}/{test_year}/{router_mode}: 已有 {len(done_idx)} 筆，繼續跑剩下的')

    draw_size = args.draw_size if args.draw_size is not None else len(args.base_seeds)
    # bootstrap 抽樣的 RNG 只依 (model_type, test_year) 決定，router_mode 之間共用同一組
    # 候選池序列，這樣 static vs validation_selected 是在完全一樣的候選池上比較，
    # 差異只來自 router 選 k 的方式，而不是又抽到了不同的 seed 組合
    rng = np.random.RandomState(_combo_seed(args.bootstrap_seed, model_type, test_year))

    pbar = tqdm(range(args.n_bootstrap), desc=f'{model_type}/{test_year}/{router_mode}')
    for b in pbar:
        # 即使 resume 時跳過已完成的 replicate，仍要照順序呼叫 rng，
        # 確保之後抽到的 pool 跟從頭跑一次時完全一樣（可重現）
        pool = _bootstrap_pool(rng, args.base_seeds, draw_size)

        if b in done_idx:
            continue
        if args.dry_run:
            print(f'  [dry_run] replicate {b}: mode={router_mode} pool={pool}')
            continue

        metrics = _run_router(model_type, test_year, pool, router_mode, args)
        if not metrics:
            continue

        row = {
            'model_type':    model_type,
            'test_year':     test_year,
            'router_mode':   router_mode,
            'replicate_idx': b,
            'pool_size':     len(pool),
            'pool_seeds':    ' '.join(str(s) for s in pool),
            **metrics,
        }
        rows.append(row)

        # 每筆寫一次檔，中斷後用 --resume 續跑不會遺失（--no_save 時完全不寫檔）
        if raw_path is not None:
            pd.DataFrame(rows).to_csv(raw_path, index=False)

    return pd.DataFrame(rows)


def _summarize(df: pd.DataFrame, model_type: str, test_year: int, router_mode: str) -> dict:
    row = {'model_type': model_type, 'test_year': test_year,
           'router_mode': router_mode, 'n_replicates': len(df)}
    for col in _metric_keys(router_mode) + ['used_top_k']:
        if col not in df.columns:
            continue
        vals = df[col].dropna().to_numpy()
        if len(vals) == 0:
            continue
        row[f'{col}_mean']  = float(np.mean(vals))
        row[f'{col}_std']   = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        row[f'{col}_ci_lo'] = float(np.percentile(vals, 2.5))
        row[f'{col}_ci_hi'] = float(np.percentile(vals, 97.5))
    return row


def main():
    parser = argparse.ArgumentParser(description='Bootstrap variance estimation for the router')

    parser.add_argument('--model_types', type=str, nargs='+',
                        default=['mlp', 'lstm', 'transformer'],
                        choices=['mlp', 'lstm', 'transformer'])
    parser.add_argument('--test_years', type=int, nargs='+',
                        default=[2022, 2023, 2024, 2025],
                        choices=[2022, 2023, 2024, 2025])
    parser.add_argument('--router_modes', type=str, nargs='+', default=['static'],
                        choices=['static', 'validation_selected', 'ensemble'],
                        help="'ensemble'：不看 router，只記錄候選池內所有 seed 等權平均的一般 ensemble 結果"
                             "（結果另存 bootstrap_summary_ensemble*.csv、分開顯示）；"
                             "'static'：每個 replicate 都用固定 --top_k；"
                             "'validation_selected'：每個 replicate 各自在自己候選池的驗證集上"
                             "對 --gs_top_k 做 grid search 選 k（對應 multi_experts_router_testing.py "
                             "的 --grid_search）。可同時指定兩者以便直接比較")
    parser.add_argument('--base_seeds', type=int, nargs='+', default=BASE_SEEDS,
                        help='已訓練好的 20 個 expert seed（bootstrap 抽樣的母體）')
    parser.add_argument('--draw_size', type=int, default=None,
                        help='每次 replicate 有放回抽幾次；預設 = len(base_seeds)。'
                             '調小會讓平均候選池變小、變異變大，可用來做敏感度分析')
    parser.add_argument('--n_bootstrap', type=int, default=30,
                        help='每個 (model_type, test_year) 要重複幾次 bootstrap replicate')
    parser.add_argument('--bootstrap_seed', type=int, default=42,
                        help='控制 bootstrap 抽樣本身的隨機性，確保可重現（與 router 內部的 --seed 無關）')

    # ── router 超參數：預設值對齊正式跑法（router_params_used.json 核實過的設定）──
    parser.add_argument('--checkpoint_type', type=str, default='checkpoint_best.pt')
    parser.add_argument('--base_checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--data_path', type=str, default='./dataset/market_index/')
    parser.add_argument('--transaction_cost', type=float, default=0.002)
    parser.add_argument('--lookback_days', type=int, default=60)
    parser.add_argument('--warmup_days', type=int, default=20)
    parser.add_argument('--review_freq', type=int, default=60)
    parser.add_argument('--top_k', type=int, default=3,
                        help='router_mode=static 時固定使用的 k')
    parser.add_argument('--gs_top_k', type=int, nargs='+', default=[1, 3, 5, 7, 9, 11],
                        help='router_mode=validation_selected 時，grid search 的候選 k 清單')
    parser.add_argument('--grid_search_metric', type=str, default='sharpe',
                        choices=['sharpe', 'sortino', 'total_ret', 'ann_ret'],
                        help='router_mode=validation_selected 時，grid search 在驗證集上的目標')
    parser.add_argument('--bottom_k', type=int, default=0)
    parser.add_argument('--rolling_window', type=int, default=0)
    parser.add_argument('--score_method', type=str, default='cumret',
                        choices=['sharpe', 'cumret', 'mu_corr', 'ic_ir', 'w_corr', 'cumret_mucorr'])
    parser.add_argument('--combo_beta', type=float, default=0.94)
    parser.add_argument('--combo_alpha', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--torch_seed', type=int, default=42,
                        help='傳給 multi_experts_router_testing.py 的 --seed（其內部 torch 隨機性用，'
                             '與這裡的 bootstrap 抽樣是兩件事）')

    parser.add_argument('--output_dir', type=str, default='./bootstrap_results')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='raw csv 已存在時，跳過已經跑過的 replicate_idx，接續未完成的部分'
                             '（與 --no_save 互斥，沒有落地的 csv 就無法接續）')
    parser.add_argument('--no_save', action='store_true', default=False,
                        help='不寫任何檔案到硬碟（不建 output_dir、不存 raw/summary csv），'
                             '只在跑完後於畫面上印出彙總表；適合單純想看結果、不想留東西在專案裡的試跑')
    parser.add_argument('--dry_run', action='store_true', default=False,
                        help='只印出每個 replicate 抽到的候選池，不實際執行 router')

    args = parser.parse_args()

    if args.no_save and args.resume:
        print('[WARN] --no_save 不會落地任何 csv，--resume 會被忽略')
        args.resume = False

    out_dir = Path(args.output_dir)
    if not args.no_save and not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    combos = list(itertools.product(args.model_types, args.test_years, args.router_modes))
    draw_size = args.draw_size if args.draw_size is not None else len(args.base_seeds)

    print(f'{"="*70}')
    print(f'Router Bootstrap Variance Estimation')
    print(f'{"="*70}')
    print(f'組合數: {len(combos)} 個 (model_type, test_year, router_mode) x {args.n_bootstrap} replicates '
          f'= {len(combos) * args.n_bootstrap} 次 router 執行')
    print(f'router_modes={args.router_modes}')
    print(f'router 設定: score_method={args.score_method}, '
          f'warmup_days={args.warmup_days}, review_freq={args.review_freq}, '
          f'rolling_window={args.rolling_window}, combo_beta={args.combo_beta}')
    if 'static' in args.router_modes:
        print(f'  [static]              top_k={args.top_k}')
    if 'validation_selected' in args.router_modes:
        print(f'  [validation_selected] gs_top_k={args.gs_top_k}, '
              f'grid_search_metric={args.grid_search_metric}')
    print(f'bootstrap 抽樣: base_pool={len(args.base_seeds)} 個 seed, 每次抽 {draw_size} 次（有放回，去重複後使用）')
    print(f'{"="*70}\n')

    t0 = time.time()
    all_dfs = []
    for model_type, test_year, router_mode in combos:
        df = _run_one_combo(model_type, test_year, router_mode, args, out_dir)
        all_dfs.append((model_type, test_year, router_mode, df))

    if args.dry_run:
        print('\n[dry_run] 完成，未實際執行任何 router。')
        return

    elapsed = time.time() - t0
    print(f'\n{"="*70}')
    print(f'完成，共耗時 {elapsed/60:.1f} 分鐘')
    if args.no_save:
        print('（--no_save 模式：未寫入任何檔案）')

    # Ensemble 與 Router（static / validation_selected）分開彙總、分開存檔、分開顯示
    groups = [
        ('Ensemble（一般等權 ensemble）', '_ensemble', lambda m: m == 'ensemble'),
        ('Router',                       '',          lambda m: m != 'ensemble'),
    ]
    for title, suffix, belongs in groups:
        dfs = [(mt, ty, rm, df) for mt, ty, rm, df in all_dfs if belongs(rm) and len(df) > 0]
        if not dfs:
            continue

        # 數值版彙總（每個 metric 拆成 _mean/_std/_ci_lo/_ci_hi 四欄），適合再拿去做圖表或其他統計
        summary_df = pd.DataFrame([_summarize(df, mt, ty, rm) for mt, ty, rm, df in dfs])
        # 人類可讀版彙總（每個 metric 一欄，內容是 "mean ± std" 字串），適合直接貼進論文表格
        formatted_df = pd.DataFrame([_format_summary_row(df, mt, ty, rm) for mt, ty, rm, df in dfs])

        print(f'{"="*70}')
        print(f'[{title}]')
        if not args.no_save:
            summary_path   = out_dir / f'bootstrap_summary{suffix}.csv'
            formatted_path = out_dir / f'bootstrap_summary{suffix}_formatted.csv'
            summary_df.to_csv(summary_path, index=False)
            formatted_df.to_csv(formatted_path, index=False, encoding='utf-8-sig')
            print(f'數值版彙總（mean/std/95% CI 分欄）: {summary_path}')
            print(f'可讀版彙總（每個指標 mean ± std）: {formatted_path}')
        print(formatted_df.to_string(index=False))
    print(f'{"="*70}')


if __name__ == '__main__':
    main()
