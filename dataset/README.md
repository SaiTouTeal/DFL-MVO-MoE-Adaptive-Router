# Dataset

The raw data used in the paper is licensed vendor data (Bloomberg / ICE) and is **not redistributed** in this repository.
Place your own copies in this folder using the layout and formats below; all scripts read from these paths by default.

```
dataset/
├── market_index/                         # the 20-asset universe, one CSV per asset
│   ├── C0A0_Index.csv
│   ├── CCMP.csv
│   ├── ...
│   └── TWSE.csv
└── risk_free_index/
    └── US zero coupon bond-rf.csv        # risk-free rate (USD Treasury zero-coupon yield)
```

## `market_index/` — asset price panel

One file per asset. The **file name is the asset name** used everywhere in the code (column headers, weight files, plots),
and files are loaded in **sorted file-name order**, which fixes the asset order seen by the models.
A trained checkpoint therefore only works with the same set of 20 file names.

| File | Bloomberg ticker | Index |
|---|---|---|
| `SPX.csv` | SPX Index | S&P 500 |
| `CCMP.csv` | CCMP Index | Nasdaq Composite |
| `RTY.csv` | RTY Index | Russell 2000 |
| `MXWO.csv` | MXWO Index | MSCI World |
| `MXWD.csv` | MXWD Index | MSCI ACWI |
| `MXEF.csv` | MXEF Index | MSCI Emerging Markets |
| `SXXP.csv` | SXXP Index | STOXX Europe 600 |
| `NKY.csv` | NKY Index | Nikkei 225 |
| `HSCEI.csv` | HSCEI Index | Hang Seng China Enterprises |
| `TWSE.csv` | TWSE Index | TAIEX |
| `SENSEX.csv` | SENSEX Index | S&P BSE Sensex |
| `IBOV.csv` | IBOV Index | Ibovespa |
| `LEGATRUU_Index.csv` | LEGATRUU Index | Bloomberg Global Aggregate |
| `LBUSTRUU_Index.csv` | LBUSTRUU Index | Bloomberg US Aggregate |
| `G0Q0_Index.csv` | G0Q0 Index | ICE BofA US Treasury |
| `C0A0_Index.csv` | C0A0 Index | ICE BofA US Corporate |
| `H0A0_Index.csv` | H0A0 Index | ICE BofA US High Yield |
| `EMUSTRUU_Index.csv` | EMUSTRUU Index | Bloomberg EM USD Aggregate |
| `LP06TREU_Index.csv` | LP06TREU Index | Bloomberg Pan-European Aggregate |
| `LP01TREU_Index.csv` | LP01TREU Index | Bloomberg Pan-European High Yield |

**Format.** A header row followed by one row per trading day. Two columns are required; any other columns are ignored:

| Column | Content |
|---|---|
| `Dates` | trading date, any format `pandas.to_datetime` can parse (e.g. `2022/1/3`) |
| `PX_LAST` | index level (total-return series) |

```
Dates,PX_LAST
2022/1/3,4796.56
2022/1/4,4793.54
...
```

The files are joined on `Dates`, and **any date missing in at least one file is dropped**, so the usable sample is the
common date range of all 20 assets. The paper uses data from 2000 through 2025-10-09
(training from 2000, test years 2022–2025; see Table 3 of the paper).

**Preprocessing is expected to be done before export.** Section 4.1 of the paper converts all index levels to U.S. dollars
and lags Asian and European markets by one trading day relative to U.S. markets. The code does not perform these steps;
it computes simple daily returns directly from `PX_LAST`.

## `risk_free_index/US zero coupon bond-rf.csv` — risk-free rate

A Bloomberg export of the USD Treasury zero-coupon yield curve. Used as the risk-free asset in the 60/40 risky/risk-free
allocation and as the benchmark for Sharpe/Sortino ratios.

- The **first 6 rows are skipped** (Bloomberg export header block); data starts on row 7.
- Column 0 is the date; **column 1 is the 3-month yield (`I02503M Index`) in percent per annum**. Remaining columns are ignored.
- Bloomberg placeholders `#N/A N/A` and `#NAME?` are treated as missing.
- The annual yield is converted to a daily rate as `(1 + y/100)^(1/252) - 1`.

```
<6 header rows>
1/3/2022,0.05,...
1/4/2022,0.06,...
```

`multi_experts_router_testing.py` can alternatively use the daily return of `market_index/G0Q0_Index.csv` as the risk-free
series (`--risk_free_source g0q0`); the paper uses the 3-month Treasury yield above.
