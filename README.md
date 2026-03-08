# Poland GPW Stock Market Analysis

A quantitative analysis of five major stocks listed on the Warsaw Stock Exchange (GPW), covering three years of historical data (March 2023 – March 2026). The project combines technical indicator analysis with machine learning models to explore price trends and short-term return predictability.

---

## Stocks Analyzed

| Ticker | Company | Sector | 3-Year Return | Ann. Volatility |
|--------|---------|--------|---------------|-----------------|
| **PKO.WA** | PKO Bank Polski | Banking | +221.1% | 31.2% |
| **PKN.WA** | PKN Orlen | Oil & Gas | +135.0% | 27.2% |
| **LPP.WA** | LPP S.A. | Fashion Retail | +129.3% | 42.6% |
| **PZU.WA** | PZU Group | Insurance | +122.9% | 26.0% |
| **CDR.WA** | CD Projekt | Video Games | +89.1% | 37.8% |

---

## Technical Stack

| Category | Library / Tool |
|----------|---------------|
| Data acquisition | `yfinance` |
| Data manipulation | `pandas`, `numpy` |
| Visualization | `matplotlib` |
| Machine learning | `scikit-learn` |
| Language | Python 3.12 |

---

## Project Structure

```
poland-stock-analysis/
├── notebooks/
│   └── poland_stock_analysis.py   # Main analysis script
├── output/                        # Generated charts (auto-created)
│   ├── PKO_WA_technical.png
│   ├── PKO_WA_prediction.png
│   ├── PKN_WA_technical.png
│   ├── PKN_WA_prediction.png
│   ├── LPP_WA_technical.png
│   ├── LPP_WA_prediction.png
│   ├── PZU_WA_technical.png
│   ├── PZU_WA_prediction.png
│   ├── CDR_WA_technical.png
│   ├── CDR_WA_prediction.png
│   └── summary_comparison.png
├── data/                          # Reserved for cached data
├── src/                           # Reserved for utility modules
└── README.md
```

---

## Methodology

### 1. Data Collection
Historical OHLCV data is downloaded via `yfinance` for the period **2023-03-09 to 2026-03-06** (746 trading days per stock).

### 2. Technical Indicators
Four indicators are computed for each stock:

- **MA20 / MA50** — 20-day and 50-day simple moving averages to identify trend direction and crossover signals.
- **RSI (14)** — Relative Strength Index to detect overbought (>70) and oversold (<30) conditions.
- **Bollinger Bands (20-period, ±2σ)** — Upper/lower bands, midline, and band-width percentage to measure volatility regimes.

### 3. Feature Engineering
Nine features derived from price and volume data are used as model inputs:

| Feature | Description |
|---------|-------------|
| `ret_1` | 1-day return |
| `ret_5` | 5-day return |
| `ret_20` | 20-day return |
| `ma20` | MA20 deviation from price |
| `ma50` | MA50 deviation from price |
| `rsi` | RSI(14) value |
| `bb_pos` | Price position within Bollinger Bands |
| `bb_width` | Bollinger Band width (%) |
| `vol_ratio` | Volume relative to 20-day average |

**Target variable:** 5-day forward return (percentage change).

### 4. Predictive Models
An 80/20 chronological train/test split is used (no shuffling, to prevent look-ahead bias).

- **Linear Regression** — Baseline model with standardized features.
- **Random Forest** — Ensemble model (200 trees, max depth 8) capturing non-linear relationships.

---

## Key Findings

### Price Performance
PKO Bank Polski was the standout performer, returning **+221%** over three years — significantly outpacing the other four stocks. PZU (insurance) and PKN (oil & gas) offered the lowest volatility profiles, making them relatively defensive holdings.

### Technical Signals
- **CDR.WA** exhibited the widest Bollinger Bands (highest `bb_width`), reflecting the sharp drawdown in 2023–2024 followed by a strong recovery.
- **PKO.WA** and **PKN.WA** maintained consistent uptrends with price spending most of the period above both MA20 and MA50.
- RSI overbought signals (>70) appeared repeatedly for PKO.WA during its rally but did not reliably predict reversals.

### Model Performance

| Ticker | LR R² | RF R² | LR RMSE | RF RMSE |
|--------|--------|--------|---------|---------|
| PKO.WA | −0.332 | −0.553 | 4.27% | 4.61% |
| PKN.WA | −0.203 | −0.372 | 4.19% | 4.47% |
| CDR.WA | −0.086 | −0.308 | 4.97% | 5.45% |
| PZU.WA | −0.133 | −0.181 | 2.95% | 3.01% |
| LPP.WA | −0.075 | −0.097 | 5.26% | 5.32% |

Negative R² values indicate that both models underperform a naive mean prediction on the test set. This is expected: short-term equity returns are close to a random walk, and technical features alone carry minimal predictive signal over a 5-day horizon. The results serve as a useful baseline and highlight the difficulty of systematic short-term forecasting.

---

## Output Charts

Each stock produces two charts:

1. **`<TICKER>_technical.png`** — Four-panel chart showing:
   - Closing price with MA20, MA50, and Bollinger Bands
   - Trading volume (millions)
   - RSI with overbought/oversold zones
   - Bollinger Band width over time

2. **`<TICKER>_prediction.png`** — Two-panel chart showing:
   - Actual vs. predicted 5-day returns (Linear Regression and Random Forest)
   - Random Forest feature importance ranking

3. **`summary_comparison.png`** — Cross-stock comparison of normalized price performance (base 100) and model metrics side by side.

---

## Getting Started

### Prerequisites
Python 3.10+ is required.

### Installation

```bash
git clone <repository-url>
cd poland-stock-analysis
pip install yfinance pandas numpy matplotlib scikit-learn scipy
```

### Run

```bash
python notebooks/poland_stock_analysis.py
```

All charts are saved automatically to the `output/` directory. A summary table is printed to the console at the end of the run.

---

## Limitations & Future Work

- **Prediction horizon:** Only 5-day forward returns are modeled. Longer horizons (e.g., 20-day) may yield different dynamics.
- **Feature scope:** Macro indicators (interest rates, EUR/PLN FX), sector ETF flows, and sentiment data are not included.
- **Model selection:** More expressive models (XGBoost, LSTM) and proper time-series cross-validation could improve robustness.
- **Transaction costs:** No slippage or commission is modeled; live trading results would differ.

---

## License

This project is for educational and research purposes only. It does not constitute financial advice.
