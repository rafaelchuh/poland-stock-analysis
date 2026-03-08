"""
波兰 GPW 交易所股票数据分析
- 下载5支主要股票近3年数据
- 计算技术指标：MA20、MA50、RSI、布林带
- 可视化价格走势与技术指标
- Linear Regression 与 Random Forest 预测股价趋势
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# ── 中文字体设置 ──────────────────────────────────────────────
plt.rcParams['font.family'] = ['STHeiti', 'Heiti TC', 'PingFang HK', 'Songti SC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ── 全局配置 ──────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

STOCKS = {
    'PKN.WA': 'PKN Orlen (石油天然气)',
    'PKO.WA': 'PKO Bank Polski (银行)',
    'CDR.WA': 'CD Projekt (游戏)',
    'PZU.WA': 'PZU (保险)',
    'LPP.WA': 'LPP (时尚零售)',
}

END_DATE   = datetime.today()
START_DATE = END_DATE - timedelta(days=3 * 365)

COLORS = {
    'price':  '#2196F3',
    'ma20':   '#FF9800',
    'ma50':   '#E91E63',
    'bb_up':  '#9C27B0',
    'bb_low': '#9C27B0',
    'bb_mid': '#607D8B',
    'rsi':    '#00BCD4',
    'vol':    '#78909C',
    'pred_lr': '#FF5722',
    'pred_rf': '#4CAF50',
}


# ══════════════════════════════════════════════════════════════
# 1. 数据下载
# ══════════════════════════════════════════════════════════════

def download_data(tickers: dict, start: datetime, end: datetime) -> dict:
    data = {}
    for ticker, name in tickers.items():
        print(f"  下载 {ticker} ({name}) ...")
        try:
            df = yf.download(ticker, start=start.strftime('%Y-%m-%d'),
                             end=end.strftime('%Y-%m-%d'), progress=False)
            if df.empty:
                print(f"    ⚠ {ticker} 无数据，跳过")
                continue
            # 展平多层列（yfinance ≥ 0.2 可能返回 MultiIndex）
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.dropna(inplace=True)
            data[ticker] = df
            print(f"    ✓ {len(df)} 条记录 ({df.index[0].date()} ~ {df.index[-1].date()})")
        except Exception as e:
            print(f"    ✗ 下载失败: {e}")
    return data


# ══════════════════════════════════════════════════════════════
# 2. 技术指标计算
# ══════════════════════════════════════════════════════════════

def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df['Close']

    # 移动平均
    df['MA20'] = close.rolling(20).mean()
    df['MA50'] = close.rolling(50).mean()

    # RSI(14)
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - 100 / (1 + rs)

    # 布林带(20, 2σ)
    std20 = close.rolling(20).std()
    df['BB_mid'] = df['MA20']
    df['BB_up']  = df['MA20'] + 2 * std20
    df['BB_low'] = df['MA20'] - 2 * std20
    df['BB_width'] = (df['BB_up'] - df['BB_low']) / df['BB_mid'] * 100

    return df


# ══════════════════════════════════════════════════════════════
# 3. 可视化
# ══════════════════════════════════════════════════════════════

def plot_stock(ticker: str, df: pd.DataFrame, name: str) -> str:
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(f'{ticker}  {name}\n价格走势与技术指标分析',
                 fontsize=16, fontweight='bold', y=0.98)

    gs = GridSpec(4, 1, figure=fig, hspace=0.45,
                  height_ratios=[3, 1.2, 1.2, 1])

    # ── 价格 + 均线 + 布林带 ──────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df.index, df['Close'], color=COLORS['price'],
             linewidth=1.4, label='收盘价', zorder=3)
    ax1.plot(df.index, df['MA20'],  color=COLORS['ma20'],
             linewidth=1.2, linestyle='--', label='MA20')
    ax1.plot(df.index, df['MA50'],  color=COLORS['ma50'],
             linewidth=1.2, linestyle='--', label='MA50')
    ax1.fill_between(df.index, df['BB_low'], df['BB_up'],
                     alpha=0.12, color=COLORS['bb_up'], label='布林带(±2σ)')
    ax1.plot(df.index, df['BB_up'],  color=COLORS['bb_up'],
             linewidth=0.7, linestyle=':', alpha=0.7)
    ax1.plot(df.index, df['BB_low'], color=COLORS['bb_low'],
             linewidth=0.7, linestyle=':', alpha=0.7)
    ax1.set_ylabel('价格 (PLN)', fontsize=11)
    ax1.legend(loc='upper left', fontsize=9, ncol=4)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    # ── 成交量 ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.bar(df.index, df['Volume'] / 1e6, color=COLORS['vol'],
            alpha=0.7, width=1, label='成交量')
    ax2.set_ylabel('成交量 (百万)', fontsize=11)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ── RSI ──────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(df.index, df['RSI'], color=COLORS['rsi'],
             linewidth=1.2, label='RSI(14)')
    ax3.axhline(70, color='red',   linestyle='--', linewidth=0.8, alpha=0.7)
    ax3.axhline(30, color='green', linestyle='--', linewidth=0.8, alpha=0.7)
    ax3.fill_between(df.index, 30, df['RSI'],
                     where=(df['RSI'] < 30), alpha=0.3, color='green', label='超卖区')
    ax3.fill_between(df.index, 70, df['RSI'],
                     where=(df['RSI'] > 70), alpha=0.3, color='red',   label='超买区')
    ax3.set_ylim(0, 100)
    ax3.set_ylabel('RSI', fontsize=11)
    ax3.legend(loc='upper left', fontsize=9, ncol=3)
    ax3.grid(True, alpha=0.3)

    # ── 布林带宽度 ───────────────────────────────────
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.plot(df.index, df['BB_width'], color=COLORS['bb_mid'],
             linewidth=1.0, label='布林带宽度 (%)')
    ax4.set_ylabel('带宽 %', fontsize=11)
    ax4.set_xlabel('日期', fontsize=11)
    ax4.legend(loc='upper left', fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax4.get_xticklabels(), rotation=30, ha='right', fontsize=8)

    path = os.path.join(OUTPUT_DIR, f'{ticker.replace(".", "_")}_technical.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════
# 4. 特征工程
# ══════════════════════════════════════════════════════════════

def build_features(df: pd.DataFrame, horizon: int = 5) -> tuple[pd.DataFrame, pd.Series]:
    feat = pd.DataFrame(index=df.index)
    close = df['Close']

    feat['ret_1']    = close.pct_change(1)
    feat['ret_5']    = close.pct_change(5)
    feat['ret_20']   = close.pct_change(20)
    feat['ma20']     = df['MA20'] / close - 1
    feat['ma50']     = df['MA50'] / close - 1
    feat['rsi']      = df['RSI']
    feat['bb_pos']   = (close - df['BB_low']) / (df['BB_up'] - df['BB_low'])
    feat['bb_width'] = df['BB_width']
    feat['vol_ratio']= df['Volume'] / df['Volume'].rolling(20).mean()

    # 目标：未来 horizon 天涨跌幅
    target = close.pct_change(horizon).shift(-horizon)

    combined = feat.join(target.rename('target')).dropna()
    X = combined.drop('target', axis=1)
    y = combined['target']
    return X, y


# ══════════════════════════════════════════════════════════════
# 5. 模型训练与预测
# ══════════════════════════════════════════════════════════════

def train_predict(X: pd.DataFrame, y: pd.Series, ticker: str) -> dict:
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_tr_s, y_train)
    y_pred_lr = lr.predict(X_te_s)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=200, max_depth=8,
                               random_state=42, n_jobs=-1)
    rf.fit(X_tr_s, y_train)
    y_pred_rf = rf.predict(X_te_s)

    results = {
        'dates':     X_test.index,
        'y_true':    y_test.values,
        'y_pred_lr': y_pred_lr,
        'y_pred_rf': y_pred_rf,
        'lr_r2':     r2_score(y_test, y_pred_lr),
        'rf_r2':     r2_score(y_test, y_pred_rf),
        'lr_rmse':   np.sqrt(mean_squared_error(y_test, y_pred_lr)),
        'rf_rmse':   np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        'feat_imp':  dict(zip(X.columns, rf.feature_importances_)),
    }
    return results


# ══════════════════════════════════════════════════════════════
# 6. 预测结果可视化
# ══════════════════════════════════════════════════════════════

def plot_predictions(ticker: str, name: str, results: dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{ticker}  {name}\n股价趋势预测（未来5日涨跌幅）',
                 fontsize=14, fontweight='bold')

    dates   = results['dates']
    y_true  = results['y_true']
    y_lr    = results['y_pred_lr']
    y_rf    = results['y_pred_rf']

    # ── 预测 vs 实际 ────────────────────────────────
    ax = axes[0]
    ax.plot(dates, y_true * 100, color='black',          linewidth=1.2,
            label='实际涨跌幅', alpha=0.85)
    ax.plot(dates, y_lr   * 100, color=COLORS['pred_lr'], linewidth=1.0,
            linestyle='--', label=f'线性回归  R²={results["lr_r2"]:.3f}')
    ax.plot(dates, y_rf   * 100, color=COLORS['pred_rf'], linewidth=1.0,
            linestyle='-.',  label=f'随机森林  R²={results["rf_r2"]:.3f}')
    ax.axhline(0, color='grey', linewidth=0.7, linestyle=':')
    ax.set_xlabel('日期', fontsize=11)
    ax.set_ylabel('5日涨跌幅 (%)', fontsize=11)
    ax.set_title('预测 vs 实际', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=8)

    # ── 特征重要性 ───────────────────────────────────
    ax2 = axes[1]
    feat_imp = results['feat_imp']
    names  = list(feat_imp.keys())
    values = list(feat_imp.values())
    idx    = np.argsort(values)
    colors_bar = ['#4CAF50' if v >= 0 else '#F44336' for v in [values[i] for i in idx]]
    bars = ax2.barh([names[i] for i in idx], [values[i] for i in idx],
                    color=colors_bar, edgecolor='white', linewidth=0.5)
    ax2.set_xlabel('重要性', fontsize=11)
    ax2.set_title('随机森林特征重要性', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='x')

    # 指标标注
    for bar, val in zip(bars, [values[i] for i in idx]):
        ax2.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                 f'{val:.3f}', va='center', fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'{ticker.replace(".", "_")}_prediction.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════
# 7. 汇总对比图
# ══════════════════════════════════════════════════════════════

def plot_summary(all_data: dict, all_results: dict) -> str:
    n = len(all_data)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 10))
    fig.suptitle('波兰 GPW 主要股票综合对比', fontsize=16, fontweight='bold', y=1.01)

    for col, (ticker, df) in enumerate(all_data.items()):
        name = STOCKS[ticker]

        # 上行：归一化价格走势
        ax_top = axes[0][col]
        norm = df['Close'] / df['Close'].iloc[0] * 100
        ax_top.plot(df.index, norm, color=COLORS['price'], linewidth=1.2)
        ax_top.fill_between(df.index, 100, norm,
                            where=(norm >= 100), alpha=0.15, color='green')
        ax_top.fill_between(df.index, 100, norm,
                            where=(norm < 100),  alpha=0.15, color='red')
        ax_top.axhline(100, color='grey', linewidth=0.7, linestyle='--')
        ax_top.set_title(f'{ticker}\n{name}', fontsize=10, fontweight='bold')
        ax_top.set_ylabel('归一化价格 (起点=100)', fontsize=8)
        ax_top.grid(True, alpha=0.3)
        ax_top.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.setp(ax_top.get_xticklabels(), rotation=30, ha='right', fontsize=7)

        # 最终涨跌幅标注
        total_ret = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
        color = 'green' if total_ret >= 0 else 'red'
        ax_top.text(0.98, 0.05, f'总涨跌: {total_ret:+.1f}%',
                    transform=ax_top.transAxes, ha='right', fontsize=9,
                    color=color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        # 下行：模型对比柱状图
        if ticker in all_results:
            ax_bot = axes[1][col]
            res = all_results[ticker]
            labels = ['线性回归', '随机森林']
            r2s    = [res['lr_r2'], res['rf_r2']]
            rmses  = [res['lr_rmse'] * 100, res['rf_rmse'] * 100]
            x = np.arange(2)
            bars1 = ax_bot.bar(x - 0.2, r2s,   0.35, label='R²',
                               color=['#FF5722', '#4CAF50'], alpha=0.85)
            ax_bot2 = ax_bot.twinx()
            bars2 = ax_bot2.bar(x + 0.2, rmses, 0.35, label='RMSE(%)',
                                color=['#FF5722', '#4CAF50'], alpha=0.4)
            ax_bot.set_xticks(x)
            ax_bot.set_xticklabels(labels, fontsize=9)
            ax_bot.set_ylabel('R²', fontsize=9)
            ax_bot2.set_ylabel('RMSE (%)', fontsize=9)
            ax_bot.set_title('预测模型性能', fontsize=10)
            ax_bot.set_ylim(bottom=min(0, min(r2s) - 0.1))
            ax_bot.axhline(0, color='grey', linewidth=0.5)
            ax_bot.grid(True, alpha=0.3, axis='y')
            lines1, labels1 = ax_bot.get_legend_handles_labels()
            lines2, labels2 = ax_bot2.get_legend_handles_labels()
            ax_bot.legend(lines1 + lines2, labels1 + labels2,
                          fontsize=8, loc='upper right')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'summary_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  波兰 GPW 股票数据分析")
    print(f"  数据区间: {START_DATE.date()} ~ {END_DATE.date()}")
    print("=" * 60)

    # 1. 下载数据
    print("\n[1] 下载历史数据")
    all_data = download_data(STOCKS, START_DATE, END_DATE)

    if not all_data:
        print("所有股票数据下载失败，请检查网络连接。")
        return

    # 2. 计算指标 & 可视化
    print("\n[2] 计算技术指标 & 绘制图表")
    all_results = {}
    for ticker, df in all_data.items():
        name = STOCKS[ticker]
        print(f"  处理 {ticker} ...")
        df = calc_indicators(df)
        all_data[ticker] = df

        # 技术指标图
        p1 = plot_stock(ticker, df, name)
        print(f"    已保存: {os.path.basename(p1)}")

        # 3. 特征工程 + 模型训练
        try:
            X, y = build_features(df)
            if len(X) < 100:
                print(f"    ⚠ 数据量不足，跳过预测")
                continue
            results = train_predict(X, y, ticker)
            all_results[ticker] = results
            p2 = plot_predictions(ticker, name, results)
            print(f"    LR  R²={results['lr_r2']:.4f}  RMSE={results['lr_rmse']*100:.4f}%")
            print(f"    RF  R²={results['rf_r2']:.4f}  RMSE={results['rf_rmse']*100:.4f}%")
            print(f"    已保存: {os.path.basename(p2)}")
        except Exception as e:
            print(f"    预测失败: {e}")

    # 4. 汇总对比图
    print("\n[3] 生成汇总对比图")
    p3 = plot_summary(all_data, all_results)
    print(f"  已保存: {os.path.basename(p3)}")

    print("\n" + "=" * 60)
    print(f"  全部完成！图表保存在: {OUTPUT_DIR}")
    print("=" * 60)

    # 打印统计摘要
    print("\n── 统计摘要 ─────────────────────────────────────")
    for ticker, df in all_data.items():
        close = df['Close']
        ret   = (close.iloc[-1] / close.iloc[0] - 1) * 100
        vol   = close.pct_change().std() * np.sqrt(252) * 100
        print(f"  {ticker:<10} 总涨跌: {ret:+7.2f}%   年化波动率: {vol:.1f}%"
              f"   最新收盘: {close.iloc[-1]:.2f} PLN")

    if all_results:
        print("\n── 模型性能摘要 ─────────────────────────────────")
        print(f"  {'股票':<10} {'LR R²':>8} {'RF R²':>8} {'LR RMSE':>10} {'RF RMSE':>10}")
        for ticker, res in all_results.items():
            print(f"  {ticker:<10} {res['lr_r2']:>8.4f} {res['rf_r2']:>8.4f}"
                  f" {res['lr_rmse']*100:>9.4f}% {res['rf_rmse']*100:>9.4f}%")


if __name__ == '__main__':
    main()
