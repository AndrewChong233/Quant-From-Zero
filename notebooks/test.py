# ========== 第一章爽点：下载真实股票数据 ==========
import os
import warnings                           # 导入警告控制模块
warnings.filterwarnings('ignore')   # 隐藏不影响学习的警告信息

import matplotlib.pyplot as plt     # 画图
import pandas as pd
import requests                     # 导入请求模块用于伪装
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import yfinance as yf               # 从雅虎财经免费下载行情

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False      # 坐标轴负号正常显示

# ---------- 🛠️ 核心防封优化：创建带伪装和重试功能的 Session ----------
session = requests.Session()
# 伪装成真实的 Chrome 浏览器 User-Agent
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
})

# 设置自动重试策略（如果请求失败，最多重试 3 次，间隔指数增加）
retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))
# ----------------------------------------------------------------------

# 本地缓存文件名
cache_file = "aapl_6mo_data.csv"

# 优先从本地缓存读取，如果没有才从雅虎下载
if os.path.exists(cache_file):
    aapl = pd.read_csv(cache_file, index_col=0, parse_dates=True)
    print('📦 成功从本地缓存读取数据！')
else:
    # 传入 session 进行伪装下载
    aapl = yf.download('AAPL', period='6mo', progress=False, multi_level_index=False, session=session)
    if not aapl.empty:
        aapl.to_csv(cache_file)  # 下载成功后保存一份到本地
        print('🎉 恭喜！你已通过伪装请求成功拿到真实股票数据，并已缓存至本地')

# 检查数据是否成功加载
if aapl.empty:
    print("❌ 数据下载失败或仍处于限制期，建议开启/切换 VPN 节点后再试。")
else:
    print(f'    共 {len(aapl)} 个交易日')                               # 行数 = 交易日个数
    print(f'    最新收盘价: ${aapl["Close"].iloc[-1]:.2f}')            # iloc[-1] = 最后一行
    print(aapl.tail(5))   # 在 Notebook 里美观地显示最后 5 行表格

    # ========== 上图收盘价、下图成交量 ==========
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True,       # 2行子图，横轴对齐
                             gridspec_kw={'height_ratios': [3, 1]})    # 上图占 3 份高度
    axes[0].plot(aapl.index, aapl['Close'], color='tab:blue', linewidth=1.5)  # 折线：收盘价
    axes[0].set_title('真实数据 · 苹果 AAPL 收盘价', fontsize=14)  # 设置上图标题
    axes[0].set_ylabel('美元')  # 设置上图纵轴
    axes[0].grid(True, alpha=0.3)  # 上图显示网格

    axes[1].bar(aapl.index, aapl['Volume'], width=0.8, color='gray', alpha=0.5)  # 柱状：成交量
    axes[1].set_ylabel('成交量')  # 设置下图纵轴
    axes[1].set_xlabel('日期')  # 设置下图横轴（日期）
    axes[1].grid(True, alpha=0.3)  # 下图显示网格

    plt.tight_layout()                       # 自动调整子图间距，避免标签被裁切
    plt.show()

    # ========== 第二章：均线交叉策略回测 ==========

    # 计算均线
    aapl['MA5'] = aapl['Close'].rolling(5).mean()
    aapl['MA20'] = aapl['Close'].rolling(20).mean()

    # 生成交易信号：MA5 > MA20 为 1（持仓），否则为 0（空仓）
    aapl['Signal'] = (aapl['MA5'] > aapl['MA20']).astype(int)

    # 计算每日收益率
    aapl['Return'] = aapl['Close'].pct_change()

    # 策略收益：前一天的信号 * 当天收益
    aapl['Strategy'] = aapl['Signal'].shift(1) * aapl['Return']

    # 累积资金曲线
    fig, ax = plt.subplots(figsize=(12, 6))
    (aapl[['Return', 'Strategy']].cumsum()).plot(ax=ax)
    ax.set_title("MA5 vs MA20 策略回测资金曲线", fontsize=14)
    ax.set_ylabel("累计收益率")
    ax.grid(True, alpha=0.3)
    plt.show()

    # 输出绩效指标
    total_return = aapl['Strategy'].sum()
    annual_return = aapl['Strategy'].mean() * 252
    max_drawdown = (aapl['Strategy'].cumsum().cummax() - aapl['Strategy'].cumsum()).max()
    sharpe_ratio = (aapl['Strategy'].mean() / aapl['Strategy'].std()) * (252 ** 0.5)

    print('MA5 vs MA20')
    print(f"总收益: {total_return:.2%}")
    print(f"年化收益: {annual_return:.2%}")
    print(f"最大回撤: {max_drawdown:.2%}")
    print(f"夏普比率: {sharpe_ratio:.2f}")


# ========== 第三章：MA10 vs MA50 策略回测 ==========

# 计算均线
aapl['MA10'] = aapl['Close'].rolling(10).mean()
aapl['MA50'] = aapl['Close'].rolling(50).mean()

# 生成交易信号：MA10 > MA50 为 1（持仓），否则为 0（空仓）
aapl['Signal_MA10_50'] = (aapl['MA10'] > aapl['MA50']).astype(int)

# 计算每日收益率
aapl['Return'] = aapl['Close'].pct_change()

# 策略收益：前一天的信号 * 当天收益
aapl['Strategy_MA10_50'] = aapl['Signal_MA10_50'].shift(1) * aapl['Return']

# 累积资金曲线
fig, ax = plt.subplots(figsize=(12,6))
(aapl[['Return','Strategy_MA10_50']].cumsum()).plot(ax=ax)
ax.set_title("MA10 vs MA50 策略回测资金曲线", fontsize=14)
ax.set_ylabel("累计收益率")
ax.grid(True, alpha=0.3)
plt.show()

# 输出绩效指标
total_return = aapl['Strategy_MA10_50'].sum()
annual_return = aapl['Strategy_MA10_50'].mean() * 252
max_drawdown = (aapl['Strategy_MA10_50'].cumsum().cummax() - aapl['Strategy_MA10_50'].cumsum()).max()
sharpe_ratio = (aapl['Strategy_MA10_50'].mean() / aapl['Strategy_MA10_50'].std()) * (252**0.5)

print('MA10 vs MA50')
print(f"总收益: {total_return:.2%}")
print(f"年化收益: {annual_return:.2%}")
print(f"最大回撤: {max_drawdown:.2%}")
print(f"夏普比率: {sharpe_ratio:.2f}")

