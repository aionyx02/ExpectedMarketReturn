<a id="readme-top"></a>

---

[![Contributors](https://img.shields.io/github/contributors/aionyx02/ExpectedMarketReturn.svg?style=for-the-badge)](https://github.com/aionyx02/ExpectedMarketReturn/graphs/contributors)
[![Forks](https://img.shields.io/github/forks/aionyx02/ExpectedMarketReturn.svg?style=for-the-badge)](https://github.com/aionyx02/ExpectedMarketReturn/network/members)
[![Stars](https://img.shields.io/github/stars/aionyx02/ExpectedMarketReturn.svg?style=for-the-badge)](https://github.com/aionyx02/ExpectedMarketReturn/stargazers)
[![License](https://img.shields.io/github/license/aionyx02/ExpectedMarketReturn.svg?style=for-the-badge)](https://github.com/aionyx02/ExpectedMarketReturn/blob/master/LICENSE)

<br />

<div align="center">
  <h3 align="center">Expected Market Return</h3>

  <p align="center">
    MVP Quant Pipeline — A quantitative investment decision system integrating macro risk and market breadth.
    <br />
    <a href="https://github.com/aionyx02/ExpectedMarketReturn"><strong>Explore the docs »</strong></a>
    <br /><br />
    <a href="https://github.com/aionyx02/ExpectedMarketReturn">View Demo</a>
    ·
    <a href="https://github.com/aionyx02/ExpectedMarketReturn/issues">Report Bug</a>
    ·
    <a href="https://github.com/aionyx02/ExpectedMarketReturn/issues">Request Feature</a>
  </p>
</div>

---

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#key-features">Key Features</a></li>
    <li><a href="#methodology">Methodology</a></li>
    <li><a href="#built-with">Built With</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

---

## About The Project

**ExpectedMarketReturn** is a Python-based quantitative finance pipeline designed to mitigate emotional investing through objective data analysis.

The core philosophy:

> **Increase leverage when the macro environment is safe and the trend is positive;  
> reduce exposure when risks accumulate.**

The system integrates macro liquidity indicators (FRED), market performance (Yahoo Finance), and market breadth diagnostics.

<img width="2096" height="1142" alt="螢幕擷取畫面 2026-01-27 185542" src="https://github.com/user-attachments/assets/b88a033b-6b62-41fa-83db-465fb590a40a" />
<img width="1803" height="995" alt="螢幕擷取畫面 2026-01-27 185600" src="https://github.com/user-attachments/assets/b834e870-2ed6-442e-a6e4-d2366ff606ff" />

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Key Features

* **Multi-source Data Integration**  
  Combines FRED macroeconomic indicators (interest rates, labor market) with Yahoo Finance market returns and market breadth metrics.

* **Dynamic Leverage Decision Engine**  
  Automatically outputs:
    * **2x Bull**
    * **1x Neutral**
    * **0x Bear**

* **Scientific Gap Filling (Mean Reversion)**  
  Uses a mean-reversion mechanism to handle macro data release delays and avoid look-ahead bias.

* **Nowcasting Market Regime**  
  Generates daily diagnostic reports providing actionable leverage and allocation guidance.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Methodology

The core predictive framework is **MVB (Macro–Valuation–Breadth)**.

### 1. Macro Risk Adjustment (Macro Factor)

The system monitors:

- 10Y–2Y yield spread
- Unemployment claims trend

When yield spreads compress or employment weakens, the macro coefficient is reduced from `1.0`, compressing expected returns and shifting the system to defensive positioning.

---

### 2. Market Breadth Diagnosis

Designed to detect index fragility.

We compare:

- **Cap-weighted indices**
- **Equal-weighted indices**

Logic:

- **Broad participation** → leverage allowed
- **Narrow leadership** → signal classified as `FRAGILE`, leverage reduced to avoid structural risk

---

### 3. Data Gap Filling: Mean Reversion

Because FRED data is delayed (1–4 weeks), a simplified Ornstein–Uhlenbeck process is applied:

```math
X_{t+1} = X_t + \kappa(\theta - X_t)\Delta t
```
This ensures missing macro values converge smoothly to long-term equilibrium $\theta$ instead of producing extreme bias.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

Built With

Python

Pandas

NumPy

Matplotlib

yfinance

scipy

<p align="right">(<a href="#readme-top">back to top</a>)</p>

Getting Started
Prerequisites

Python 3.10+

pip or uv

Installation

```sh
git clone https://github.com/aionyx02/ExpectedMarketReturn.git
cd ExpectedMarketReturn
pip install pandas numpy matplotlib yfinance scipy python-dateutil
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

Usage
```sh
python main.py
```
Expected output:
```
============================================================
 【量化模型：市場診斷報告 】
============================================================
數據基準日: 2026-01-01
1️ 宏觀風險指數 : 1.00 ✅ 安全
2 預期年化報酬 : 4.66%
3️ 系統決策訊號 : 【NEUTRAL】
------------------------------------------------------------
 【最終執行指令】:
    建議: 1.0x 現貨 (SPY/VOO)
============================================================


[Step 8] Running Backtest...
 正在進行 Phase 4 回測：動態槓桿 (Dynamic Leverage)...

==================================================
 【Phase 4 回測：動態槓桿版】
==================================================
指標 (Metric)          | 大盤 (S&P 500)    | MVP 2x (Strategy)
------------------------------------------------------------
總報酬率 (Total Ret)     | 3769.19%          | 18172.18%
最大回撤 (Max DD)        | -46.70% (痛!)     | -22.47% (穩)
夏普比率 (Sharpe)        |   1.35            |   1.58
------------------------------------------------------------
 恭喜！動態槓桿策略成功【碾壓大盤】！
 關鍵：在牛市開 2 倍加速，在熊市 0 倍保命。
==================================================


==========================================
[Step 10] Executing High-Frequency Nowcasting...
==========================================
⚠️ 檢測到數據缺失，執行自動填充 (ffill)...

📅 數據基準日: 2026-01-27
--------------------------------------------------
📊 模型指標摘要:
   - 預期年化報酬: 4.66%
   - 宏觀風險修正: x1.00
   - 市場廣度狀態: HEALTHY
--------------------------------------------------
🎯 修正後預期回報: 4.66%

🚀 【MVP 實戰操作指令】
--------------------------------------------------
指令動態：🔵 正常持有 (Neutral/Buy)
槓桿倍數：1.0x
建議配置：100% 部位投資於 SPY/VOO，0% 留存現金
理由詳述：環境穩健但回報空間一般，建議 100% 現貨持倉（SPY/VOO），不開槓桿。
--------------------------------------------------
```
Backtest compares strategy vs S&P 500 with dynamic leverage control.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

Roadmap

MVP pipeline (FRED + Yahoo Finance integration)

Mean Reversion data filling

Market breadth diagnostics

Notification system (Line Bot / Telegram)

Kelly Criterion position sizing

Stress testing (2008, 2020 crash scenarios)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
Contributing

Contributions are welcome.

Fork the project

Create a branch

Commit changes

Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>
License

Distributed under the MIT License.
See LICENSE for details.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
Contact

Aionyx — GitHub
Project Link:
https://github.com/aionyx02/ExpectedMarketReturn

<p align="right">(<a href="#readme-top">back to top</a>)</p>

