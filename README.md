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

<img width="1803" height="995" alt="螢幕擷取畫面 2026-01-27 185600" src="https://github.com/user-attachments/assets/c0eb5535-896a-4629-80b1-ba433c6a5027" />

<img width="2088" height="1141" alt="螢幕擷取畫面 2026-01-27 194307" src="https://github.com/user-attachments/assets/b6c62631-d152-4814-bac8-08ff37b294b7" />


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

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=white)
![yfinance](https://img.shields.io/badge/yfinance-Yahoo!-410099?style=for-the-badge&logo=yahoo&logoColor=white)

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
Profile customization (global-ready):
```sh
# Use default profile
python src/main.py

# Use a custom profile (PowerShell)
$env:EMR_PROFILE='profiles/global_default.yaml'; python src/main.py
```
You can copy `profiles/global_default.yaml` and change tickers, macro series, thresholds, and leverage rules per market.

Expected output:
```
==================================================
 【回測：動態槓桿】
==================================================
指標 (Metric)          | 大盤 (S&P 500)    | MVP 2x (Strategy)
------------------------------------------------------------
指標名稱                   |  Benchmark   |   Strategy  
----------------------------------------------------
總報酬率 (Total Ret)       |      3784.99% |     13662.56%
最大回撤 (Max DD)          |       -46.70% |       -22.47%
夏普比率 (Sharpe)          |         1.35  |         1.47 
------------------------------------------------------------
==================================================

2026-01-28 19:59:21.439 | INFO | root | [Step 10] Executing High-Frequency Nowcasting...

 數據基準日: 2026-01-28
--------------------------------------------------
 模型指標摘要:
   - 預期年化報酬: 4.57%
   - 宏觀風險修正: x0.95
   - 市場廣度狀態: HEALTHY
--------------------------------------------------
 修正後預期回報: 4.34%

 【推薦動作】
--------------------------------------------------
指令動態： 正常持有 (Neutral/Buy)
槓桿倍數：1.0x
建議配置：100% 部位投資於 SPY/VOO，0% 留存現金
理由詳述：環境穩健但回報空間一般，建議現貨持倉（SPY/VOO），不開槓桿。
--------------------------------------------------
```
Backtest compares strategy vs S&P 500 with dynamic leverage control.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

Roadmap

![MVP](https://img.shields.io/badge/MVP_Pipeline-Integration-005963?style=for-the-badge&logo=gitlabbici&logoColor=white)
![FRED](https://img.shields.io/badge/FRED-Macro_Data-990000?style=for-the-badge&logo=databricks&logoColor=white)
![Breadth](https://img.shields.io/badge/Market_Breadth-Diagnostics-2E8B57?style=for-the-badge&logo=analogue&logoColor=white)
![Line](https://img.shields.io/badge/LINE_Bot-Notification-00C300?style=for-the-badge&logo=line&logoColor=white)
![Telegram](https://img.shields.io/badge/Telegram_Bot-Notification-26A5E4?style=for-the-badge&logo=telegram&logoColor=white)
![Kelly](https://img.shields.io/badge/Kelly_Criterion-Position_Sizing-DAA520?style=for-the-badge&logo=scales&logoColor=white)
![Stress Test](https://img.shields.io/badge/Stress_Testing-2008_%2F_2020-8B0000?style=for-the-badge&logo=speedtest&logoColor=white)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

We welcome contributions from the community! To keep the codebase clean and functional, please follow these steps:

1. Development Process
Issue First: Before making major changes, please open an issue to discuss what you would like to change.

Branch Naming: Use descriptive names like fix/data-filling or feat/market-breadth.

2. Pull Request Guidelines
Style: Ensure your code follows PEP 8 standards.

Testing: If you add new financial indicators or stress test scenarios, please include corresponding unit tests.

Documentation: Update the README.md or docstrings if you introduce new API integrations (e.g., new FRED endpoints).

3. Getting Started
   .Fork the Project.  
   .Create your Feature Branch.  
   .Commit your Changes.  
   .Push to the Branch.  
   .Open a Pull Request.  

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
