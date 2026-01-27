<a id="readme-top"></a>
<br />
<div align="center">
  <h3 align="center">Expected Market Return</h3>

  <p align="center">
    A quantitative investment strategy pipeline leveraging macroeconomic data and technical analysis to optimize market returns via dynamic leverage.
    <br />
    <a href="https://github.com/aionyx02/ExpectedMarketReturn"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/aionyx02/ExpectedMarketReturn">View Demo</a>
    &middot;
    <a href="https://github.com/aionyx02/ExpectedMarketReturn/issues">Report Bug</a>
    &middot;
    <a href="https://github.com/aionyx02/ExpectedMarketReturn/issues">Request Feature</a>
  </p>
</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## About The Project

**ExpectedMarketReturn** is a Python-based quantitative finance pipeline designed to solve the problem of emotional investing. It combines macroeconomic liquidity indicators (M2, GDP) with technical trend analysis to generate actionable trading signals.

The core philosophy is simple: **Increase leverage when the macro environment is safe and the trend is up; protect capital when risks are high.**
<img width="2097" height="1135" alt="螢幕擷取畫面 2026-01-27 133104" src="https://github.com/user-attachments/assets/8ba42048-c996-41e7-ae0a-04f2a97164f8" />
<img width="1795" height="989" alt="螢幕擷取畫面 2026-01-27 133026" src="https://github.com/user-attachments/assets/c49a5680-bd34-459a-b2af-2fd67e56fa5a" />
### Key Features

* **Real-World Data Integration**: Fetches live macroeconomic data from **FRED** (Federal Reserve Economic Data) and market data from **Yahoo Finance**.
* **Scientific Gap Filling (Mean Reversion)**: Utilizes a **Mean Reversion** algorithm to intelligently bridge the gap between lagged real-world data (e.g., 2025) and the current system date. This prevents "look-ahead bias" while providing realistic simulations.
* **Dynamic Leverage Strategy**: Implements a strategy that automatically switches between:
    * **2x Bull**: (e.g., SSO/Futures) when Macro is safe + Trend is up.
    * **1x Neutral**: (e.g., SPY) when market is uncertain.
    * **0x Bear**: (e.g., Cash/SHV) when risks are high.
* **Automated Diagnosis**: Generates a console-based market diagnosis report and visualizes backtest performance against the S&P 500 benchmark.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

This project utilizes a modern Python data science stack:

* [![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
* [![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
* [![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
* [![Matplotlib](https://img.shields.io/badge/Matplotlib-ffffff?style=for-the-badge&logo=matplotlib&logoColor=black)](https://matplotlib.org/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

You need Python 3.10 or higher installed on your machine. This project utilizes `uv` for package management, but standard `pip` works perfectly as well.

### Installation

1.  Clone the repo
    ```sh
    git clone [https://github.com/aionyx02/ExpectedMarketReturn.git](https://github.com/aionyx02/ExpectedMarketReturn.git)
    ```
2.  Navigate to the project directory
    ```sh
    cd ExpectedMarketReturn
    ```
3.  Install dependencies
    ```sh
    pip install pandas numpy matplotlib yfinance scipy python-dateutil
    ```
    *(Or if you use `uv`, simply run `uv sync`)*

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

Run the main pipeline to fetch data, calculate signals, and generate the report:
Expected Output: The script will output a console report diagnosing the current market status and generate a visualization chart in a pop-up window.
```sh
python main.py
==========================================
 MVP Quant Pipeline: Scientific Trend Projection
 Target Date : 2026-01-27 (Auto-Detected)
==========================================
...
[Step 7] Analyzing Market Status...

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
總報酬率 (Total Ret)     | 3769.19%          | 12208.90%
最大回撤 (Max DD)        | -46.70% (痛!)     | -20.98% (穩)
夏普比率 (Sharpe)        |   1.35            |   1.44
------------------------------------------------------------
 恭喜！動態槓桿策略成功【碾壓大盤】！
 關鍵：在牛市開 2 倍加速，在熊市 0 倍保命。
==================================================

```
Roadmap
[x] MVP Release: Core pipeline with FRED/Yahoo data integration.

[x] Scientific Data Filling: Implemented Mean Reversion for handling data lag.

[x] Project Structure: Optimized project layout and git configuration.

[ ] Docker Support: Containerize the application for cloud deployment.

[ ] Notification System: Integration with Line Bot / Telegram for daily alerts.

[ ] Expanded Indicators:

[ ] VIX (Volatility Index)

[ ] Corporate Earnings (EPS)

[ ] Bitcoin Correlation

See the open issues for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

Contributing
Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

Fork the Project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your Changes (git commit -m 'Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

License
Distributed under the MIT License. See LICENSE for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

Contact
Aionyx - GitHub Profile

Project Link: https://github.com/aionyx02/ExpectedMarketReturn

<p align="right">(<a href="#readme-top">back to top</a>)</p>

Acknowledgments
FRED (Federal Reserve Economic Data)

Yahoo Finance (yfinance)

<p align="right">(<a href="#readme-top">back to top</a>)</p> 
