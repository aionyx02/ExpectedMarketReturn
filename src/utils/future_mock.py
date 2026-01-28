import pandas as pd
import numpy as np
import os
from dateutil.relativedelta import relativedelta
import logging
from config.path import PathConfig
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
def mock_future_data(target_date_str,
                     path_macro = PathConfig.MACRO_FACTOR_CSV,
                     path_market = PathConfig.MARKET_RETURN_CSV):
    """
    使用「均值回歸 (Mean Reversion) & 長期成長」邏輯來填補空白。。
    """
    target_date : pd.Timestamp = pd.to_datetime(target_date_str)
    logging.info(f" 啟動均值回歸推算：正在將數據平滑延伸至 {target_date.strftime('%Y-%m')}...")

    # ---------------------------------------------------------
    #  補齊 Macro Factors (macro_factor.csv)
    # ---------------------------------------------------------
    if os.path.exists(path_macro):
        df = pd.read_csv(path_macro, parse_dates=["date"])

        last_date = pd.Timestamp(df["date"].max())

        if last_date < target_date:
            new_rows = []
            curr_date = last_date

            last_val = float(df.iloc[-1]["macro_factor"])

            # 設定回歸目標
            target_mean = 1.0
            decay_rate = 0.1

            while curr_date < target_date:
                curr_date += relativedelta(months=1)
                if curr_date > target_date: break

                next_val = last_val + (target_mean - last_val) * decay_rate

                # 加入微小雜訊(暫時拔除 提高準確性)
                #noise = np.random.normal(0, 0.005)
                #next_val += noise

                mock_row = df.iloc[-1].copy()
                mock_row["date"] = curr_date
                mock_row["macro_factor"] = next_val

                new_rows.append(mock_row)
                last_val = next_val

            if new_rows:
                df_mock = pd.DataFrame(new_rows)
                df = pd.concat([df, df_mock], ignore_index=True)
                df.to_csv(path_macro, index=False)
                logging.info(f"    Macro Factor: 已依照均值回歸邏輯推算 (Target: 1.0)")

    # ---------------------------------------------------------
    #  Market Returns (market_return.csv)
    # ---------------------------------------------------------
    if os.path.exists(path_market):
        df = pd.read_csv(path_market, parse_dates=["date"])

        # 強制轉型
        last_date = pd.Timestamp(df["date"].max())

        if last_date < target_date:
            if "Close" not in df.columns:
                logging.error(" 錯誤: market_return.csv 缺少 Close 欄位")
            else:
                long_term_growth = 0.0058

                new_rows = []
                curr_date = last_date

                # 強制轉型為 float
                last_price = float(df.iloc[-1]["Close"])
                last_exp_ret = float(df.iloc[-1]["expected_return"])

                target_exp_ret = 0.05

                while curr_date < target_date:
                    curr_date += relativedelta(months=1)
                    if curr_date > target_date: break

                    monthly_change = long_term_growth + np.random.normal(0, 0.01)
                    new_price = last_price * (1 + monthly_change)

                    new_exp_ret = last_exp_ret + (target_exp_ret - last_exp_ret) * 0.1

                    mock_row = df.iloc[-1].copy()
                    mock_row["date"] = curr_date
                    mock_row["Close"] = new_price
                    mock_row["expected_return"] = new_exp_ret

                    # 這裡比較簡單，可以直接判斷
                    mock_row["trend_signal"] = True

                    new_rows.append(mock_row)
                    last_price = new_price
                    last_exp_ret = new_exp_ret

                if new_rows:
                    df_mock = pd.DataFrame(new_rows)
                    df = pd.concat([df, df_mock], ignore_index=True)
                    df.to_csv(path_market, index=False)
                    logging.info(f" Market Price: 已依照長期成長模型推算")