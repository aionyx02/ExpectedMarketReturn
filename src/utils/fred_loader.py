import pandas as pd
import requests
import io
import os
from functools import reduce
import logging
from config.path import PathConfig
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
def update_all_fred(output_dir = PathConfig.RAW_DATA_DIR):
    """
    從 FRED 官網下載 CSV 數據 (Requests Mode)。
    """
    logging.info("   [FRED] 開始下載最新宏觀數據 (Requests Mode)...")

    series_map = {
        'DGS10': '10Y_Yield',        # 10年期公債殖利率
        'DGS2': '2Y_Yield',          # 2年期公債殖利率
        'ICSA': 'Jobless_Claims',    # 初領失業金人數
        'T10Y2Y': 'Yield_Spread'     # 10Y-2Y 利差
    }

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    data_frames = []

    try:
        session = requests.Session()
        session.headers.update(headers)

        for fred_code, col_name in series_map.items():
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={fred_code}"
            logging.info(f"   - Fetching {col_name} ({fred_code})...")

            try:
                response = session.get(url, timeout=10)

                if response.status_code != 200:
                    logging.info(f"  HTTP 錯誤 {response.status_code}: {url}")
                    continue

                content_text = response.text

                if "DATE" not in content_text[:50] and "observation_date" not in content_text[:50]:
                    logging.warning(f" 下載內容異常 (可能是 HTML 錯誤頁):")
                    logging.info(f"     內容預覽: {content_text[:100]}...")
                    continue

                # 使用 io.StringIO 讀取
                csv_data = io.StringIO(content_text)

                df = pd.read_csv(csv_data, na_values='.')

                if 'observation_date' in df.columns:
                    df = df.rename(columns={'observation_date': 'DATE'})

                # 確保有 DATE 欄位才繼續
                if 'DATE' not in df.columns:
                    logging.warning(f"  CSV 缺少日期欄位，跳過。欄位: {df.columns}")
                    continue

                # 轉換日期格式並設為 Index
                df['DATE'] = pd.to_datetime(df['DATE'])
                df = df.set_index('DATE')

                # 重新命名數值欄位 (例如 DGS10 -> 10Y_Yield)
                df = df.rename(columns={fred_code: col_name})

                # 強制轉為數值
                df[col_name] = pd.to_numeric(df[col_name], errors='coerce')

                data_frames.append(df)

            except Exception as e_inner:
                logging.error(f"下載或解析失敗 {fred_code}: {e_inner}")
                continue

        if not data_frames:
            raise ValueError("所有數據下載皆失敗。")

        logging.info("   [System] 合併數據中...")
        # 合併所有 DataFrame
        df_merged = reduce(lambda left, right: pd.merge(left, right, on='DATE', how='outer', sort=True), data_frames)

        # 重置索引，讓 DATE 變回欄位以便存檔
        df_merged = df_merged.reset_index()


        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_path = os.path.join(output_dir, "fred_raw.csv")
        df_merged.to_csv(output_path, index=False)

        logging.info(f"   [FRED] 下載成功！數據已儲存至: {output_path}")
        # 顯示最新幾筆數據的日期，確認是否為最新的
        last_date = df_merged['DATE'].max()
        logging.info(f"   [FRED] 最新數據日期: {last_date.strftime('%Y-%m-%d')}")

    except Exception as e:
        logging.error(f"   [Error] FRED 流程中止: {e}")
        pass

if __name__ == "__main__":
    update_all_fred()