import os
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from screener import (
    get_asx_metadata_by_code,
    analyze_growth_stock_with_data,
    aggregate_analysis,
)
from screener import (
    _asx_code_to_yfinance_ticker,
    _strip_chart_data,
    _flatten_analysis_for_csv,
    _chunks,
)


def _analyze_asx_ticker_for_csv(
    code: str,
    today_date: str,
    early_horizon_years: int,
    late_horizon_years: int,
    metadata: dict | None = None,
) -> dict:
    ticker = _asx_code_to_yfinance_ticker(code)
    metadata = metadata or {}
    base_row = {
        "asx_code": str(code).strip().upper(),
        "ticker": ticker,
        "asx_title": metadata.get("asx_title", ""),
        "gics_sector": metadata.get("gics_sector", ""),
        "asx_type_code": metadata.get("asx_type_code", ""),
        "asx_type_label": metadata.get("asx_type_label", ""),
        "status": "ok",
        "error": "",
        "aggregate_error": "",
    }

    try:
        result_with_data = analyze_growth_stock_with_data(
            ticker,
            today_date,
            early_horizon_years,
            late_horizon_years,
        )
        result = _strip_chart_data(result_with_data)
        row = {
            **base_row,
            **_flatten_analysis_for_csv(result),
        }

        try:
            row["aggregate_score"] = aggregate_analysis(result)
        except Exception as exc:
            row["aggregate_score"] = np.nan
            row["aggregate_error"] = str(exc)

        return row
    except Exception as exc:
        return {
            **base_row,
            "status": "error",
            "error": str(exc),
            "aggregate_score": np.nan,
            "today_date": today_date,
            "early_horizon_years": early_horizon_years,
            "late_horizon_years": late_horizon_years,
        }


def analyze_all_asx_tickers_to_csv(
    output_dir: str = "data",
    today_date: str | None = None,
    early_horizon_years: int = 5,
    late_horizon_years: int = 1,
    batch_size: int = 50,
    max_workers: int | None = None,
    limit: int | None = None,
) -> str:
    """
    Analyze all current ASX tickers and write flattened metrics to a dated CSV.

    The task is mostly network IO, with a small regression/scoring step per ticker,
    so it uses bounded thread pools in batches rather than launching all downloads
    at once.
    """
    if today_date is None:
        today_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    if max_workers is None:
        cpu_count = os.cpu_count() or 1
        max_workers = min(16, max(4, cpu_count * 4))

    metadata_by_code = get_asx_metadata_by_code()

    codes = [code for code in metadata_by_code if code]

    if limit is not None:
        codes = codes[:limit]

    output_path = Path(output_dir) / f"asx_analysis_{today_date}.csv"
    rows = []

    for batch_number, batch in enumerate(_chunks(codes, batch_size), start=1):
        print(
            f"Processing batch {batch_number}: "
            f"{len(batch)} tickers with {max_workers} workers"
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _analyze_asx_ticker_for_csv,
                    code,
                    today_date,
                    early_horizon_years,
                    late_horizon_years,
                    metadata_by_code.get(code, {}),
                )
                for code in batch
            ]

            for future in as_completed(futures):
                rows.append(future.result())

    df = pd.DataFrame(rows)
    df = df.sort_values(
        by=["aggregate_score", "ticker"],
        ascending=[False, True],
        na_position="last",
    )
    df.to_csv(output_path, index=False)

    return str(output_path)


if __name__ == "__main__":
    analyze_all_asx_tickers_to_csv()
