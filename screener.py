import numpy as np
import yfinance as yf
from scipy.stats import linregress
import pandas as pd
import requests
import re
import html
import json
import math


def get_nasdaq():
    url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"

    df = pd.read_csv(url, sep="|")
    df = df[df["Symbol"] != "File Creation Time"]

    return df


def get_asx():
    session = requests.Session()

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/137.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
    }

    r = session.get(
        "https://www.marketindex.com.au/asx-listed-companies",
        headers=headers,
        timeout=30,
    )

    text = r.text

    # 1. Extract the Vue prop value
    match = re.search(r':companies="(.*?)"', text, re.DOTALL)

    if not match:
        raise ValueError("Companies data not found")

    raw = match.group(1)

    # 2. Decode HTML entities (&quot; → ")
    raw = html.unescape(raw)

    # 3. Parse JSON
    companies = json.loads(raw)

    # 4. DataFrame
    df = pd.DataFrame(companies)

    return df


def _get_horizon_dates(
    today_date: str,
    early_horizon_years: int,
    late_horizon_years: int,
):
    today = pd.Timestamp(today_date).normalize()
    early_horizon = today - pd.DateOffset(years=early_horizon_years)
    late_horizon = today - pd.DateOffset(years=late_horizon_years)

    if late_horizon < early_horizon:
        raise ValueError(
            "late_horizon_years must be less than or equal to early_horizon_years"
        )

    return today, early_horizon, late_horizon


def _download_adjusted_prices(
    ticker: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
):
    print("Downloading data...")
    df = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=(end_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )

    if len(df) < 10:
        raise ValueError(f"Insufficient data returned for {ticker}")

    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    price_df = pd.DataFrame(
        {"close": close.to_numpy(dtype=float).flatten()},
        index=df.index,
    )
    price_df = price_df.dropna()

    if len(price_df) < 10:
        raise ValueError(f"Insufficient closing price data returned for {ticker}")

    return price_df


# TODO: yahoo finance data is garbage. Need to look into robust (Huber) regression + volatility weighting
# maybe also build in time-decay weighting to discount earlier periods
# maybe look at "hybrid trust score" weighting per data point based on volatility, volume, outlier, and time
# maybe look at rolling regression to track over time "has this stock seemed to be around X% yearly return for a long time"


def _analyze_price_window(
    ticker: str,
    window_df: pd.DataFrame,
):
    if len(window_df) < 10:
        raise ValueError(f"Insufficient data returned for {ticker} in analysis window")

    prices = window_df["close"].to_numpy(dtype=float).flatten()

    # calendar-day axis
    dates = window_df.index
    x = np.asarray(
        (dates - dates[0]).days,
        dtype=float,
    )

    log_y = np.log(prices)

    fit = linregress(x, log_y)

    slope = fit.slope
    intercept = fit.intercept
    slope_stderr = fit.stderr

    fitted_log_y = intercept + slope * x
    fitted_price = np.exp(fitted_log_y)
    fitted_latest_price = fitted_price[-1]
    latest_price = prices[-1]

    residuals = log_y - fitted_log_y

    residual_volatility = np.std(
        residuals,
        ddof=1,
    )

    # Typical multiplicative deviation from trend
    trend_error_pct = (np.exp(residual_volatility) - 1.0) * 100.0

    # annualized growth estimate
    annual_growth = np.exp(365.25 * slope) - 1.0

    # 95% confidence interval
    z = 1.96

    slope_low = slope - z * slope_stderr
    slope_high = slope + z * slope_stderr

    annual_growth_low = np.exp(365.25 * slope_low) - 1.0

    annual_growth_high = np.exp(365.25 * slope_high) - 1.0

    metrics = {
        "n_observations": len(window_df),
        # Growth metrics
        "annualized_growth_pct": 100.0 * annual_growth,
        "annualized_growth_ci95_pct": (
            100.0 * annual_growth_low,
            100.0 * annual_growth_high,
        ),
        # Fit quality
        "r_squared": fit.rvalue**2,
        # Residual statistics
        "residual_volatility": residual_volatility,
        "trend_error_pct": trend_error_pct,
        # Additional diagnostics
        "daily_continuous_growth_rate": slope,
        "daily_growth_pct": 100.0 * (np.exp(slope) - 1.0),
        "slope_stderr": slope_stderr,
        "value_ratio": (latest_price - fitted_latest_price) / latest_price,
    }

    fit_df = pd.DataFrame(
        {"fitted_close": fitted_price},
        index=window_df.index,
    )

    return metrics, fit_df


def analyze_growth_stock_with_data(
    ticker: str,
    today_date: str,
    early_horizon_years: int,
    late_horizon_years: int,
):
    """
    Return growth-stock analysis plus chart-ready prices and fitted curves.
    """
    today, early_horizon, late_horizon = _get_horizon_dates(
        today_date,
        early_horizon_years,
        late_horizon_years,
    )

    price_df = _download_adjusted_prices(
        ticker,
        early_horizon,
        today,
    )

    early_df = price_df[price_df.index >= early_horizon]
    late_df = price_df[price_df.index >= late_horizon]

    early_metrics, early_fit_df = _analyze_price_window(ticker, early_df)
    late_metrics, late_fit_df = _analyze_price_window(ticker, late_df)

    return {
        "ticker": ticker,
        "today_date": today.strftime("%Y-%m-%d"),
        "early_horizon_date": early_horizon.strftime("%Y-%m-%d"),
        "late_horizon_date": late_horizon.strftime("%Y-%m-%d"),
        "early_horizon_years": early_horizon_years,
        "late_horizon_years": late_horizon_years,
        "early": early_metrics,
        "late": late_metrics,
        "prices": price_df,
        "fits": {
            "early": early_fit_df,
            "late": late_fit_df,
        },
    }


def aggregate_analysis(result: dict) -> float:
    score = 0

    early_growth = result["early"]["annualized_growth_pct"]
    late_growth = result["late"]["annualized_growth_pct"]

    early_r_squared = result["early"]["r_squared"]
    late_r_squared = result["late"]["r_squared"]

    score = math.log(early_growth) + math.log(late_growth)
    score += math.log(early_r_squared) + math.log(late_r_squared)

    return score


def _strip_chart_data(result: dict) -> dict:
    return {
        key: value for key, value in result.items() if key not in {"prices", "fits"}
    }


def _flatten_analysis_for_csv(result: dict) -> dict:
    row = {
        "ticker": result["ticker"],
        "today_date": result["today_date"],
        "early_horizon_date": result["early_horizon_date"],
        "late_horizon_date": result["late_horizon_date"],
        "early_horizon_years": result["early_horizon_years"],
        "late_horizon_years": result["late_horizon_years"],
    }

    for horizon in ("early", "late"):
        for metric_name, value in result[horizon].items():
            if metric_name == "annualized_growth_ci95_pct":
                low, high = value
                row[f"{horizon}_annualized_growth_ci95_pct_low"] = low
                row[f"{horizon}_annualized_growth_ci95_pct_high"] = high
            else:
                row[f"{horizon}_{metric_name}"] = value

    return row


def _asx_code_to_yfinance_ticker(code: str) -> str:
    return f"{str(code).strip().upper()}.AX"


def _yfinance_ticker_to_asx_code(ticker: str) -> str | None:
    normalized = str(ticker).strip().upper()
    if normalized.endswith(".AX"):
        return normalized[:-3]
    return None


def _extract_gics_sector(company_sector) -> str:
    if isinstance(company_sector, dict):
        return company_sector.get("gics_sector", "")
    return ""


def _asx_type_label(type_code) -> str:
    labels = {
        "01": "Listed company",
        "06": "Listed trust/stapled security",
        "07": "ETF",
        "16": "Structured product",
        "17": "Preference/security",
        "21": "Hybrid/security",
        "33": "ETF",
        "36": "Active ETF",
        "51": "Warrant/option/security",
        "53": "Exchange traded product",
    }
    return labels.get(str(type_code).strip(), "")


def get_asx_metadata_by_code() -> dict:
    asx = get_asx()
    if "code" not in asx.columns:
        raise ValueError("Expected get_asx() to return a 'code' column")

    metadata = {}

    for _, row in asx.iterrows():
        code = str(row["code"]).strip().upper()
        if not code:
            continue

        type_code = str(row.get("type", "")).strip()
        metadata[code] = {
            "asx_code": code,
            "asx_title": row.get("title", ""),
            "gics_sector": _extract_gics_sector(row.get("company_sector")),
            "asx_type_code": type_code,
            "asx_type_label": _asx_type_label(type_code),
        }

    return metadata


def get_asx_metadata_for_tickers(tickers: list[str]) -> dict:
    asx_codes = {
        code for ticker in tickers if (code := _yfinance_ticker_to_asx_code(ticker))
    }

    if not asx_codes:
        return {}

    metadata_by_code = get_asx_metadata_by_code()

    return {
        _asx_code_to_yfinance_ticker(code): metadata_by_code.get(code, {})
        for code in asx_codes
    }


def _chunks(items: list, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def analyze_growth_stock(
    ticker: str,
    today_date: str,
    early_horizon_years: int,
    late_horizon_years: int,
):
    """
    Fit an exponential model

        price = A * exp(k * t)

    to historical adjusted daily prices over early and late horizons and return:

    - annualized growth estimate
    - 95% confidence interval on annualized growth
    - R² of log-price fit
    - residual volatility
    - trend error (% deviation from trend)
    - current value ratio versus the fitted trend

    Parameters
    ----------
    ticker : str
        e.g. "AAPL"
    today_date : str
        YYYY-MM-DD
    early_horizon_years : int
        Number of years before today_date for the longer analysis window.
    late_horizon_years : int
        Number of years before today_date for the shorter analysis window.

    Returns
    -------
    dict
    """
    result = analyze_growth_stock_with_data(
        ticker,
        today_date,
        early_horizon_years,
        late_horizon_years,
    )

    return {
        key: value for key, value in result.items() if key not in {"prices", "fits"}
    }


if __name__ == "__main__":
    ticker = "AAPL"
    today_date = "2026-05-31"
    early_horizon_years = 5
    late_horizon_years = 1
    print("About to run...")
    result = analyze_growth_stock(
        ticker,
        today_date,
        early_horizon_years,
        late_horizon_years,
    )
    print(result)
