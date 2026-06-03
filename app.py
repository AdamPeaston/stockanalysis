from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from screener import analyze_growth_stock_with_data
from screener import aggregate_analysis
from screener import get_asx_metadata_for_tickers


METRIC_COLUMNS = [
    "n_observations",
    "annualized_growth_pct",
    "annualized_growth_ci95_pct_low",
    "annualized_growth_ci95_pct_high",
    "r_squared",
    "residual_volatility",
    "trend_error_pct",
    "daily_continuous_growth_rate",
    "daily_growth_pct",
    "slope_stderr",
    "value_ratio",
]


def parse_tickers(raw_tickers: str):
    tickers = raw_tickers.replace(",", "\n").splitlines()
    return [
        ticker.strip().upper()
        for ticker in tickers
        if ticker.strip()
    ]


def flatten_metrics(result):
    row = {
        "show_chart": False,
        "ticker": result["ticker"],
        "aggregate_score": result["aggregate_score"],
        "gics_sector": result.get("gics_sector", ""),
        "asx_type_code": result.get("asx_type_code", ""),
        "asx_type_label": result.get("asx_type_label", ""),
        "today_date": result["today_date"],
        "early_horizon_date": result["early_horizon_date"],
        "late_horizon_date": result["late_horizon_date"],
    }

    for horizon in ("early", "late"):
        metrics = result[horizon]

        for metric_name, value in metrics.items():
            if metric_name == "annualized_growth_ci95_pct":
                low, high = value
                row[f"{horizon}_annualized_growth_ci95_pct_low"] = low
                row[f"{horizon}_annualized_growth_ci95_pct_high"] = high
            else:
                row[f"{horizon}_{metric_name}"] = value

    return row


@st.cache_data(show_spinner=False)
def analyze_tickers(
    tickers,
    today_date: str,
    early_horizon_years: int,
    late_horizon_years: int,
):
    results = {}
    errors = {}
    asx_metadata = get_asx_metadata_for_tickers(tickers)

    for ticker in tickers:
        try:
            result = analyze_growth_stock_with_data(
                ticker,
                today_date,
                early_horizon_years,
                late_horizon_years,
            )
            result.update(asx_metadata.get(ticker, {}))
            metrics = {
                key: value
                for key, value in result.items()
                if key not in {"prices", "fits"}
            }
            result["aggregate_score"] = aggregate_analysis(result)
            results[ticker] = result
        except Exception as exc:
            errors[ticker] = str(exc)

    table = pd.DataFrame(
        [
            flatten_metrics(result)
            for result in results.values()
        ]
    )

    return results, table, errors


def make_chart(result):
    prices = result["prices"]
    early_fit = result["fits"]["early"]
    late_fit = result["fits"]["late"]
    late_horizon_date = pd.Timestamp(result["late_horizon_date"])
    late_prices = prices[prices.index >= late_horizon_date]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=prices.index,
            y=prices["close"],
            mode="lines",
            name="Adjusted close",
            line={"color": "rgba(130, 130, 130, 0.55)", "width": 2},
        )
    )

    fig.add_trace(
        go.Scatter(
            x=late_prices.index,
            y=late_prices["close"],
            mode="lines",
            name="Late horizon close",
            line={"color": "#1f77b4", "width": 2.5},
        )
    )

    fig.add_trace(
        go.Scatter(
            x=early_fit.index,
            y=early_fit["fitted_close"],
            mode="lines",
            name="Early fitted curve",
            line={"color": "#ff7f0e", "dash": "dash", "width": 2},
        )
    )

    fig.add_trace(
        go.Scatter(
            x=late_fit.index,
            y=late_fit["fitted_close"],
            mode="lines",
            name="Late fitted curve",
            line={"color": "#2ca02c", "dash": "dot", "width": 2},
        )
    )

    fig.update_layout(
        title=result["ticker"],
        xaxis_title="Date",
        yaxis_title="Adjusted close",
        height=460,
        margin={"l": 32, "r": 20, "t": 56, "b": 36},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
    )

    return fig


def main():
    st.set_page_config(
        page_title="Stock Growth Screener",
        layout="wide",
    )

    st.title("Stock Growth Screener")
    st.caption(
        "Compare early and late exponential growth fits across stocks or ETFs."
    )

    with st.sidebar:
        st.header("Inputs")
        raw_tickers = st.text_area(
            "Tickers",
            value="AAPL\nMSFT\nNVDA\nNAB.AX",
            help="Enter one ticker per line, or separate tickers with commas.",
            height=140,
        )
        today = st.date_input(
            "Today date",
            value=date.today(),
        )
        early_horizon_years = st.number_input(
            "Early horizon years",
            min_value=1,
            max_value=50,
            value=5,
            step=1,
        )
        late_horizon_years = st.number_input(
            "Late horizon years",
            min_value=1,
            max_value=50,
            value=1,
            step=1,
        )
        run_analysis = st.button(
            "Run analysis",
            type="primary",
            use_container_width=True,
        )

    tickers = parse_tickers(raw_tickers)

    if not tickers:
        st.info("Enter at least one ticker to begin.")
        return

    if late_horizon_years > early_horizon_years:
        st.error("Late horizon years must be less than or equal to early horizon years.")
        return

    if not run_analysis and "analysis_inputs" not in st.session_state:
        st.info("Set your inputs, then click Run analysis.")
        return

    if run_analysis:
        st.session_state["analysis_inputs"] = (
            tuple(tickers),
            today.strftime("%Y-%m-%d"),
            int(early_horizon_years),
            int(late_horizon_years),
        )

    analysis_inputs = st.session_state["analysis_inputs"]

    with st.spinner("Downloading data and running analysis..."):
        results, table, errors = analyze_tickers(*analysis_inputs)

    if errors:
        st.warning("Some tickers could not be analysed.")
        st.dataframe(
            pd.DataFrame(
                [
                    {"ticker": ticker, "error": error}
                    for ticker, error in errors.items()
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

    if table.empty:
        st.error("No ticker analyses completed successfully.")
        return

    st.subheader("Results")
    st.write(
        "Sort by any column, then tick `show_chart` for each ticker you want to plot."
    )

    column_order = (
        [
            "show_chart",
            "ticker",
            "aggregate_score",
            "gics_sector",
            "asx_type_code",
            "asx_type_label",
            "today_date",
            "early_horizon_date",
            "late_horizon_date",
        ]
        + [f"early_{column}" for column in METRIC_COLUMNS]
        + [f"late_{column}" for column in METRIC_COLUMNS]
    )

    edited_table = st.data_editor(
        table[column_order],
        column_config={
            "show_chart": st.column_config.CheckboxColumn(
                "show_chart",
                help="Tick to display this ticker's chart below.",
            ),
        },
        disabled=[
            column
            for column in column_order
            if column != "show_chart"
        ],
        hide_index=True,
        use_container_width=True,
        height=420,
    )

    selected_tickers = edited_table.loc[
        edited_table["show_chart"],
        "ticker",
    ].tolist()

    if not selected_tickers:
        st.info("Tick one or more rows to display charts.")
        return

    st.subheader("Charts")
    chart_columns = st.columns(len(selected_tickers))

    for column, ticker in zip(chart_columns, selected_tickers):
        with column:
            st.plotly_chart(
                make_chart(results[ticker]),
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
