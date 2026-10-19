"""Streamlit dashboard for Bluesky post sentiment analytics."""

import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import requests
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, adfuller, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

pio.templates.default = "plotly_white"

# Resolve data paths from the project root so the app works from any cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scraper.collector import BlueskyCollector

# Page configuration
st.set_page_config(
    page_title="Bluesky Sentiment Analytics",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded",
)

EXPORTS_DIR = PROJECT_ROOT / "data" / "exports"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
HISTORY_DIR = PROJECT_ROOT / "data" / "history"
FORECAST_MENTION_PATTERNS = {
    "ChatGPT": r"\bchat[\s-]?gpt\b",
    "OpenAI": r"\bopen\s*ai\b",
    "iPhone": r"\biphone\b",
}

st.markdown(
    """
    <style>
    :root {
        --ink: #262632;
        --mint: #eaf6f4;
        --coral: #ff806c;
        --green: #83d990;
        --paper: #ffffff;
    }
    [data-testid="stAppViewContainer"] { background: var(--mint); color: var(--ink); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] {
        background: var(--ink); width: 17rem !important;
        min-width: 17rem !important; max-width: 17rem !important;
    }
    [data-testid="stSidebar"] * { color: #f7f8fa; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label { padding: 0.18rem 0; }
    [data-testid="stAppViewContainer"] .main p,
    [data-testid="stAppViewContainer"] .main label { color: var(--ink) !important; }
    [data-testid="stAppViewContainer"] .main [data-baseweb="select"] > div {
        background: var(--paper) !important; border-color: #c6e0dc !important;
    }
    [data-testid="stAppViewContainer"] .main [data-baseweb="select"] * { color: var(--ink) !important; }
    .block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem; }
    h1, h2, h3 { color: var(--ink); font-family: 'Trebuchet MS', sans-serif; }
    [data-testid="stMetric"] {
        background: var(--paper); border-top: 5px solid var(--ink);
        padding: 1rem 1.1rem; min-height: 112px;
    }
    [data-testid="stMetricLabel"] { color: #5d626b; }
    [data-testid="stMetricValue"] { color: var(--ink); }
    [data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {
        background: var(--paper); padding: 0.55rem; border-radius: 5px;
    }
    [data-testid="stAlert"] { border-radius: 5px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def show_plotly(figure):
    """Render a Plotly figure using the dashboard's light chart palette."""
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"color": "#262632"},
    )
    st.plotly_chart(figure, use_container_width=True, theme=None)


@st.cache_data
def load_data():
    """Load Spark JSON output, including files nested in Spark output folders."""
    data_files = sorted(EXPORTS_DIR.rglob("*.json"))
    
    if not data_files:
        return pd.DataFrame()
    
    all_data = []
    for file in data_files:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    all_data.append(json.loads(line))
                except (json.JSONDecodeError, TypeError):
                    continue
    
    df = pd.DataFrame(all_data)
    
    if not df.empty:
        if "created_at" in df.columns:
                df["created_at"] = pd.to_datetime(df["created_at"], format="mixed", errors="coerce", utc=True)
        for column, default in (("username", "unknown"), ("text", ""), ("sentiment", "Neutral"),
                                ("keyword_used", "Uncategorized"), ("likes", 0), ("retweets", 0)):
            if column not in df.columns:
                df[column] = default
        if "tweet_id" in df.columns:
            df = df.drop_duplicates("tweet_id", keep="last")
    
    return df


@st.cache_data
def load_history():
    """Load the saved raw history archive without requiring a live API connection."""
    history_files = sorted(HISTORY_DIR.glob("*.jsonl"))
    records = []
    for file in history_files:
        with file.open("r", encoding="utf-8") as source:
            for line in source:
                try:
                    records.append(json.loads(line))
                except (json.JSONDecodeError, TypeError):
                    continue

    history = pd.DataFrame(records)
    if not history.empty:
        history["created_at"] = pd.to_datetime(history.get("created_at"), format="mixed", errors="coerce", utc=True)
        history = history.dropna(subset=["created_at", "keyword_used"])
        if {"tweet_id", "keyword_used"}.issubset(history.columns):
            history = history.drop_duplicates(["tweet_id", "keyword_used"], keep="last")
    return history


def combine_posts(df, history):
    """Combine processed posts and the local archive, preferring analyzed records."""
    columns = ("tweet_id", "created_at", "username", "text", "likes", "retweets", "keyword_used", "sentiment")
    frames = []
    for source in (history, df):
        if source.empty or not {"created_at", "keyword_used"}.issubset(source.columns):
            continue
        frame = source.reindex(columns=columns).copy()
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=columns)

    posts = pd.concat(frames, ignore_index=True)
    posts["created_at"] = pd.to_datetime(posts["created_at"], format="mixed", errors="coerce", utc=True)
    posts = posts.dropna(subset=["created_at", "keyword_used"])
    posts["text"] = posts["text"].fillna("").astype(str)
    posts["username"] = posts["username"].fillna("unknown")
    for column in ("likes", "retweets"):
        posts[column] = pd.to_numeric(posts[column], errors="coerce").fillna(0)

    posts["_dedupe_key"] = posts["tweet_id"].where(
        posts["tweet_id"].notna(),
        posts["created_at"].astype(str) + "|" + posts["text"],
    )
    posts = posts.drop_duplicates(["_dedupe_key", "keyword_used"], keep="last").drop(columns="_dedupe_key")

    missing_sentiment = posts["sentiment"].isna() | posts["sentiment"].astype(str).str.strip().eq("")
    if missing_sentiment.any():
        analyzer = SentimentIntensityAnalyzer()
        scores = posts.loc[missing_sentiment, "text"].map(
            lambda text: analyzer.polarity_scores(text)["compound"]
        )
        posts.loc[missing_sentiment, "sentiment"] = np.select(
            [scores >= 0.05, scores <= -0.05],
            ["Positive", "Negative"],
            default="Neutral",
        )
    return posts


@st.cache_data
def load_analytics():
    """Load the newest CSV from each Spark output directory."""
    data = {}
    analytics_paths = {
        "hourly": "hourly_sentiment",
        "daily": "daily_sentiment",
        "volume": "tweet_volume",
        "keyword": "keyword_sentiment",
    }
    for key, prefix in analytics_paths.items():
        output_dirs = sorted(PROCESSED_DATA_DIR.glob(f"{prefix}_*"), key=lambda path: path.stat().st_mtime)
        files = list(output_dirs[-1].rglob("*.csv")) if output_dirs else []
        flat_files = list(PROCESSED_DATA_DIR.glob(f"{prefix}_*.csv"))
        candidates = files or flat_files
        if candidates:
            try:
                newest_file = max(candidates, key=lambda path: path.stat().st_mtime)
                data[key] = pd.read_csv(newest_file)
            except (OSError, pd.errors.ParserError):
                data[key] = pd.DataFrame()

    return data


def topic_monthly_series(df, topic):
    """Build a monthly count series for posts collected under one search keyword."""
    topic_df = df.loc[df["keyword_used"].astype(str) == topic].copy()
    topic_df = topic_df.dropna(subset=["created_at"])
    if topic_df.empty:
        return pd.Series(dtype=float)

    months = topic_df["created_at"].dt.tz_convert(None).dt.to_period("M")
    monthly = topic_df.groupby(months).size()
    calendar = pd.period_range(monthly.index.min(), monthly.index.max(), freq="M")
    values = monthly.reindex(calendar, fill_value=0).astype(float)
    values.index = calendar.to_timestamp()
    return values


def forecast_topic_monthly_series(df, topic):
    """Use direct keyword results and archived posts that mention sparse topics."""
    topic_df = df.loc[df["keyword_used"].astype(str) == topic].copy()
    mention_pattern = FORECAST_MENTION_PATTERNS.get(topic)
    if mention_pattern and "_forecast_source" in df.columns and "text" in df.columns:
        archived_mentions = df.loc[
            df["_forecast_source"].eq("archive")
            & df["text"].fillna("").astype(str).str.contains(mention_pattern, case=False, regex=True)
        ].copy()
        archived_mentions["keyword_used"] = topic
        topic_df = pd.concat([topic_df, archived_mentions], ignore_index=True)

    if "tweet_id" in topic_df.columns:
        topic_df = topic_df.drop_duplicates("tweet_id", keep="first")
    return topic_monthly_series(topic_df, topic)


def series_diagnostics(series):
    """Choose one difference when needed and calculate ACF/PACF for that series."""
    clean = series.astype(float).dropna()
    p_value = 1.0
    if len(clean) >= 8 and clean.nunique() > 1:
        try:
            p_value = float(adfuller(clean, autolag="AIC")[1])
        except (ValueError, np.linalg.LinAlgError):
            p_value = 1.0

    difference_order = int(p_value >= 0.05)
    analyzed = clean.diff().dropna() if difference_order else clean
    if analyzed.nunique() < 2 or len(analyzed) < 4:
        return {"p_value": p_value, "difference_order": difference_order, "acf": [], "pacf": [], "confidence": 0.0}

    max_lags = min(40, len(analyzed) // 2 - 1)
    if max_lags < 1:
        return {"p_value": p_value, "difference_order": difference_order, "acf": [], "pacf": [], "confidence": 0.0}

    return {
        "p_value": p_value,
        "difference_order": difference_order,
        "acf": acf(analyzed, nlags=max_lags, fft=True).tolist(),
        "pacf": pacf(analyzed, nlags=max_lags, method="ywm").tolist(),
        "confidence": 1.96 / math.sqrt(len(analyzed)),
    }


def correlation_figure(values, confidence, title):
    """Plot a correlation sequence with approximate 95% confidence bounds."""
    figure = go.Figure(go.Bar(x=list(range(len(values))), y=values, marker_color="#398c82"))
    figure.add_hline(y=confidence, line_dash="dash", line_color="#ff806c")
    figure.add_hline(y=-confidence, line_dash="dash", line_color="#ff806c")
    figure.update_layout(title=title, xaxis_title="Lag (months)", yaxis_title="Correlation", showlegend=False)
    return figure


def compare_forecast_models(series, horizon, difference_order, ar_order, ma_order):
    """Backtest candidate models chronologically, then forecast from all history."""
    if len(series) < 24:
        return [], max(3, math.ceil(len(series) * 0.2))

    holdout_size = max(3, math.ceil(len(series) * 0.2))
    train, actual = series.iloc[:-holdout_size], series.iloc[-holdout_size:]
    candidates = {
        f"AR({ar_order})": ((ar_order, 0, 0), (0, 0, 0, 0), True),
        f"MA({ma_order})": ((0, 0, ma_order), (0, 0, 0, 0), True),
        f"ARMA({ar_order},{ma_order})": ((ar_order, 0, ma_order), (0, 0, 0, 0), True),
        f"ARIMA({ar_order},d,{ma_order})": ((ar_order, difference_order, ma_order), (0, 0, 0, 0), False),
        f"SARIMA({ar_order},d,{ma_order})(1,0,1,12)": ((ar_order, difference_order, ma_order), (1, 0, 1, 12), False),
    }
    results = []

    for name, (order, seasonal_order, manually_difference) in candidates.items():
        try:
            train_values = train.to_numpy()
            all_values = series.to_numpy()
            if manually_difference and difference_order:
                train_values = np.diff(train_values)
                all_values = np.diff(all_values)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                backtest = ARIMA(
                    train_values,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit()
                predicted_values = np.asarray(backtest.forecast(steps=holdout_size), dtype=float)
                fitted = ARIMA(
                    all_values,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit()
                future_values = np.asarray(fitted.forecast(steps=horizon), dtype=float)

            if manually_difference and difference_order:
                predicted_values = series.iloc[len(train) - 1] + np.cumsum(predicted_values)
                future_values = series.iloc[-1] + np.cumsum(future_values)
            predicted = pd.Series(predicted_values, index=actual.index).clip(lower=0)
            future_index = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
            future = pd.Series(future_values, index=future_index).clip(lower=0)
            errors = actual.to_numpy() - predicted.to_numpy()
            parameter_count = len(fitted.params)
            aicc = float(fitted.aic + (2 * parameter_count * (parameter_count + 1)) / (len(all_values) - parameter_count - 1))
            residuals = pd.Series(fitted.resid).dropna()
            diagnostic_lag = min(10, max(1, len(residuals) // 5))
            ljung_box_p_value = float(acorr_ljungbox(residuals, lags=[diagnostic_lag], return_df=True)["lb_pvalue"].iloc[0])
            results.append({
                "model": name,
                "order": order,
                "seasonal_order": seasonal_order,
                "holdout": predicted,
                "future": future,
                "residuals": residuals,
                "mae": float(abs(errors).mean()),
                "rmse": float((errors ** 2).mean() ** 0.5),
                "aicc": aicc,
                "ljung_box_p_value": ljung_box_p_value,
            })
        except (ValueError, np.linalg.LinAlgError, ZeroDivisionError, IndexError) as error:
            results.append({"model": name, "error": str(error)})

    return results, holdout_size


def forecast_page(df, history):
    """Compare two archived topic series using the applied workflow from the slides."""
    st.title("Applied Time-Series Comparison")
    st.caption("Two real topic samples · ADF and differencing · ACF/PACF · 12-month seasonality · out-of-sample model comparison · residual checks")
    frames = []
    for source in (history, df):
        if not source.empty and {"created_at", "keyword_used"}.issubset(source.columns):
            columns = [column for column in ("created_at", "keyword_used", "tweet_id", "text") if column in source.columns]
            frame = source[columns].copy()
            frame["_forecast_source"] = "archive" if source is history else "processed"
            frames.append(frame)
    if not frames:
        st.info("Run the historical backfill command to build a local topic archive.")
        return

    series_data = pd.concat(frames, ignore_index=True)
    series_data["created_at"] = pd.to_datetime(series_data["created_at"], format="mixed", errors="coerce", utc=True)
    series_data = series_data.dropna(subset=["created_at", "keyword_used"])
    if {"tweet_id", "keyword_used"}.issubset(series_data.columns):
        series_data = series_data.drop_duplicates(["tweet_id", "keyword_used"], keep="last")
    all_topics = sorted(
        set(series_data["keyword_used"].dropna().astype(str).unique())
        | set(FORECAST_MENTION_PATTERNS)
    )
    monthly_series = {
        topic: forecast_topic_monthly_series(series_data, topic)
        for topic in all_topics
    }
    topics = [topic for topic in all_topics if not monthly_series[topic].empty]
    if len(topics) < 2:
        st.info(
            "At least two topics need saved posts before a forecast can be shown."
        )
        return
    st.caption(
        "ChatGPT, OpenAI, and iPhone include archived posts whose text mentions each topic, "
        "alongside directly searched posts. Sparse samples are shown with a simple baseline."
    )

    topic_a_default = topics.index("AI") if "AI" in topics else 0
    topic_b_default = topics.index("IPL") if "IPL" in topics else (1 if topic_a_default == 0 else 0)
    topic_controls = st.columns(2)
    topic_a = topic_controls[0].selectbox("Dataset A", topics, index=topic_a_default)
    topic_b = topic_controls[1].selectbox("Dataset B", topics, index=topic_b_default)
    model_controls = st.columns([1.5, 1, 1])
    horizon = model_controls[0].selectbox(
        "Forecast",
        [3, 6, 12],
        index=0,
        format_func=lambda months: f"{months} month" if months == 1 else f"{months} months",
    )
    ar_order = model_controls[1].selectbox("AR order p (PACF)", [1, 2, 3], index=0)
    ma_order = model_controls[2].selectbox("MA order q (ACF)", [1, 2, 3], index=0)
    if topic_a == topic_b:
        st.warning("Choose two different topics to compare their histories.")
        return

    current_month = pd.Timestamp.now(tz="UTC").tz_localize(None).to_period("M")
    partial_current_month_topics = set()
    series_by_topic = {}
    for topic in (topic_a, topic_b):
        series = monthly_series[topic]
        if not series.empty and series.index[-1].to_period("M") == current_month:
            series = series.loc[series.index.to_period("M") < current_month]
            partial_current_month_topics.add(topic)
        if series.empty:
            st.info(f"{topic} has no completed monthly observations to forecast yet.")
            return
        series_by_topic[topic] = series

    start_date = min(series.index.min() for series in series_by_topic.values() if not series.empty)
    end_date = max(series.index.max() for series in series_by_topic.values() if not series.empty)
    st.markdown(f"**{start_date:%Y-%m} to {end_date:%Y-%m}** · {len(series_data):,} archived and current posts · monthly counts")
    if partial_current_month_topics:
        st.caption("The in-progress month is excluded from model fitting so a partial-month count does not skew the forecasts.")

    comparison = go.Figure()
    palette = {topic_a: "#ff806c", topic_b: "#398c82"}
    topic_maxima = {topic: float(series.max()) for topic, series in series_by_topic.items()}
    comparison_axis_topic = None
    lower_topic = min(topic_maxima, key=topic_maxima.get)
    higher_topic = max(topic_maxima, key=topic_maxima.get)
    if (
        lower_topic in FORECAST_MENTION_PATTERNS
        and higher_topic not in FORECAST_MENTION_PATTERNS
        and topic_maxima[lower_topic] > 0
        and topic_maxima[higher_topic] >= topic_maxima[lower_topic] * 3
    ):
        comparison_axis_topic = lower_topic

    for topic, series in series_by_topic.items():
        trace = go.Scatter(x=series.index, y=series.values, name=topic, line={"color": palette[topic], "width": 2})
        if topic == comparison_axis_topic:
            trace.yaxis = "y2"
        comparison.add_trace(trace)
    comparison_layout = {
        "title": "Observed monthly topic activity",
        "xaxis_title": "Month",
        "yaxis_title": (
            f"Posts in the local sample · {higher_topic}"
            if comparison_axis_topic
            else "Posts in the local sample"
        ),
        "hovermode": "x unified",
    }
    if comparison_axis_topic:
        comparison_layout["yaxis2"] = {
            "title": f"Posts in the local sample · {comparison_axis_topic}",
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
            "rangemode": "tozero",
        }
    comparison.update_layout(**comparison_layout)
    show_plotly(comparison)

    st.subheader("Stationarity and correlation structure")
    diagnostics_by_topic = {}
    for topic, series in series_by_topic.items():
        diagnostics = series_diagnostics(series)
        diagnostics_by_topic[topic] = diagnostics
        with st.expander(f"{topic} · {len(series):,} monthly observations · ADF p-value {diagnostics['p_value']:.4f}", expanded=True):
            if diagnostics["difference_order"]:
                st.caption("ADF does not reject a unit root at 5%; first differences are shown and used for the AR, MA, and ARMA baselines.")
            else:
                st.caption("ADF rejects a unit root at 5%; the observed series is treated as stationary.")
            if not diagnostics["acf"]:
                st.info("This topic needs more varied observations for ACF/PACF analysis.")
                continue
            correlation_columns = st.columns(2)
            correlation_columns[0].plotly_chart(
                correlation_figure(diagnostics["acf"], diagnostics["confidence"], "ACF"),
                use_container_width=True,
            )
            correlation_columns[1].plotly_chart(
                correlation_figure(diagnostics["pacf"], diagnostics["confidence"], "PACF"),
                use_container_width=True,
            )

    model_results = {}
    score_rows = []
    failures = []
    for topic, series in series_by_topic.items():
        if len(series) < 24:
            continue
        forecast_steps = horizon + int(topic in partial_current_month_topics)
        results, holdout_size = compare_forecast_models(
            series,
            forecast_steps,
            diagnostics_by_topic[topic]["difference_order"],
            ar_order,
            ma_order,
        )
        if topic in partial_current_month_topics:
            for result in results:
                if "future" in result:
                    result["future"] = result["future"].iloc[1:]
        model_results[topic] = results
        for result in results:
            if "error" in result:
                failures.append(f"{topic} {result['model']}")
                continue
            score_rows.append({
                "Dataset": topic,
                "Model": result["model"],
                "Holdout MAE": result["mae"],
                "Holdout RMSE": result["rmse"],
                "AICc": result["aicc"],
                "Ljung-Box p-value": result["ljung_box_p_value"],
            })

    if failures:
        st.warning("Some fits failed for this sample: " + ", ".join(failures))
    st.subheader("Model comparison")
    if score_rows:
        st.caption(f"PACF-informed AR order p={ar_order}; ACF-informed MA order q={ma_order}. Each model is evaluated on the last 20% of its monthly series. Lower MAE/RMSE is better; Ljung-Box p-values above 0.05 are consistent with uncorrelated residuals.")
        score_table = pd.DataFrame(score_rows).sort_values(["Dataset", "Holdout MAE"])
        st.dataframe(score_table.round(3), hide_index=True, use_container_width=True)
    else:
        st.info("No topic has enough monthly history for model comparison. Limited-history baseline forecasts are shown below.")

    st.subheader("Forecasts and residual checks")
    forecast_columns = st.columns(2)
    for column, topic in zip(forecast_columns, (topic_a, topic_b)):
        with column:
            results = model_results.get(topic, [])
            successful = [result for result in results if "error" not in result]
            series = series_by_topic[topic]
            figure = go.Figure()
            figure.add_trace(go.Scatter(x=series.index, y=series.values, name="Observed", line={"color": "#262632", "width": 3}))
            if not successful:
                future_index = pd.date_range(
                    series.index[-1] + pd.offsets.MonthBegin(1),
                    periods=horizon,
                    freq="MS",
                )
                baseline = pd.Series(float(series.mean()), index=future_index)
                figure.add_trace(go.Scatter(
                    x=baseline.index,
                    y=baseline.values,
                    name="Limited-history baseline",
                    line={"color": "#ff806c", "dash": "dash"},
                ))
                st.caption(
                    f"{topic}: only {len(series)} monthly observation(s); "
                    "the dashed line extends the observed monthly average and is not a fitted ARIMA forecast."
                )
                figure.update_layout(title=f"{topic}: {horizon}-month baseline", xaxis_title="Month", yaxis_title="Posts in the local sample", hovermode="x unified")
                show_plotly(figure)
                continue
            for index, result in enumerate(successful):
                figure.add_trace(go.Scatter(
                    x=result["future"].index,
                    y=result["future"].values,
                    name=result["model"],
                    line={"color": ["#ff806c", "#83d990", "#398c82", "#5b6474", "#e1ac42"][index], "dash": "dash"},
                ))
            figure.update_layout(title=f"{topic}: {horizon}-month model forecasts", xaxis_title="Month", yaxis_title="Posts in the local sample", hovermode="x unified")
            show_plotly(figure)
            best = min(successful, key=lambda result: result["mae"])
            residual_acf = acf(best["residuals"], nlags=min(30, max(1, len(best["residuals"]) // 2 - 1)), fft=True).tolist()
            show_plotly(correlation_figure(residual_acf, 1.96 / math.sqrt(len(best["residuals"])), f"{topic}: residual ACF ({best['model']})"))
            st.caption(f"Lowest holdout MAE: {best['model']} ({best['mae']:.2f}). Forecasts represent this archived sample, not all Bluesky activity.")


def overview_page(df):
    """Page 1: Overview"""
    st.title("📊 Overview")
    
    if df.empty:
        st.warning("No data available. Please run the data collection and processing pipeline first.")
        return
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_posts = len(df)
    positive_pct = (df['sentiment'] == 'Positive').sum() / total_posts * 100
    negative_pct = (df['sentiment'] == 'Negative').sum() / total_posts * 100
    neutral_pct = (df['sentiment'] == 'Neutral').sum() / total_posts * 100
    total_keywords = df['keyword_used'].nunique()
    
    col1.metric("Total Posts", f"{total_posts:,}")
    col2.metric("Positive %", f"{positive_pct:.1f}%")
    col3.metric("Negative %", f"{negative_pct:.1f}%")
    col4.metric("Neutral %", f"{neutral_pct:.1f}%")
    st.caption(f"Monitoring {total_keywords} search topics")
    
    # Recent activity
    st.subheader("Recent Activity")
    recent_df = df.sort_values('created_at', ascending=False).head(10)
    
    for _, row in recent_df.iterrows():
        with st.expander(f"@{row['username']} - {row['created_at'].strftime('%Y-%m-%d %H:%M')}"):
            st.write(f"**Text:** {row['text']}")
            st.write(f"**Sentiment:** {row['sentiment']}")
            st.write(f"**Engagement:** {row['likes']} likes, {row['retweets']} reposts")


def sentiment_page(df):
    """Page 2: Sentiment Analysis"""
    st.title("😊 Sentiment Analysis")
    
    if df.empty:
        st.warning("No data available.")
        return
    
    # Sentiment distribution
    sentiment_counts = df['sentiment'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sentiment Distribution (Pie)")
        fig_pie = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#95a5a6'}
        )
        show_plotly(fig_pie)
    
    with col2:
        st.subheader("Sentiment Counts (Bar)")
        fig_bar = px.bar(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            color=sentiment_counts.index,
            color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#95a5a6'}
        )
        fig_bar.update_layout(xaxis_title="Sentiment", yaxis_title="Count")
        show_plotly(fig_bar)
    
    # Sentiment by keyword
    st.subheader("Sentiment by Keyword")
    keyword_sentiment = df.groupby(['keyword_used', 'sentiment']).size().unstack(fill_value=0)
    st.bar_chart(keyword_sentiment)


def trends_page(df, history):
    """Show post volume and sentiment trends across current and archived posts."""
    st.title("📈 Trends")

    posts = combine_posts(df, history)
    if posts.empty:
        st.warning("No current or archived posts are available.")
        return

    latest_date = posts["created_at"].max().normalize()
    range_options = {
        "Last 7 days": 7,
        "Last 30 days": 30,
        "Last 90 days": 90,
        "Last year": 365,
        "All stored history": None,
    }
    controls = st.columns([1, 1, 1])
    selected_range = controls[0].selectbox("Date range", list(range_options), index=1)
    topics = sorted(posts["keyword_used"].dropna().astype(str).unique())
    selected_topic = controls[1].selectbox("Topic", ["All topics", *topics])
    interval = controls[2].selectbox("Group by", ["Day", "Week", "Month"], index=1)

    filtered = posts.copy()
    days = range_options[selected_range]
    if days is not None:
        filtered = filtered.loc[filtered["created_at"] >= latest_date - pd.Timedelta(days=days - 1)]
    if selected_topic != "All topics":
        filtered = filtered.loc[filtered["keyword_used"].astype(str) == selected_topic]
    if filtered.empty:
        st.info("No archived posts match this topic and date range.")
        return

    bucket_frequency = {"Day": "D", "Week": "W-SUN", "Month": "MS"}[interval]
    bucket_label = {"Day": "Date", "Week": "Week", "Month": "Month"}[interval]
    date_indexed = filtered.set_index("created_at").sort_index()
    volume = date_indexed.resample(bucket_frequency, label="left", closed="left").size().rename("Posts")

    metrics = st.columns(3)
    metrics[0].metric("Posts in view", f"{len(filtered):,}")
    metrics[1].metric("Topics in view", f"{filtered['keyword_used'].nunique():,}")
    metrics[2].metric("Latest saved post", latest_date.strftime("%Y-%m-%d"))
    st.caption(
        f"Showing {filtered['created_at'].min():%Y-%m-%d} to "
        f"{filtered['created_at'].max():%Y-%m-%d}; archived posts are included."
    )

    st.subheader("Post volume")
    volume_for_chart = volume.where(volume.between(0, 600))
    volume_figure = px.line(
        volume_for_chart.reset_index(),
        x="created_at",
        y="Posts",
        markers=True,
        title=f"Posts per {interval.lower()}",
    )
    volume_figure.update_layout(xaxis_title=bucket_label, yaxis_title="Posts", hovermode="x unified")
    volume_figure.update_yaxes(range=[0, 600])
    show_plotly(volume_figure)

    st.subheader("Sentiment mix")
    sentiment_order = ["Positive", "Neutral", "Negative"]
    sentiment_dates = filtered["created_at"].dt.tz_convert(None)
    sentiment_counts = (
        filtered.assign(period=sentiment_dates.dt.to_period(
            {"Day": "D", "Week": "W-SUN", "Month": "M"}[interval]
        ).dt.start_time)
        .groupby(["period", "sentiment"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=sentiment_order, fill_value=0)
    )
    sentiment_share = sentiment_counts.div(
        sentiment_counts.sum(axis=1).replace(0, np.nan),
        axis=0,
    ).mul(100).fillna(0)
    sentiment_figure = go.Figure()
    colors = {"Positive": "#2ecc71", "Neutral": "#95a5a6", "Negative": "#e74c3c"}
    for sentiment in sentiment_order:
        sentiment_figure.add_trace(go.Scatter(
            x=sentiment_share.index,
            y=sentiment_share[sentiment],
            name=sentiment,
            mode="lines",
            stackgroup="sentiment",
            line={"color": colors[sentiment]},
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}%<extra>" + sentiment + "</extra>",
        ))
    sentiment_figure.update_layout(
        title=f"Sentiment share per {interval.lower()}",
        xaxis_title=bucket_label,
        yaxis_title="Share of posts",
        yaxis={"range": [0, 100], "ticksuffix": "%"},
        hovermode="x unified",
    )
    show_plotly(sentiment_figure)


def hashtags_page(df):
    """Page 4: Hashtags"""
    st.title("#️⃣ Hashtags")
    
    if df.empty:
        st.warning("No data available.")
        return
    
    # Extract all hashtags
    all_hashtags = []
    for hashtags in df['hashtags']:
        if isinstance(hashtags, list):
            all_hashtags.extend(hashtags)
        elif isinstance(hashtags, str):
            all_hashtags.append(hashtags)
    
    if not all_hashtags:
        st.info("No hashtags found in the data.")
        return
    
    # Count hashtags
    from collections import Counter
    hashtag_counts = Counter(all_hashtags)
    top_hashtags = hashtag_counts.most_common(20)
    
    # Display
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 20 Hashtags")
        hashtag_df = pd.DataFrame(top_hashtags, columns=['Hashtag', 'Count'])
        st.dataframe(hashtag_df, use_container_width=True)
    
    with col2:
        st.subheader("Hashtag Frequency")
        fig_hashtags = px.bar(
            hashtag_df.head(10),
            x='Hashtag',
            y='Count',
            title='Top 10 Hashtags'
        )
        show_plotly(fig_hashtags)


def recent_posts_page(df, history):
    """Page 5: Recent Bluesky posts"""
    st.title("Recent Posts")

    posts = combine_posts(df, history)
    if posts.empty:
        st.warning("No current or archived posts are available.")
        return
    
    # Search
    search_term = st.text_input("Search posts", "")
    
    # Filter
    filtered_df = posts.copy()
    if search_term:
        filtered_df = filtered_df[
            filtered_df['text'].str.contains(search_term, case=False, na=False) |
            filtered_df['username'].str.contains(search_term, case=False, na=False)
        ]
    
    # Sort by date
    filtered_df = filtered_df.sort_values('created_at', ascending=False)
    
    # Display
    st.subheader(f"Showing {len(filtered_df)} posts")
    
    for _, row in filtered_df.head(50).iterrows():
        sentiment_color = {
            'Positive': '🟢',
            'Negative': '🔴',
            'Neutral': '⚪'
        }.get(row['sentiment'], '⚪')
        
        with st.container():
            st.markdown(f"**{sentiment_color} @{row['username']}** - {row['created_at'].strftime('%Y-%m-%d %H:%M')}")
            st.markdown(f"*{row['text']}*")
            st.caption(f"Sentiment: {row['sentiment']} | Likes: {row['likes']} | Reposts: {row['retweets']} | Keyword: {row['keyword_used']}")
            st.divider()


def main():
    """Main app"""
    # Load data
    df = load_data()
    history = load_history()
    
    # Sidebar navigation
    st.sidebar.title("Bluesky Analytics")
    st.sidebar.caption("PUBLIC POST MONITOR")
    page = st.sidebar.radio(
        "Select Page",
        ["Overview", "Sentiment", "Trends", "Hashtags", "Recent Posts", "Topic Forecasts"]
    )
    if st.sidebar.button("Reload and fetch latest posts", use_container_width=True):
        try:
            collector = BlueskyCollector()
            latest_posts, failed_topics = collector.collect_latest_posts()
            if latest_posts:
                collector.save_latest_posts(latest_posts)
                if failed_topics:
                    st.session_state.refresh_feedback = (
                        "warning",
                        f"Saved {len(latest_posts):,} posts; live search failed for "
                        f"{', '.join(failed_topics)}. Those topics still use saved data.",
                    )
                else:
                    st.session_state.refresh_feedback = (
                        "success",
                        f"Live search checked all topics and saved {len(latest_posts):,} posts.",
                    )
            elif failed_topics:
                st.session_state.refresh_feedback = (
                    "warning",
                    f"Live search failed for {', '.join(failed_topics)}. Showing the latest posts saved locally.",
                )
            else:
                st.session_state.refresh_feedback = (
                    "info",
                    "Live search checked all topics, but returned no posts. Showing the newest posts saved locally.",
                )
        except requests.RequestException:
            st.session_state.refresh_feedback = (
                "warning",
                "The live API is unavailable or blocked. Showing the newest posts in the local archive.",
            )
        except OSError as error:
            st.session_state.refresh_feedback = (
                "error",
                f"Could not save refreshed posts locally: {error}",
            )
        st.cache_data.clear()
        df = load_data()
        history = load_history()

    refresh_feedback = st.session_state.get("refresh_feedback")
    if refresh_feedback:
        level, message = refresh_feedback
        getattr(st.sidebar, level)(message)
    st.sidebar.caption(f"{len(df):,} processed · {len(history):,} archived")
    
    # Render selected page
    if page == "Overview":
        overview_page(df)
    elif page == "Sentiment":
        sentiment_page(df)
    elif page == "Trends":
        trends_page(df, history)
    elif page == "Hashtags":
        hashtags_page(df)
    elif page == "Recent Posts":
        recent_posts_page(df, history)
    elif page == "Topic Forecasts":
        forecast_page(df, history)


if __name__ == "__main__":
    main()
