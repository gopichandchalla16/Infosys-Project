# ==============================================================================
# PROJECT: Real-Time Market Intelligence System
# FILE: app.py
# DESCRIPTION: Main Streamlit dashboard for strategic market analysis,
#              AI-driven sentiment, forecasting, and Slack alerting.
# ==============================================================================

import sys
import os
from datetime import datetime, timedelta

# -----------------------------
# DIRECTORY SETUP
# -----------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# -----------------------------
# IMPORT LIBRARIES
# -----------------------------
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# Core Strategy Logic
from core.strategy import compute_competitive_index, classify_strategic_signal
from core.llm_strategy import generate_strategic_explanation
from core.market_data import fetch_market_data
from core.news_fetcher import fetch_news
from core.sentiment import analyze_sentiment
from core.forecast import run_prophet
from core.alerts import build_alert, send_slack
from core.utils import get_ticker
from core.utils import ALLOWED_COMPANIES

# ------------------------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Real-Time Market Intelligence",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------------
# GLOBAL PREMIUM UI (GLASS + DARK THEME)
# ------------------------------------------------------------------------------
st.markdown(
    """
<style>
:root{
  --bg0:#05070c;
  --bg1:#090d18;
  --card: rgba(255,255,255,0.06);
  --card2: rgba(255,255,255,0.08);
  --stroke: rgba(255,255,255,0.10);
  --txt:#e8e8e8;
  --muted:#a7b0c0;
  --aqua:#00e5ff;
  --amber:#ffb347;
  --red:#ff4d6d;
  --green:#22c55e;
}

.stApp{
  background: radial-gradient(1200px 600px at 10% 0%, #0b1c2a 0%, var(--bg0) 55%) , linear-gradient(180deg, var(--bg1), var(--bg0));
  color: var(--txt);
}

.block-container{ padding-top: 1.4rem; padding-bottom: 3rem; }

h1, h2, h3, h4 { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; }

.header-wrap{
  background: linear-gradient(135deg, rgba(0,229,255,0.10), rgba(255,179,71,0.07));
  border: 1px solid var(--stroke);
  border-radius: 18px;
  padding: 18px 18px 14px 18px;
  backdrop-filter: blur(10px);
  box-shadow: 0 14px 50px rgba(0,0,0,0.35);
  margin-bottom: 14px;
  margin-top:44px;
}

.title{
  font-size: 2.0rem;
  font-weight: 800;
  letter-spacing: -0.4px;
  margin: 0;
  background: linear-gradient(90deg, var(--aqua), var(--amber));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.subtitle{
  margin-top: 6px;
  color: var(--muted);
  font-size: 0.98rem;
}

.glass{
  background: var(--card);
  border: 1px solid var(--stroke);
  border-radius: 16px;
  padding: 14px 14px 12px 14px;
  backdrop-filter: blur(10px);
  box-shadow: 0 10px 34px rgba(0,0,0,0.35);
}

.glass-hover{
  transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
}
.glass-hover:hover{
  transform: translateY(-3px);
  border-color: rgba(0,229,255,0.30);
  box-shadow: 0 14px 44px rgba(0,229,255,0.10), 0 18px 60px rgba(0,0,0,0.45);
}

.kpi-label{ color: var(--muted); font-size: 0.85rem; margin-bottom: 6px; }
.kpi-value{ font-size: 1.55rem; font-weight: 800; letter-spacing: -0.2px; }
.kpi-sub{ color: var(--muted); font-size: 0.82rem; margin-top: 4px; }

.badge{
  display:inline-block;
  padding: 2px 10px;
  border-radius: 999px;
  font-size: 0.78rem;
  border: 1px solid var(--stroke);
  background: rgba(255,255,255,0.05);
  color: var(--txt);
}

.badge-pos{ border-color: rgba(34,197,94,0.35); background: rgba(34,197,94,0.12); }
.badge-neg{ border-color: rgba(255,77,109,0.35); background: rgba(255,77,109,0.12); }
.badge-neu{ border-color: rgba(167,176,192,0.35); background: rgba(167,176,192,0.10); }

.news-card{
  padding: 14px;
  margin-bottom: 10px;
}
.news-title{
  font-size: 1.03rem;
  font-weight: 750;
  text-decoration: none;
  color: var(--aqua);
}
.news-meta{ color: var(--muted); font-size: 0.82rem; margin-top: 3px; }
.news-desc{ color: #d7dbe6; font-size: 0.92rem; margin-top: 8px; line-height: 1.35rem; }

hr{
  border: none;
  height: 1px;
  background: rgba(255,255,255,0.07);
  margin: 12px 0;
}

[data-testid="stSidebar"]{
  background: rgba(255,255,255,0.03);
  border-right: 1px solid rgba(255,255,255,0.06);
}

.stButton>button{
  background: linear-gradient(90deg, var(--aqua), var(--amber));
  border: none;
  color: #0a0a0a;
  font-weight: 800;
  border-radius: 999px;
  padding: 0.55rem 1rem;
  transition: transform 0.16s ease;
  width: 100%;
}
.stButton>button:hover{ transform: scale(1.02); }

.small-note{ color: var(--muted); font-size: 0.85rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------
def _safe_pct(a: float, b: float) -> float:
    """Safely compute percentage change."""
    if b == 0 or pd.isna(b) or pd.isna(a):
        return 0.0
    return float((a - b) / b * 100.0)

def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index."""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.bfill().fillna(50)

def _sentiment_badge(label: str) -> str:
    """Return HTML badge for sentiment labels."""
    lbl = (label or "").lower()
    if lbl == "positive":
        return '<span class="badge badge-pos">Positive</span>'
    if lbl == "negative":
        return '<span class="badge badge-neg">Negative</span>'
    return '<span class="badge badge-neu">Neutral</span>'

def build_slack_message(company, ticker, last_price, forecast_df, sentiment_score, strategic_signal, llm_summary):
    """
    FIXED: Constructs a professional strategic alert for Slack.
    Uses Markdown blocks for a premium appearance in Slack.
    """
    # Forecast direction calculation
    f_status = "No data"
    if forecast_df is not None and not forecast_df.empty:
        start_f = forecast_df['yhat'].iloc[0]
        end_f = forecast_df['yhat'].iloc[-1]
        pct = ((end_f - start_f)/start_f)*100
        f_status = f"{'📈 Bullish' if pct > 0 else '📉 Bearish'} ({pct:+.2f}%)"

    # Strategic Signal Formatting
    sig_val = strategic_signal.get('signal', 'NEUTRAL')
    emoji = "🚀" if sig_val == "OPPORTUNITY" else "⚠️" if sig_val == "THREAT" else "⚖️"

    # Construct plain text for Slack
    msg = (
        f"{emoji} *STRATEGIC INTELLIGENCE ALERT: {company} ({ticker})*\n"
        f"──────────────────────────────────\n"
        f"💰 *Market Price:* ${last_price:,.2f}\n"
        f"🎭 *Net Sentiment:* {sentiment_score:+.1f}%\n"
        f"🔮 *7-Day Forecast:* {f_status}\n"
        f"🎯 *Strategic Signal:* `{sig_val}`\n"
        f"──────────────────────────────────\n"
        f"*AI Strategic Summary:*\n"
        f"_{llm_summary[:400]}..._\n"
        f"──────────────────────────────────\n"
        f"📍 _Generated by Market Intel AI Engine_"
    )
    return msg

# -----------------------------
# CACHED DATA LOADERS
# -----------------------------
@st.cache_data(show_spinner=False, ttl=300)
def load_market_data(company: str) -> pd.DataFrame:
    return fetch_market_data(company)

@st.cache_data(show_spinner=False, ttl=300)
def load_news(company: str):
    return fetch_news(company)

# ------------------------------------------------------------------------------
# SIDEBAR CONTROLS
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Analysis Controls")
    company = st.selectbox(
        "Select Target Company",
        sorted(ALLOWED_COMPANIES),
    )

    ticker = get_ticker(company)

    range_label = st.selectbox(
        "Historical Time Range",
        ["1M", "3M", "6M", "1Y", "MAX"],
        index=1,
        help="Select window for trend analysis and KPI computation.",
    )

    st.markdown("---")
    st.markdown("### 📊 Visualization Prefs")
    show_candles = st.toggle("Candlestick View", value=True)
    show_volume = st.toggle("Show Volume Bar", value=True)
    show_indicators = st.toggle("Show Indicators (MA + RSI)", value=True)

    st.markdown("---")
    st.markdown("### 📰 Content & Live Data")
    news_limit = st.slider("Max News Articles", 5, 30, 12)
    auto_refresh = st.toggle("Enable Auto-Refresh", value=False)
    refresh_secs = st.slider("Interval (sec)", 15, 120, 30, disabled=not auto_refresh)

    st.markdown("---")
    slack_enabled = st.toggle("Slack Integration", value=True)
    st.markdown(
        "<div class='small-note'>Press <b>'R'</b> to manually force clear cache.</div>",
        unsafe_allow_html=True,
    )

# Auto refresh handler
if auto_refresh:
    st_autorefresh(interval=refresh_secs * 1000, key="market_auto_refresh")

# ------------------------------------------------------------------------------
# DASHBOARD HEADER
# ------------------------------------------------------------------------------
st.markdown(
    f"""
<div class="header-wrap">
  <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:12px; flex-wrap:wrap;">
    <div>
      <div class="title">Market Intelligence & Strategy Engine</div>
      <div class="subtitle">
        Current Asset: <b>{company}</b> ({ticker}) · Competitive Intel · AI Sentiment · Prophet Forecast
      </div>
    </div>
    <div style="text-align:right;">
      <div class="badge">Enterprise Version</div>
      <div class="news-meta">{datetime.now().strftime("%d %b %Y · %I:%M %p")}</div>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# DATA ACQUISITION & PROCESSING
# ------------------------------------------------------------------------------
with st.spinner(f"Acquiring {company} market dynamics..."):
    market_df = load_market_data(company)

if market_df is None or market_df.empty:
    st.error("Market data unavailable. Please verify API availability or ticker mappings.")
    st.stop()

# Datetime normalization
if not isinstance(market_df.index, pd.DatetimeIndex):
    if "Date" in market_df.columns:
        market_df["Date"] = pd.to_datetime(market_df["Date"])
        market_df = market_df.set_index("Date")
    else:
        market_df.index = pd.to_datetime(market_df.index)

market_df = market_df.sort_index()

# Window Filtering
end_dt = market_df.index.max()
if range_label == "1M":
    start_dt = end_dt - pd.Timedelta(days=31)
elif range_label == "3M":
    start_dt = end_dt - pd.Timedelta(days=93)
elif range_label == "6M":
    start_dt = end_dt - pd.Timedelta(days=186)
elif range_label == "1Y":
    start_dt = end_dt - pd.Timedelta(days=366)
else:
    start_dt = market_df.index.min()

df = market_df.loc[market_df.index >= start_dt].copy()
if df.empty:
    df = market_df.tail(120).copy()

# Feature Engineering
df["Return"] = df["Close"].pct_change() * 100
df["MA7"] = df["Close"].rolling(7).mean()
df["MA21"] = df["Close"].rolling(21).mean()
df["RSI14"] = _compute_rsi(df["Close"], 14)

# News Feed & FinBERT Sentiment Analysis
with st.spinner("Processing NLP news sentiment..."):
    news = load_news(company) or []
    sentiment_details, sentiment_counts = analyze_sentiment(news)

# Time Series Forecasting
with st.spinner("Calculating Prophet forecast..."):
    forecast_df = run_prophet(market_df) if len(market_df) >= 60 else None

# ------------------------------------------------------------------------------
# CORE STRATEGY CALCULATION
# ------------------------------------------------------------------------------
competitive_index = compute_competitive_index(
    market_df=market_df,
    sentiment_counts=sentiment_counts or {},
    forecast_df=forecast_df,
)

strategic_signal = classify_strategic_signal(
    market_df=market_df,
    sentiment_counts=sentiment_counts or {},
    forecast_df=forecast_df,
)

alert_payload = build_alert(
    company,
    ticker,
    sentiment_counts or {},
    strategic={
        "competitive_index": competitive_index,
        "strategic_signal": strategic_signal,
    },
)

llm_explanation = generate_strategic_explanation(
    company=company,
    competitive_index=competitive_index,
    strategic_signal=strategic_signal,
    sentiment_counts=sentiment_counts or {},
)

# ------------------------------------------------------------------------------
# KPI SECTION
# ------------------------------------------------------------------------------
last_close = float(df["Close"].iloc[-1])
prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else last_close
chg_1d = _safe_pct(last_close, prev_close)

first_close = float(df["Close"].iloc[0])
chg_range = _safe_pct(last_close, first_close)

volatility = float(df["Return"].std(skipna=True)) if df["Return"].notna().any() else 0.0
avg_volume = float(df["Volume"].tail(20).mean()) if "Volume" in df.columns else np.nan

pos = int((sentiment_counts or {}).get("positive", 0))
neg = int((sentiment_counts or {}).get("negative", 0))
neu = int((sentiment_counts or {}).get("neutral", 0))
total_news = max(pos + neg + neu, len(news))

sent_score = 0.0
if total_news > 0:
    sent_score = ((pos - neg) / total_news) * 100.0

k1, k2, k3, k4, k5 = st.columns(5)

def kpi_card(col, label, value, sub):
    col.markdown(
        f"""
<div class="glass glass-hover">
  <div class="kpi-label">{label}</div>
  <div class="kpi-value">{value}</div>
  <div class="kpi-sub">{sub}</div>
</div>
""",
        unsafe_allow_html=True,
    )

kpi_card(k1, "Latest Price", f"${last_close:,.2f}", f"Change: {chg_1d:+.2f}%")
kpi_card(k2, f"Range Performance", f"{chg_range:+.2f}%", f"Window: {range_label}")
kpi_card(k3, "Daily Volatility", f"{volatility:.2f}%", "Std Dev Returns")
kpi_card(k4, "Net Sentiment", f"{sent_score:+.1f}", f"P: {pos} | N: {neg} | O: {neu}")
kpi_card(
    k5,
    "Strategic Risk",
    f"{(alert_payload or {}).get('alert_type','N/A')}",
    (alert_payload or {}).get("strategic_action", "Monitoring")[:34] + "...",
)

st.markdown("<hr/>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# MAIN INTERFACE TABS
# ------------------------------------------------------------------------------
tab_strategy, tab_overview, tab_forecast, tab_sentiment, tab_news, tab_alerts = st.tabs(
    ["🧠 Strategic Intel", "📈 Market Dynamics", "🔮 Forecast", "💬 NLP Sentiment", "📰 Newsfeed", "🔔 Alert Hub"]
)

# =========================================================
# TAB 1: STRATEGIC INTELLIGENCE
# =========================================================
with tab_strategy:
    st.markdown("### 🧠 Comprehensive Strategic Positioning")
    c1, c2 = st.columns([1, 1.2], gap="large")

    with c1:
        strength = (
            "Dominant" if competitive_index >= 80 else
            "Strong" if competitive_index >= 65 else
            "Neutral" if competitive_index >= 45 else
            "Under-performing"
        )
        st.markdown(f"""
        <div class="glass glass-hover">
          <div class="kpi-label">Competitive Positioning Index</div>
          <div class="kpi-value">{competitive_index} / 100</div>
          <div class="kpi-sub">Classification: <b>{strength}</b></div>
          <br/>
          <div class="small-note">Computed based on Momentum, Sentiment Polarity, Forecast Delta, and News Volatility.</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 🤖 AI-Generated Strategic Brief")
        st.markdown(f"""
        <div class="glass" style="max-height: 400px; overflow-y: auto;">
          <div class="kpi-label">Executive Analysis</div>
          <div style="font-size: 0.95rem; line-height: 1.6; color: #e8e8e8; margin-top: 10px; white-space: pre-wrap;">
{llm_explanation}
          </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        sig_data = strategic_signal["signal"]
        sig_color = "#22c55e" if sig_data == "OPPORTUNITY" else "#ff4d6d" if sig_data == "THREAT" else "#ffb347"
        
        st.markdown(f"""
        <div class="glass glass-hover">
          <div class="kpi-label">Primary Strategic Signal</div>
          <div class="kpi-value" style="color:{sig_color}">{sig_data}</div>
          <div class="kpi-sub">Model Confidence: {strategic_signal.get('confidence', 'High')}</div>
          <hr/>
          <div class="kpi-label">Reasoning Factors:</div>
          <ul style="margin-left:18px; color: #d7dbe6; font-size: 0.9rem;">
            {''.join([f"<li style='margin-bottom:6px;'>{r}</li>" for r in strategic_signal.get("reason", ["Analyzing indicators..."])])}
          </ul>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# TAB 2: MARKET DYNAMICS
# =========================================================
with tab_overview:
    left, right = st.columns([2, 1], gap="large")

    with left:
        st.markdown("### 📊 Pricing & Indicator Analysis")
        fig = go.Figure()
        if show_candles and all(c in df.columns for c in ["Open", "High", "Low", "Close"]):
            fig.add_trace(go.Candlestick(
                x=df.index, open=df["Open"], high=df["High"],
                low=df["Low"], close=df["Close"], name="OHLC",
                increasing_line_color="#22c55e", decreasing_line_color="#ff4d6d"
            ))
        else:
            fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Closing Price", line=dict(color=vars["--aqua"], width=2)))

        if show_indicators:
            fig.add_trace(go.Scatter(x=df.index, y=df["MA7"], name="7-Day MA", line=dict(dash='dot', width=1.5)))
            fig.add_trace(go.Scatter(x=df.index, y=df["MA21"], name="21-Day MA", line=dict(dash='dash', width=1.5)))

        fig.update_layout(template="plotly_dark", height=450, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        if show_volume:
            fig_v = px.bar(df, x=df.index, y="Volume", color_discrete_sequence=["#4a5568"])
            fig_v.update_layout(template="plotly_dark", height=150, margin=dict(l=10, r=10, t=0, b=10))
            st.plotly_chart(fig_v, use_container_width=True)

    with right:
        st.markdown("### 🔍 Technical Snapshot")
        tr_val = "UPTREND" if df["MA7"].iloc[-1] > df["MA21"].iloc[-1] else "DOWNTREND"
        rsi_val = df["RSI14"].iloc[-1]
        
        st.markdown(f"""
        <div class="glass">
          <div class="kpi-label">Market Momentum</div>
          <div class="kpi-value">{tr_val}</div>
          <div class="kpi-sub">Based on MA Crossover</div>
        </div><br/>
        <div class="glass">
          <div class="kpi-label">RSI (14)</div>
          <div class="kpi-value">{rsi_val:.2f}</div>
          <div class="kpi-sub">{'Overbought' if rsi_val > 70 else 'Oversold' if rsi_val < 30 else 'Neutral Range'}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.download_button("Export Dataset", data=df.to_csv(), file_name=f"{ticker}_data.csv", mime="text/csv")

# =========================================================
# TAB 3: FORECAST
# =========================================================
with tab_forecast:
    st.markdown("### 🔮 Predictor (Meta Prophet Engine)")
    if forecast_df is None:
        st.warning("Insufficient historical depth for forecasting.")
    else:
        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(x=market_df.index, y=market_df["Close"], name="Historical", line=dict(color="#6366f1")))
        fig_f.add_trace(go.Scatter(x=forecast_df["ds"], y=forecast_df["yhat"], name="AI Projection", line=dict(color="#ffb347", width=3)))
        fig_f.add_trace(go.Scatter(x=forecast_df["ds"], y=forecast_df["yhat_upper"], fill=None, mode='lines', line_color='rgba(255,255,255,0)', showlegend=False))
        fig_f.add_trace(go.Scatter(x=forecast_df["ds"], y=forecast_df["yhat_lower"], fill='tonexty', mode='lines', line_color='rgba(255,255,255,0)', fillcolor='rgba(255,179,71,0.1)', showlegend=False))
        
        fig_f.update_layout(template="plotly_dark", height=500, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_f, use_container_width=True)
        st.dataframe(forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7), use_container_width=True)

# =========================================================
# TAB 4: NLP SENTIMENT
# =========================================================
with tab_sentiment:
    st.markdown("### 💬 Semantic News Analysis")
    s1, s2 = st.columns(2)
    with s1:
        fig_p = px.pie(values=[pos, neu, neg], names=["Positive", "Neutral", "Negative"], hole=0.6,
                       color_discrete_map={"Positive":"#22c55e", "Neutral":"#a7b0c0", "Negative":"#ff4d6d"})
        st.plotly_chart(fig_p, use_container_width=True)
    with s2:
        fig_g = go.Figure(go.Indicator(mode="gauge+number", value=sent_score, title={'text': "Sentiment Index"},
                                       gauge={'axis': {'range': [-100, 100]}, 'bar': {'color': "#00e5ff"}}))
        st.plotly_chart(fig_g, use_container_width=True)
    
    st.dataframe(pd.DataFrame(sentiment_details), use_container_width=True)

# =========================================================
# TAB 5: NEWSFEED
# =========================================================
with tab_news:
    st.markdown("### 📰 Latest Strategic Headlines")
    if not news:
        st.info("No recent news found for this ticker.")
    else:
        for n in news[:news_limit]:
            lbl = n.get("sentiment", "neutral")
            st.markdown(f"""
            <div class="glass news-card">
              <div style="display:flex; justify-content:space-between;">
                <a href="{n.get('link','#')}" target="_blank" class="news-title">{n.get('title','Untitled')}</a>
                {_sentiment_badge(lbl)}
              </div>
              <div class="news-meta">{n.get('source','Unknown Source')} · {n.get('published','Recent')}</div>
              <div class="news-desc">{n.get('description','')[:200]}...</div>
            </div>
            """, unsafe_allow_html=True)

# =========================================================
# TAB 6: ALERT HUB (FIXED & REWRITTEN)
# =========================================================
with tab_alerts:
    st.markdown("### 🔔 Slack Notification Center")
    
    a_left, a_right = st.columns(2, gap="large")
    
    with a_left:
        st.markdown("""
        <div class="glass">
            <h4>Manual Strategic Alert</h4>
            <p class="small-note">Instantly push the current AI-generated strategy and market metrics to your linked Slack channel.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚨 Dispatch Strategic Slack Alert", use_container_width=True):
            if not slack_enabled:
                st.warning("Slack Integration is currently disabled in Sidebar.")
            else:
                with st.spinner("Compiling strategic blocks..."):
                    # Generate the fixed Slack message
                    msg_text = build_slack_message(
                        company=company,
                        ticker=ticker,
                        last_price=last_close,
                        forecast_df=forecast_df,
                        sentiment_score=sent_score,
                        strategic_signal=strategic_signal,
                        llm_summary=llm_explanation
                    )
                    
                    # Send via core function
                    success = send_slack({"text": msg_text})
                    
                    if success:
                        st.success("✅ Strategic Alert successfully delivered to Slack.")
                    else:
                        st.error("❌ Notification failed. Check Webhook URL in secrets.")

    with a_right:
        st.markdown("""
        <div class="glass">
            <h4>System Diagnostics</h4>
            <p class="small-note">Verify that your connection to the Slack API is active and functioning.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🧪 Send Connection Test", use_container_width=True):
            test_ok = send_slack({"text": f"🔧 *Diagnostic:* Intelligence Engine connection test for {company} successful."})
            if test_ok:
                st.toast("Connection verified.", icon="✔")
            else:
                st.error("Diagnostics failed.")

    st.markdown("---")
    with st.expander("🛠️ Developer: View Alert JSON Payload"):
        st.json(alert_payload)

# ------------------------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------------------------
st.markdown(
    "<br/><div style='text-align:center; color: #4a5568; font-size: 0.8rem;'>Market Intelligence Engine v2.4.0-Stable | Proprietary Internal Use Only</div>",
    unsafe_allow_html=True
)
