
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "cyberattack_model.pkl"
DEFAULT_DATASET = BASE_DIR / "dataset" / "DDoS-TCP_Flood.pcap.csv"


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="IoT Cyberattack Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# DARK CYBERSECURITY UI
# ---------------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 10% 0%, #102b4c 0%, transparent 32%),
            linear-gradient(135deg, #061323 0%, #07182b 55%, #030b17 100%);
        color: #eaf3ff;
    }

    [data-testid="stSidebar"] {
        background: #061426;
        border-right: 1px solid #12345a;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    h1, h2, h3 {
        color: #eef6ff !important;
    }

    p, label, span {
        color: #c4d6ed;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #0b2340, #08172b);
        border: 1px solid #174675;
        padding: 18px;
        border-radius: 14px;
        box-shadow: 0 0 18px rgba(0, 153, 255, 0.05);
    }

    div[data-testid="stMetricLabel"] {
        color: #a9c6e8 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #f4f9ff !important;
    }

    div[data-testid="stMetricDelta"] {
        color: #00e6a0 !important;
    }

    .top-banner {
        background: linear-gradient(110deg, #0c2b4d, #091a30);
        border: 1px solid #164775;
        border-radius: 18px;
        padding: 22px 26px;
        margin-bottom: 18px;
    }

    .top-title {
        font-size: 27px;
        font-weight: 750;
        color: #f1f7ff;
    }

    .top-subtitle {
        color: #9fc3e9;
        font-size: 14px;
        margin-top: 6px;
    }

    .status-pill {
        display: inline-block;
        background: #073b35;
        border: 1px solid #087d69;
        color: #4df5c2;
        border-radius: 30px;
        padding: 7px 13px;
        font-size: 13px;
        font-weight: 650;
    }

    .section-card {
        background: linear-gradient(145deg, #0b2038, #08172a);
        border: 1px solid #15385e;
        border-radius: 16px;
        padding: 18px;
    }

    .small-muted {
        color: #8faecc;
        font-size: 12px;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #15385e;
        border-radius: 12px;
    }

    .stButton button {
        background: linear-gradient(90deg, #087cf0, #075bd5);
        color: white;
        border: 1px solid #2994ff;
        border-radius: 10px;
        font-weight: 650;
        min-height: 44px;
    }

    .stButton button:hover {
        background: #0a8bff;
        border-color: #5bb1ff;
        color: white;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

model = load_model()
expected_features = list(model.feature_names_in_)


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

header_left, header_right = st.columns([5, 1])

with header_left:
    st.markdown(
        """
        <div class="top-banner">
            <div class="top-title">
                🛡️ AI-Based Unknown Cyberattack Detection
                for IoT Networks
            </div>
            <div class="top-subtitle">
                Detect &nbsp; • &nbsp; Analyze &nbsp; • &nbsp; Protect
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with header_right:
    st.markdown(
        """
        <div style="text-align:right; padding-top:12px;">
            <span class="status-pill">● Model Ready</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.markdown(
    """
    <div style="text-align:center; padding:12px 0 24px 0;">
        <div style="font-size:54px;">🛡️</div>
        <div style="font-size:19px; font-weight:700; color:#eaf4ff;">
            IoT Secure AI
        </div>
        <div style="font-size:12px; color:#7fa7d2;">
            Cyberattack Detection
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Live Detection",
        "Dataset Overview",
        "Model Information",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### System Status")
st.sidebar.success("Model loaded")
st.sidebar.caption(f"Features expected: {len(expected_features)}")
st.sidebar.caption("Live packet capture: not connected")


# ---------------------------------------------------------
# DATA LOADING AND PREDICTION
# ---------------------------------------------------------

def prepare_features(data):
    missing = [
        feature
        for feature in expected_features
        if feature not in data.columns
    ]

    if missing:
        raise ValueError(
            "Missing model features: " + ", ".join(missing)
        )

    X = data[expected_features].copy()

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.clip(lower=-1e10, upper=1e10)
    X = X.fillna(0)

    return X


def run_detection(data):
    X = prepare_features(data)
    predictions = model.predict(X)

    result = data.copy()
    result["Prediction"] = predictions
    result["Prediction Name"] = result["Prediction"].map(
        {0: "Normal", 1: "Attack"}
    )

    return result


def load_default_dataset():
    if DEFAULT_DATASET.exists():
        return pd.read_csv(DEFAULT_DATASET, nrows=50000)

    return None


# ---------------------------------------------------------
# DATASET SELECTION
# ---------------------------------------------------------

st.markdown("### Network Traffic Analysis")
st.caption(
    "Upload a feature CSV to analyze it with the trained model. "
    "The default sample is the TCP Flood dataset, if available."
)

uploaded_file = st.file_uploader(
    "Upload network traffic CSV",
    type=["csv"],
)

if uploaded_file is not None:
    try:
        source_data = pd.read_csv(uploaded_file)
        dataset_name = uploaded_file.name
    except Exception as error:
        st.error(f"Could not read CSV: {error}")
        st.stop()
else:
    source_data = load_default_dataset()
    dataset_name = "Default TCP Flood sample"

if source_data is None:
    st.info(
        "Upload a CSV file to begin. "
        "The default TCP Flood dataset was not found."
    )
    st.stop()

if page == "Live Detection":
    st.warning(
        "This page currently analyzes uploaded or sample CSV data. "
        "It is not connected to live packet capture."
    )


# ---------------------------------------------------------
# RUN PREDICTION
# ---------------------------------------------------------

try:
    results = run_detection(source_data)
except Exception as error:
    st.error(f"Prediction could not be completed: {error}")
    st.stop()

total_records = len(results)
attack_count = int((results["Prediction"] == 1).sum())
normal_count = int((results["Prediction"] == 0).sum())

attack_percentage = (
    attack_count / total_records * 100
    if total_records > 0
    else 0
)

normal_percentage = (
    normal_count / total_records * 100
    if total_records > 0
    else 0
)


# ---------------------------------------------------------
# DASHBOARD PAGE
# ---------------------------------------------------------

if page == "Dashboard":

    st.markdown("### Welcome to the Cyberattack Detection Dashboard")
    st.caption(
        f"Currently analyzing: {dataset_name}"
    )

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Records",
        f"{total_records:,}",
        help="Number of records analyzed in the selected CSV.",
    )

    c2.metric(
        "Predicted Attacks",
        f"{attack_count:,}",
        f"{attack_percentage:.2f}% of records",
    )

    c3.metric(
        "Predicted Normal",
        f"{normal_count:,}",
        f"{normal_percentage:.2f}% of records",
    )

    c4.metric(
        "Model Features",
        f"{len(expected_features)}",
        "Loaded successfully",
    )

    st.markdown("")

    # Charts
    chart_left, chart_right = st.columns([1, 1.35])

    with chart_left:
        st.markdown("#### Traffic Classification")

        traffic_chart = pd.DataFrame(
            {
                "Traffic": ["Normal", "Attack"],
                "Records": [normal_count, attack_count],
            }
        )

        fig = px.pie(
            traffic_chart,
            names="Traffic",
            values="Records",
            hole=0.66,
            color="Traffic",
            color_discrete_map={
                "Normal": "#00d69a",
                "Attack": "#ff435b",
            },
        )

        fig.update_traces(
            textinfo="percent",
            textfont_size=14,
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=5, r=5, t=15, b=5),
            legend=dict(
                orientation="h",
                y=-0.1,
            ),
        )

        st.plotly_chart(fig, use_container_width=True)

    with chart_right:
        st.markdown("#### Prediction Summary")

        summary_chart = pd.DataFrame(
            {
                "Class": ["Normal Traffic", "Attack Traffic"],
                "Percentage": [
                    normal_percentage,
                    attack_percentage,
                ],
            }
        )

        fig = px.bar(
            summary_chart,
            x="Class",
            y="Percentage",
            color="Class",
            text="Percentage",
            color_discrete_map={
                "Normal Traffic": "#168cff",
                "Attack Traffic": "#ff435b",
            },
        )

        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside",
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Percentage",
            xaxis_title="",
            yaxis_range=[0, 110],
            showlegend=False,
            margin=dict(l=5, r=5, t=15, b=5),
        )

        st.plotly_chart(fig, use_container_width=True)

    # Lower section
    lower_left, lower_right = st.columns([1.4, 1])

    with lower_left:
        st.markdown("#### Recent Detection Records")

        display_columns = [
            col
            for col in [
                "frame.time_epoch",
                "ip.src",
                "ip.dst",
                "Protocol Type",
                "Prediction Name",
            ]
            if col in results.columns
        ]

        if not display_columns:
            display_columns = [
                "Prediction Name"
            ]

        recent = results[display_columns].tail(10).copy()

        recent = recent.rename(
            columns={
                "frame.time_epoch": "Timestamp",
                "ip.src": "Source IP",
                "ip.dst": "Destination IP",
                "Protocol Type": "Protocol",
                "Prediction Name": "Detection",
            }
        )

        st.dataframe(
            recent,
            use_container_width=True,
            hide_index=True,
        )

    with lower_right:
        st.markdown("#### Detection Status")

        if attack_count > 0:
            st.error(
                f"⚠️ {attack_count:,} records predicted as attack"
            )
            st.caption(
                "Review the detection results. Predictions are "
                "model classifications, not verified attack labels."
            )
        else:
            st.success("🛡️ No records predicted as attacks")
            st.caption(
                "No attack predictions were returned for this dataset."
            )

        st.markdown("#### Dataset")
        st.info(dataset_name)

        st.markdown("#### Model")
        st.info("Random Forest classifier")


# ---------------------------------------------------------
# LIVE DETECTION PAGE
# ---------------------------------------------------------

elif page == "Live Detection":

    st.markdown("### Detection Console")

    st.info(
        "Select a CSV above to analyze its records. "
        "This dashboard does not currently capture packets directly."
    )

    col1, col2 = st.columns(2)

    col1.metric("Records analyzed", f"{total_records:,}")
    col2.metric("Attack predictions", f"{attack_count:,}")

    st.markdown("#### Detection Results")

    st.dataframe(
        results.tail(100),
        use_container_width=True,
        hide_index=True,
    )

    csv_data = results.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Prediction Results",
        data=csv_data,
        file_name="iot_detection_results.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------
# DATASET OVERVIEW PAGE
# ---------------------------------------------------------

elif page == "Dataset Overview":

    st.markdown("### Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", f"{source_data.shape[0]:,}")
    col2.metric("Columns", f"{source_data.shape[1]:,}")
    col3.metric(
        "Missing Values",
        f"{int(source_data.isna().sum().sum()):,}",
    )

    st.markdown("#### Column Preview")

    st.dataframe(
        source_data.head(100),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Numeric Feature Statistics")

    st.dataframe(
        source_data.describe().T,
        use_container_width=True,
    )


# ---------------------------------------------------------
# MODEL INFORMATION PAGE
# ---------------------------------------------------------

elif page == "Model Information":

    st.markdown("### Model Information")

    st.markdown(
        """
        <div class="section-card">
            <h3>Random Forest Classifier</h3>
            <p>
                The trained model classifies input network traffic
                records into Normal or Attack categories.
            </p>
            <p>
                <b>Model file:</b> cyberattack_model.pkl
            </p>
            <p>
                <b>Input features:</b> 39 expected columns
            </p>
            <p>
                <b>Output classes:</b> Normal (0), Attack (1)
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Features Used by the Model")

    feature_df = pd.DataFrame(
        {
            "Feature": expected_features,
        }
    )

    st.dataframe(
        feature_df,
        use_container_width=True,
        hide_index=True,
    )

    st.warning(
        "The dashboard displays predictions. "
        "A verified accuracy score requires independently labeled "
        "test data and a comparison against the true labels."
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#7897b8; font-size:12px;">
        IoT Cyberattack Detection System • AI-powered Network Security
    </div>
    """,
    unsafe_allow_html=True,
)
