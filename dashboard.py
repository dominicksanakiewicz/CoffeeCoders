"""
Cook County High Schools — Educational Drivers Dashboard
========================================================
Visualises the Top-5 most important variables from a fixed ElasticNet
pipeline and provides a What-If scenario simulator.

Run:  streamlit run dashboard.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Paths ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "panel_yx_highschools.csv"
COEF_PATH = ROOT / "outputs" / "elasticnet_coefficients.json"

# ── Human-readable labels ────────────────────────────────────────────
FEATURE_LABELS = {
    "x_tract_urban": "Urban Census Tract",
    "x_tract_pov_share_under_1_00": "Poverty Rate (< 100% FPL)",
    "x_tract_lilatracts_vehicle": "Low Vehicle-Access Tract",
    "x_tract_lahalfand10": "Low Food Access (½ & 10 mi)",
    "x_tract_la1and10": "Low Food Access (1 & 10 mi)",
    "x_tract_la1and20": "Low Food Access (1 & 20 mi)",
    "x_tract_lilatracts_halfand10": "Low-Income & Low Access (½ & 10 mi)",
    "x_tract_lowincometracts": "Low-Income Tract",
    "x_tract_lilatracts_1and10": "Low-Income & Low Access (1 & 10 mi)",
    "x_tract_lilatracts_1and20": "Low-Income & Low Access (1 & 20 mi)",
    "x_pct_asian": "% Asian Students",
    "x_pct_el": "% English Learners",
    "x_pct_iep": "% IEP Students",
    "x_dropout_rate": "Dropout Rate",
    "x_mobility_rate": "Student Mobility Rate",
    "x_attendance_rate": "Attendance Rate",
    "x_pct_homeless": "% Homeless Students",
    "x_pct_white": "% White Students",
    "x_pct_black": "% Black Students",
    "x_pct_low_income": "% Low-Income Students",
    "x_pct_hispanic": "% Hispanic Students",
    "x_enrollment": "Enrollment",
    "x_suspension_rate": "Suspension Rate",
    "x_teacher_attendance": "Teacher Attendance",
    "x_teacher_retention": "Teacher Retention",
    "x_ap_coursework": "AP Coursework",
    "x_tract_median_hh_income": "Median Household Income",
    "x_tract_total_n_hh": "Total Households in Tract",
    "x_tract_pop20": "Tract Population (2020)",
    "x_tract_transport_length": "Transit Route Length",
}

TARGET_LABELS = {
    "y_math_prof": "Math Proficiency (%)",
    "y_grad_4yr": "4-Year Graduation Rate (%)",
}


# ── Data loading (cached) ───────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    with open(COEF_PATH) as f:
        coefs = json.load(f)
    return df, coefs


def label(feat: str) -> str:
    return FEATURE_LABELS.get(feat, feat)


# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cook County HS — Educational Drivers",
    layout="wide",
    initial_sidebar_state="expanded",
)

df, coefs = load_data()

# ── Sidebar ──────────────────────────────────────────────────────────
st.sidebar.title("Settings")
target = st.sidebar.radio(
    "Target Outcome",
    list(TARGET_LABELS.keys()),
    format_func=lambda k: TARGET_LABELS[k],
)

target_info = coefs[target]
all_coefs = target_info["coefficients"]
top5 = all_coefs[:5]
top5_features = [c["feature"] for c in top5]
top5_coefs = {c["feature"]: c["coefficient_original_units"] for c in top5}

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Performance**")
st.sidebar.metric("Train MSE", f"{target_info['train_mse']:.1f}")
st.sidebar.metric("Test MSE", f"{target_info['test_mse']:.1f}")
st.sidebar.markdown(
    f"Best α = {target_info['best_alpha']:.4f},  "
    f"L1 ratio = {target_info['best_l1_ratio']:.2f}"
)

# ── Header ───────────────────────────────────────────────────────────
st.title("Cook County High Schools — Variable Impact Explorer")
st.markdown(
    f"Showing the **Top 5 drivers** of **{TARGET_LABELS[target]}** "
    "identified by the ElasticNet model."
)

# ══════════════════════════════════════════════════════════════════════
# VISUALISATION 1 — Coefficient Bar Chart
# ══════════════════════════════════════════════════════════════════════
st.header("1. Coefficient Impact (Top 5)")

coef_df = pd.DataFrame(top5)
coef_df["label"] = coef_df["feature"].map(label)
coef_df["direction"] = np.where(
    coef_df["coefficient_original_units"] > 0, "Positive", "Negative"
)
# Sort smallest-to-largest so the longest bar appears at the top
coef_df = coef_df.sort_values("coefficient_original_units")

fig_bar = px.bar(
    coef_df,
    x="coefficient_original_units",
    y="label",
    orientation="h",
    color="direction",
    color_discrete_map={"Positive": "#2ecc71", "Negative": "#e74c3c"},
    labels={
        "coefficient_original_units": f"Change in {TARGET_LABELS[target]} per 1-unit increase",
        "label": "",
    },
)
fig_bar.update_layout(
    height=350,
    margin=dict(l=10, r=10, t=10, b=10),
    legend_title_text="Direction",
    xaxis_zeroline=True,
    xaxis_zerolinewidth=2,
    xaxis_zerolinecolor="grey",
)
st.plotly_chart(fig_bar, use_container_width=True)

st.caption(
    "Each bar shows how much the target outcome changes (in percentage points) "
    "for a 1-unit increase in the predictor, holding all else constant."
)

# ══════════════════════════════════════════════════════════════════════
# VISUALISATION 2 — Scatter Trends
# ══════════════════════════════════════════════════════════════════════
st.header("2. Scatter Trends — Raw Data + ElasticNet Slope")

cols = st.columns(min(len(top5_features), 3))

for idx, feat in enumerate(top5_features):
    col = cols[idx % len(cols)]
    with col:
        sub = df[[feat, target]].dropna()
        if sub.empty:
            st.write(f"No data for {label(feat)}")
            continue

        x_vals = sub[feat].values
        y_vals = sub[target].values

        # OLS trend for the scatter (simple bivariate, not ElasticNet)
        # but overlay the ElasticNet slope anchored at mean for comparison
        coef_val = top5_coefs[feat]
        x_mean = np.nanmean(x_vals)
        y_mean = np.nanmean(y_vals)

        x_range = np.linspace(x_vals.min(), x_vals.max(), 100)
        y_trend = y_mean + coef_val * (x_range - x_mean)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers",
                marker=dict(size=5, opacity=0.4, color="#3498db"),
                name="Observations",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y_trend,
                mode="lines",
                line=dict(color="#e74c3c", width=3),
                name=f"ElasticNet (coef={coef_val:+.2f})",
            )
        )
        fig.update_layout(
            title=dict(text=label(feat), font=dict(size=13)),
            xaxis_title=label(feat),
            yaxis_title=TARGET_LABELS[target],
            height=320,
            margin=dict(l=10, r=10, t=40, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Each scatter shows raw school-level observations. The red line is the "
    "ElasticNet regression slope (anchored at the data mean)."
)

# ══════════════════════════════════════════════════════════════════════
# WHAT-IF SCENARIO SIMULATOR
# ══════════════════════════════════════════════════════════════════════
st.header("3. What-If Scenario Simulator")
st.markdown(
    "Select a baseline school, then adjust the Top-5 drivers to see how "
    f"**{TARGET_LABELS[target]}** would change."
)

# Build a school selector: use most recent year per school
latest = df.sort_values("year").groupby("school_name").last().reset_index()
school_list = sorted(latest["school_name"].str.strip().unique())
chosen_school = st.selectbox("Baseline School", school_list)

school_row = latest[latest["school_name"].str.strip() == chosen_school].iloc[0]
baseline_y = school_row[target]

if pd.isna(baseline_y):
    st.warning(
        f"The selected school has no observed {TARGET_LABELS[target]} in its "
        "most recent year. Please choose another school."
    )
else:
    st.markdown(
        f"**Baseline {TARGET_LABELS[target]}:** {baseline_y:.1f}%  "
        f"(Year {int(school_row['year'])})"
    )

    # Build sliders
    deltas = {}
    slider_cols = st.columns(min(len(top5_features), 3))
    for idx, feat in enumerate(top5_features):
        col = slider_cols[idx % len(slider_cols)]
        feat_val = school_row[feat]
        # Fall back to median if school has NaN for this feature
        if pd.isna(feat_val):
            feat_val = df[feat].median()
        feat_val = float(feat_val)

        # Determine sensible slider range from the data distribution
        feat_min = float(df[feat].min())
        feat_max = float(df[feat].max())

        with col:
            new_val = st.slider(
                label(feat),
                min_value=feat_min,
                max_value=feat_max,
                value=feat_val,
                key=f"slider_{feat}",
            )
            deltas[feat] = new_val - feat_val

    # Compute predicted change
    total_delta = sum(deltas[f] * top5_coefs[f] for f in top5_features)
    new_y = baseline_y + total_delta

    # Display results
    st.markdown("---")
    res_cols = st.columns(3)
    res_cols[0].metric("Baseline", f"{baseline_y:.1f}%")
    res_cols[1].metric(
        "Predicted Change", f"{total_delta:+.1f} pp", delta=f"{total_delta:+.1f}"
    )
    res_cols[2].metric(
        "New Estimate",
        f"{new_y:.1f}%",
        delta=f"{total_delta:+.1f}",
    )

    # Breakdown table
    if any(d != 0 for d in deltas.values()):
        st.markdown("**Contribution Breakdown**")
        breakdown = []
        for feat in top5_features:
            d = deltas[feat]
            if d != 0:
                impact = d * top5_coefs[feat]
                breakdown.append(
                    {
                        "Variable": label(feat),
                        "Change (Δx)": f"{d:+.2f}",
                        "Coefficient": f"{top5_coefs[feat]:+.4f}",
                        "Impact on Outcome (pp)": f"{impact:+.2f}",
                    }
                )
        st.table(pd.DataFrame(breakdown))

# ── Footer ───────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Model: ElasticNet with MedianImputer + StandardScaler | "
    "GroupKFold CV by school | Data: ISBE Report Cards + ACS/USDA"
)
