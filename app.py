"""
Cook County High Schools — Variable Impact Explorer
====================================================
An insight-driven Streamlit dashboard that visualises the Top-5
drivers of every educational outcome identified by the ElasticNet
pipeline in machine_learning_clean.py.

Run:  streamlit run app.py          (inside conda env dap)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import pearsonr

# ── Paths ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "final_merged.csv"
COEF_PATH = ROOT / "outputs" / "elasticnet_coefficients.json"

# ── Human-readable labels ────────────────────────────────────────────
FEATURE_LABELS = {
    # School-level
    "x_ap_coursework": "AP Coursework",
    "x_attendance_rate": "Attendance Rate",
    "x_dropout_rate": "Dropout Rate",
    "x_enrollment": "Enrollment",
    "x_mobility_rate": "Student Mobility Rate",
    "x_pct_asian": "% Asian Students",
    "x_pct_black": "% Black Students",
    "x_pct_el": "% English Learners",
    "x_pct_hispanic": "% Hispanic Students",
    "x_pct_homeless": "% Homeless Students",
    "x_pct_iep": "% IEP Students",
    "x_pct_low_income": "% Low-Income Students",
    "x_pct_white": "% White Students",
    "x_suspension_rate": "Suspension Rate",
    "x_teacher_attendance": "Teacher Attendance",
    "x_teacher_retention": "Teacher Retention",
    # School-level counts
    "x_n_black": "# Black Students",
    "x_n_hispanic": "# Hispanic Students",
    "x_n_white": "# White Students",
    "x_n_low_income": "# Low-Income Students",
    # School-level pct (segregation)
    "x_pct_black_school": "% Black (School)",
    "x_pct_hispanic_school": "% Hispanic (School)",
    "x_pct_white_school": "% White (School)",
    "x_pct_low_income_school": "% Low-Income (School)",
    # County-level demographics
    "x_county_enrollment": "County Enrollment",
    "x_county_n_black": "County # Black Students",
    "x_county_n_hispanic": "County # Hispanic Students",
    "x_county_n_white": "County # White Students",
    "x_county_n_low_income": "County # Low-Income Students",
    "x_pct_black_county": "% Black (County)",
    "x_pct_hispanic_county": "% Hispanic (County)",
    "x_pct_white_county": "% White (County)",
    "x_pct_low_income_county": "% Low-Income (County)",
    # Segregation exposure indices
    "x_exp_black_white": "Exposure Index (Black–White)",
    "x_exp_white_black": "Exposure Index (White–Black)",
    "x_exp_black_hisp": "Exposure Index (Black–Hispanic)",
    "x_exp_hisp_white": "Exposure Index (Hispanic–White)",
    # Relative risk (segregation)
    "x_rr_black": "Relative Risk (Black)",
    "x_rr_hispanic": "Relative Risk (Hispanic)",
    "x_rr_white": "Relative Risk (White)",
    "x_rr_low_income": "Relative Risk (Low-Income)",
    # Census tract economics
    "x_median_hh_income": "Median HH Income (Tract)",
    "x_total_n_hh": "Total Households (Tract)",
    "x_pov_share_under_1_00": "Poverty Rate < 100% FPL",
    "x_hh_under_0.5_pov": "HH < 50% Poverty Line",
    "x_hh_between_0.5_0.74_pov": "HH 50–74% Poverty Line",
    "x_hh_between_0.75_0.99_pov": "HH 75–99% Poverty Line",
    "x_hh_between_1.00_to_1.24_pov": "HH 100–124% Poverty Line",
    "x_hh_between_1.25_to_1.49_pov": "HH 125–149% Poverty Line",
    "x_hh_between_1.50_to_1.74_pov": "HH 150–174% Poverty Line",
    "x_hh_between_1.75_to_1.84_pov": "HH 175–184% Poverty Line",
    "x_hh_between_1.85_to_1.99_pov": "HH 185–199% Poverty Line",
    "x_hh_between_2.00_to_2.99_pov": "HH 200–299% Poverty Line",
    "x_hh_between_3.00_to_3.99_pov": "HH 300–399% Poverty Line",
    "x_hh_between_4.00_to_4.99_pov": "HH 400–499% Poverty Line",
    "x_hh_between_5.00_and_over_pov": "HH ≥ 500% Poverty Line",
    # Food access / desert
    "x_Urban": "Urban Census Tract",
    "x_LILATracts_1And10": "Low-Inc. & Low Access (1 & 10 mi)",
    "x_LILATracts_halfAnd10": "Low-Inc. & Low Access (½ & 10 mi)",
    "x_LILATracts_1And20": "Low-Inc. & Low Access (1 & 20 mi)",
    "x_LILATracts_Vehicle": "Low Vehicle-Access Tract",
    "x_LowIncomeTracts": "Low-Income Tract",
    "x_LA1and10": "Low Food Access (1 & 10 mi)",
    "x_LAhalfand10": "Low Food Access (½ & 10 mi)",
    "x_LA1and20": "Low Food Access (1 & 20 mi)",
    # Geography / transport
    "x_transport_length": "Transit Route Length",
    "x_POP20": "Tract Population (2020)",
}

TARGET_LABELS = {
    "y_chronic_abs": "Chronic Absenteeism (%)",
    "y_ela_prof": "ELA Proficiency (%)",
    "y_grad_4yr": "4-Year Graduation Rate (%)",
    "y_math_prof": "Math Proficiency (%)",
}

DIRECTION_COLORS = {"Positive": "#2ecc71", "Negative": "#e74c3c"}


def label(feat: str) -> str:
    return FEATURE_LABELS.get(feat, feat)


# ── Data loading (cached) ───────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    with open(COEF_PATH) as f:
        coefs = json.load(f)
    return df, coefs


# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cook County HS — Variable Impact Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)

df, coefs = load_data()

# ── Income quintiles (for scatter colour coding) ─────────────────────
INCOME_COL = "x_median_hh_income"
df["income_quintile"] = pd.qcut(
    df[INCOME_COL].rank(method="first"),   # rank first to handle ties
    5,
    labels=["Q1 (Lowest)", "Q2", "Q3", "Q4", "Q5 (Highest)"],
)
QUINTILE_COLORS = {
    "Q1 (Lowest)":  "#d73027",   # red
    "Q2":           "#fc8d59",   # orange
    "Q3":           "#fee08b",   # yellow
    "Q4":           "#91bfdb",   # light blue
    "Q5 (Highest)": "#4575b4",   # dark blue
}
available_targets = [t for t in TARGET_LABELS if t in coefs]

# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════
st.sidebar.title("Settings")

# ── 1. Target Selector ───────────────────────────────────────────────
target = st.sidebar.selectbox(
    "Select Educational Outcome",
    available_targets,
    format_func=lambda k: TARGET_LABELS.get(k, k),
)
target_label = TARGET_LABELS[target]

info = coefs[target]
all_coef_rows = info["coefficients"]
top5 = all_coef_rows[:5]
top5_features = [c["feature"] for c in top5]
top5_scaled = {c["feature"]: c["coefficient_scaled"] for c in top5}
top5_raw = {c["feature"]: c["coefficient_original_units"] for c in top5}

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Diagnostics**")
st.sidebar.metric("Test R²", f"{info['r2_test']:.3f}")
st.sidebar.metric("Test MSE", f"{info['test_mse']:.1f}")
st.sidebar.caption(
    f"α = {info['best_alpha']:.4f} · L1 = {info['best_l1_ratio']:.2f}"
)

# ══════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════
st.title("Cook County High Schools — Variable Impact Explorer")
st.markdown(
    f"Exploring the **Top 5 standardised drivers** of "
    f"**{target_label}** across Cook County high schools."
)

# ══════════════════════════════════════════════════════════════════════
# SECTION 1 — Standardised Coefficient Bar Chart
# ══════════════════════════════════════════════════════════════════════
st.header("1. Standardised Impact — Top 5 Drivers")

bar_df = pd.DataFrame(top5)
bar_df["label"] = bar_df["feature"].map(label)
bar_df["direction"] = np.where(
    bar_df["coefficient_scaled"] > 0, "Positive", "Negative"
)
bar_df = bar_df.sort_values("coefficient_scaled")  # longest bar at top

fig_bar = px.bar(
    bar_df,
    x="coefficient_scaled",
    y="label",
    orientation="h",
    color="direction",
    color_discrete_map=DIRECTION_COLORS,
    labels={"coefficient_scaled": "Standardised Coefficient (σ units)", "label": ""},
)
fig_bar.update_layout(
    height=340,
    margin=dict(l=10, r=10, t=10, b=10),
    legend_title_text="Direction",
    xaxis_zeroline=True,
    xaxis_zerolinewidth=2,
    xaxis_zerolinecolor="grey",
)
st.plotly_chart(fig_bar, use_container_width=True)
st.caption(
    "Each bar shows how many points the outcome shifts when the "
    "predictor moves by 1 standard deviation, holding others constant."
)

# ══════════════════════════════════════════════════════════════════════
# SECTION 2 — Plain-English Interpretation
# ══════════════════════════════════════════════════════════════════════
st.header("2. Plain-English Interpretation")

for rank, feat in enumerate(top5_features, 1):
    sc = top5_scaled[feat]
    direction = "increase" if sc > 0 else "decrease"
    st.markdown(
        f"**#{rank} — {label(feat)}:** "
        f"A 1-standard-deviation increase in *{label(feat)}* is associated "
        f"with a **{abs(sc):.2f}-point {direction}** in {target_label}, "
        f"holding all other factors constant."
    )

# ══════════════════════════════════════════════════════════════════════
# SECTION 3 — Scatter Trends (univariate OLS, coloured by income Q)
# ══════════════════════════════════════════════════════════════════════
st.header("3. Real-World Scatter Trends")
st.markdown(
    "Each plot shows the **raw 1-on-1 relationship** between the driver "
    "and the outcome, overlaid with a simple OLS trend line. "
    "Points are coloured by **median household income quintile**."
)

# Shared colour legend (rendered once, outside the charts)
legend_items = "  ".join(
    f'<span style="color:{c}; font-size:22px;">●</span> {q}'
    for q, c in QUINTILE_COLORS.items()
)
st.markdown(f"<div style='text-align:center'>{legend_items}</div>", unsafe_allow_html=True)

n_top = len(top5_features)
cols_per_row = 3
rows_needed = (n_top + cols_per_row - 1) // cols_per_row

for row_idx in range(rows_needed):
    cols = st.columns(cols_per_row)
    for col_idx in range(cols_per_row):
        feat_idx = row_idx * cols_per_row + col_idx
        if feat_idx >= n_top:
            break
        feat = top5_features[feat_idx]
        with cols[col_idx]:
            if feat not in df.columns:
                st.write(f"No data for {label(feat)}")
                continue
            sub = df[[feat, target, "income_quintile"]].dropna(subset=[feat, target])
            if sub.empty:
                st.write(f"No data for {label(feat)}")
                continue

            x_arr = sub[feat].values.astype(float)
            y_arr = sub[target].values.astype(float)

            # Univariate OLS trendline — truncated at [0, 100] boundary
            slope, intercept_ols = np.polyfit(x_arr, y_arr, 1)
            x_line = np.linspace(x_arr.min(), x_arr.max(), 300)
            y_raw = intercept_ols + slope * x_line
            in_bounds = (y_raw >= 0) & (y_raw <= 100)
            x_line = x_line[in_bounds]
            y_line = y_raw[in_bounds]

            fig = go.Figure()

            # Scatter points by income quintile
            for q_label in ["Q1 (Lowest)", "Q2", "Q3", "Q4", "Q5 (Highest)"]:
                mask = sub["income_quintile"] == q_label
                if not mask.any():
                    continue
                fig.add_trace(go.Scatter(
                    x=sub.loc[mask, feat],
                    y=sub.loc[mask, target],
                    mode="markers",
                    marker=dict(size=5, opacity=0.45, color=QUINTILE_COLORS[q_label]),
                    name=q_label,
                    showlegend=False,
                ))

            # OLS trend line — bright white for dark mode, still visible on light
            fig.add_trace(go.Scatter(
                x=x_line, y=y_line,
                mode="lines",
                line=dict(color="#FF4B4B", width=3),
                name=f"OLS (slope={slope:+.3f})",
                showlegend=False,
            ))

            fig.update_layout(
                title=dict(text=label(feat), font=dict(size=13)),
                xaxis_title=label(feat),
                yaxis_title=target_label,
                yaxis_range=[0, 100],
                height=340,
                margin=dict(l=10, r=10, t=35, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Trend lines are simple univariate OLS, truncated where they "
    "exit the [0, 100] range."
)

# ══════════════════════════════════════════════════════════════════════
# SECTION 4 — Per-Variable Statistics Table
# ══════════════════════════════════════════════════════════════════════
st.header(f"4. Variable Statistics — {target_label}")
st.markdown(
    "Univariate relationship between each predictor and the selected outcome. "
    "**Std. Coef** and **Raw Coef** come from the multivariate ElasticNet; "
    "**Pearson r**, **R²**, and **Adj R²** are from simple 1-on-1 regressions."
)

stat_rows = []
for c in all_coef_rows:
    feat = c["feature"]
    sc = c["coefficient_scaled"]
    raw_c = c["coefficient_original_units"]
    if sc == 0:
        continue
    if feat not in df.columns:
        continue
    sub = df[[feat, target]].dropna()
    if len(sub) < 3:
        continue
    x_vals = sub[feat].values.astype(float)
    y_vals = sub[target].values.astype(float)
    corr, pval = pearsonr(x_vals, y_vals)
    r2 = corr ** 2
    n = len(sub)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - 2)
    stat_rows.append({
        "Variable": label(feat),
        "Std. Coef": round(sc, 4),
        "Raw Coef": round(raw_c, 4),
        "Pearson r": round(corr, 3),
        "R²": round(r2, 3),
        "Adj R²": round(adj_r2, 3),
        "p-value": f"{pval:.2e}" if pval < 0.001 else f"{pval:.4f}",
        "n": n,
    })

if stat_rows:
    st.dataframe(
        pd.DataFrame(stat_rows).set_index("Variable"),
        use_container_width=True,
    )

# ══════════════════════════════════════════════════════════════════════
# SECTION 5 — What-If Simulator
# ══════════════════════════════════════════════════════════════════════
st.header("5. What-If Scenario Simulator")
st.markdown(
    "Select a baseline school, then adjust the Top-5 drivers. "
    f"The linear model estimates how **{target_label}** would shift."
)

# Most recent observation per school
latest = df.sort_values("year").groupby("School Name").last().reset_index()
school_names = sorted(latest["School Name"].str.strip().unique())
chosen = st.selectbox("Baseline School", school_names)

row = latest[latest["School Name"].str.strip() == chosen].iloc[0]
baseline_y = row[target]

if pd.isna(baseline_y):
    st.warning(
        f"This school has no recorded {target_label} in its most recent year. "
        "Please choose a different school."
    )
else:
    st.markdown(
        f"**Baseline {target_label}:** {baseline_y:.1f}%  "
        f"*(Year {int(row['year'])})*"
    )

    # Sliders for Top 5
    deltas: dict[str, float] = {}
    slider_cols = st.columns(cols_per_row)
    for idx, feat in enumerate(top5_features):
        col = slider_cols[idx % cols_per_row]
        if feat not in df.columns:
            deltas[feat] = 0.0
            continue
        feat_val = row[feat]
        if pd.isna(feat_val):
            feat_val = float(df[feat].median())
        feat_val = float(feat_val)
        feat_min = float(df[feat].min())
        feat_max = float(df[feat].max())
        with col:
            new_val = st.slider(
                label(feat),
                min_value=feat_min,
                max_value=feat_max,
                value=feat_val,
                key=f"sim_{feat}",
            )
            deltas[feat] = new_val - feat_val

    # Compute predicted change using original-units coefficients
    total_delta = sum(deltas[f] * top5_raw[f] for f in top5_features)
    new_y = baseline_y + total_delta

    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("Baseline", f"{baseline_y:.1f}%")
    m2.metric("Predicted Change", f"{total_delta:+.1f} pp", delta=f"{total_delta:+.1f}")
    m3.metric("New Estimate", f"{new_y:.1f}%", delta=f"{total_delta:+.1f}")

    # Breakdown table when sliders have moved
    if any(d != 0 for d in deltas.values()):
        st.markdown("**Contribution Breakdown**")
        rows_out = []
        for feat in top5_features:
            d = deltas[feat]
            if d == 0:
                continue
            impact = d * top5_raw[feat]
            rows_out.append({
                "Variable": label(feat),
                "Change (delta x)": f"{d:+.2f}",
                "Coefficient (raw)": f"{top5_raw[feat]:+.4f}",
                "Impact on Outcome (pp)": f"{impact:+.2f}",
            })
        st.table(pd.DataFrame(rows_out))

# ── Footer ───────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Model: ElasticNet with Median Imputer + StandardScaler · "
    "GroupKFold CV by school · "
    "Top-5 ranked by |standardised coefficient| · "
    "Data: ISBE Report Cards + ACS / USDA"
)
