"""
model_baseline.py
=================
4-model comparison (Lasso, Random Forest, HistGradientBoosting, LightGBM)
for predicting y_math_prof and y_grad_4yr for Cook County high schools.

SHAP analysis (beeswarm + dependence) from LightGBM.
Permutation importance + PDP/ICE from all models.

Usage:
    /opt/miniconda3/envs/dap/bin/python model_baseline.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import shap

from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from datetime import datetime

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
OUT  = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)

# ======================================================================
# 1. Load & prepare data
# ======================================================================
print("=" * 70)
print("BASELINE MODELING PIPELINE")
print(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

df = pd.read_csv(os.path.join(DATA, "panel_yx_highschools.csv"), low_memory=False)
print(f"\n[1] Loaded panel: {df.shape[0]:,} rows × {df.shape[1]} cols")

# --- Feature definitions ---
FEATURES = [
    # Demographics (school-level)
    "x_pct_black", "x_pct_hispanic", "x_pct_white", "x_pct_asian",
    "x_pct_low_income", "x_pct_homeless", "x_pct_el", "x_pct_iep",
    # School operations
    "x_enrollment", "x_attendance_rate", "x_dropout_rate",
    "x_mobility_rate", "x_suspension_rate", "x_ap_coursework",
    # Teacher
    "x_teacher_retention",
    # Neighborhood (tract-level)
    "x_tract_median_hh_income", "x_tract_pov_share_under_1_00",
    "x_tract_pop20", "x_tract_transport_length",
    "x_tract_lowincometracts", "x_tract_urban",
    "x_tract_la1and10", "x_tract_lahalfand10", "x_tract_la1and20",
]

FEATURE_GROUPS = {
    "Demographics":  ["x_pct_black", "x_pct_hispanic", "x_pct_white",
                      "x_pct_asian", "x_pct_low_income", "x_pct_homeless",
                      "x_pct_el", "x_pct_iep"],
    "School Ops":    ["x_enrollment", "x_attendance_rate", "x_dropout_rate",
                      "x_mobility_rate", "x_suspension_rate", "x_ap_coursework"],
    "Teacher":       ["x_teacher_retention"],
    "Neighborhood":  ["x_tract_median_hh_income", "x_tract_pov_share_under_1_00",
                      "x_tract_pop20", "x_tract_transport_length",
                      "x_tract_lowincometracts", "x_tract_urban",
                      "x_tract_la1and10", "x_tract_lahalfand10",
                      "x_tract_la1and20"],
}

feat_to_group = {}
for grp, feats in FEATURE_GROUPS.items():
    for f in feats:
        feat_to_group[f] = grp

# Pretty names for plots
FEAT_LABELS = {
    "x_pct_low_income": "% Low Income",
    "x_pct_black": "% Black",
    "x_pct_hispanic": "% Hispanic",
    "x_pct_white": "% White",
    "x_pct_asian": "% Asian",
    "x_pct_homeless": "% Homeless",
    "x_pct_el": "% English Learner",
    "x_pct_iep": "% IEP",
    "x_enrollment": "Enrollment",
    "x_attendance_rate": "Attendance Rate",
    "x_dropout_rate": "Dropout Rate",
    "x_mobility_rate": "Mobility Rate",
    "x_suspension_rate": "Suspension Rate",
    "x_ap_coursework": "AP Coursework",
    "x_teacher_retention": "Teacher Retention",
    "x_tract_median_hh_income": "Tract Median HH Income",
    "x_tract_pov_share_under_1_00": "Tract Poverty Rate",
    "x_tract_pop20": "Tract Pop (2020)",
    "x_tract_transport_length": "Tract Transit Length",
    "x_tract_lowincometracts": "Low Income Tract",
    "x_tract_urban": "Urban Tract",
    "x_tract_la1and10": "Low Access 1mi+10mi",
    "x_tract_lahalfand10": "Low Access 0.5mi+10mi",
    "x_tract_la1and20": "Low Access 1mi+20mi",
}

TARGETS = {
    "y_math_prof": {"exclude_years": [2020], "label": "Math Proficiency (%)"},
    "y_grad_4yr":  {"exclude_years": [],      "label": "4-Year Graduation Rate (%)"},
}

FEATURES = [f for f in FEATURES if f in df.columns]
print(f"  Using {len(FEATURES)} features")

# ======================================================================
# 2. Model definitions
# ======================================================================

def make_lasso():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("lasso",   LassoCV(cv=5, max_iter=10000, random_state=42)),
    ])

def make_rf():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestRegressor(
            n_estimators=300, max_depth=12, min_samples_leaf=5,
            random_state=42, n_jobs=-1)),
    ])

def make_hgb():
    return HistGradientBoostingRegressor(
        max_iter=500, max_depth=8, learning_rate=0.05,
        max_leaf_nodes=31, min_samples_leaf=10,
        random_state=42,
    )

def make_lgb():
    return lgb.LGBMRegressor(
        n_estimators=500, max_depth=8, learning_rate=0.05,
        num_leaves=31, min_child_samples=10,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1, n_jobs=-1,
    )

MODEL_FACTORIES = {
    "Lasso":              make_lasso,
    "RandomForest":       make_rf,
    "GradientBoosting":   make_hgb,
    "LightGBM":           make_lgb,
}

# ======================================================================
# 3. Train & evaluate
# ======================================================================

all_results = []
all_perm_imp = {}
best_models = {}
lgb_models = {}  # for SHAP

for target, cfg in TARGETS.items():
    print(f"\n{'='*70}")
    print(f"TARGET: {target} — {cfg['label']}")
    print(f"{'='*70}")

    mask = df[target].notna()
    for yr in cfg["exclude_years"]:
        mask &= (df["year"] != yr)
    sub = df.loc[mask].copy()
    print(f"  Rows after filtering: {len(sub):,}")

    X = sub[FEATURES].copy()
    y = sub[target].astype(float)
    groups = sub["school_id"]

    # -------- Temporal split --------
    # Train: school years 2016-2023 (8 years)
    # Test:  school years 2024-2025 (2 years, most recent)
    # This simulates: "predict next year's performance using past data"
    train_mask = sub["year"] <= 2023
    test_mask  = sub["year"] >= 2024
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    groups_train = groups[train_mask]
    print(f"  Train: {len(X_train):,} rows (2016–2023)  |  Test: {len(X_test):,} rows (2024–2025)")

    best_r2 = -np.inf
    all_perm_imp[target] = {}

    for model_name, factory in MODEL_FACTORIES.items():
        print(f"\n  --- {model_name} ---")
        model = factory()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        # Metrics
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
        mae_test  = mean_absolute_error(y_test, y_pred)
        r2_test   = r2_score(y_test, y_pred)
        r2_train  = r2_score(y_train, y_pred_train)

        # GroupKFold CV (grouped by school_id to prevent leakage)
        cv_model = factory()
        gkf = GroupKFold(n_splits=5)
        cv_r2 = cross_val_score(
            cv_model, X, y, groups=groups,
            cv=gkf, scoring="r2", n_jobs=-1
        )

        print(f"    Train R²:     {r2_train:.3f}")
        print(f"    Test  R²:     {r2_test:.3f}")
        print(f"    Test  RMSE:   {rmse_test:.2f}")
        print(f"    Test  MAE:    {mae_test:.2f}")
        print(f"    CV R² (mean): {cv_r2.mean():.3f} ± {cv_r2.std():.3f}")

        # Permutation importance on test set
        if model_name in ("GradientBoosting", "LightGBM"):
            perm_result = permutation_importance(
                model, X_test, y_test, n_repeats=20,
                random_state=42, n_jobs=-1, scoring="r2"
            )
        else:
            perm_result = permutation_importance(
                model, X_test, y_test, n_repeats=20,
                random_state=42, n_jobs=-1, scoring="r2"
            )

        perm_df = pd.DataFrame({
            "feature": FEATURES,
            "importance_mean": perm_result.importances_mean,
            "importance_std": perm_result.importances_std,
            "group": [feat_to_group.get(f, "Other") for f in FEATURES],
        }).sort_values("importance_mean", ascending=False)
        all_perm_imp[target][model_name] = perm_df

        all_results.append({
            "target": target, "model": model_name,
            "train_r2": r2_train, "test_r2": r2_test,
            "test_rmse": rmse_test, "test_mae": mae_test,
            "cv_r2_mean": cv_r2.mean(), "cv_r2_std": cv_r2.std(),
        })

        if r2_test > best_r2:
            best_r2 = r2_test
            best_model_name = model_name
            best_model = model

        # Store LightGBM for SHAP
        if model_name == "LightGBM":
            lgb_models[target] = {
                "model": model, "X_train": X_train, "X_test": X_test,
                "y_test": y_test,
            }

    print(f"\n  ★ Best model for {target}: {best_model_name} (test R² = {best_r2:.3f})")
    best_models[target] = {
        "model_name": best_model_name, "model": best_model,
        "X_train": X_train, "X_test": X_test, "y_test": y_test,
    }

# ======================================================================
# 4. SHAP analysis (from LightGBM)
# ======================================================================
print(f"\n{'='*70}")
print("SHAP ANALYSIS (LightGBM)")
print("=" * 70)

all_shap_vals = {}
for target, lgb_info in lgb_models.items():
    print(f"\n  Computing SHAP for {target}...")
    model = lgb_info["model"]
    X_test = lgb_info["X_test"]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    all_shap_vals[target] = shap_values
    print(f"    ✓ SHAP values computed ({X_test.shape[0]} samples × {X_test.shape[1]} features)")

# ======================================================================
# 5. Plots
# ======================================================================
print(f"\n{'='*70}")
print("GENERATING PLOTS")
print("=" * 70)

GROUP_COLORS = {
    "Demographics":  "#4C78A8",
    "School Ops":    "#F58518",
    "Teacher":       "#E45756",
    "Neighborhood":  "#72B7B2",
    "Other":         "#999999",
}

# 5a. Model comparison bar chart (4 models × 2 targets)
results_df = pd.DataFrame(all_results)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for i, target in enumerate(TARGETS.keys()):
    sub = results_df[results_df["target"] == target]
    ax = axes[i]
    x_pos = np.arange(len(sub))
    colors = ["#4C78A8", "#72B7B2", "#F58518", "#54A24B"]
    bars = ax.bar(x_pos, sub["test_r2"], color=colors[:len(sub)], width=0.6, edgecolor="white")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sub["model"], fontsize=10, rotation=15)
    ax.set_ylabel("Test R²", fontsize=12)
    ax.set_title(TARGETS[target]["label"], fontsize=14, fontweight="bold")
    ax.set_ylim(bottom=min(0, sub["test_r2"].min() - 0.05),
                top=max(sub["test_r2"].max() * 1.3, 0.1))
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    for bar, val in zip(bars, sub["test_r2"]):
        y_pos = bar.get_height() + 0.01 if val >= 0 else bar.get_height() - 0.03
        ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                f"{val:.3f}", ha="center", va="bottom" if val >= 0 else "top",
                fontsize=11, fontweight="bold")

fig.suptitle("Model Comparison — Test R² (Train 2016–2023, Test 2024–2025)",
             fontsize=15, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "model_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  → model_comparison.png")

# 5b. SHAP beeswarm plots (from LightGBM)
for target, sv in all_shap_vals.items():
    fig = plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(sv, max_display=15, show=False)
    plt.title(f"SHAP Summary — {TARGETS[target]['label']}\n(LightGBM)",
              fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"shap_beeswarm_{target}.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → shap_beeswarm_{target}.png")

# 5c. SHAP bar chart
for target, sv in all_shap_vals.items():
    fig = plt.figure(figsize=(10, 7))
    shap.plots.bar(sv, max_display=15, show=False)
    plt.title(f"Feature Importance (mean |SHAP|) — {TARGETS[target]['label']}\n(LightGBM)",
              fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"shap_bar_{target}.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → shap_bar_{target}.png")

# 5d. SHAP dependence plots for top 3 features
for target, sv in all_shap_vals.items():
    mean_abs = np.abs(sv.values).mean(axis=0)
    top3_idx = np.argsort(mean_abs)[-3:][::-1]
    top3_feats = [FEATURES[i] for i in top3_idx]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for j, (feat, feat_idx) in enumerate(zip(top3_feats, top3_idx)):
        ax = axes[j]
        shap.plots.scatter(sv[:, feat_idx], ax=ax, show=False)
        label = FEAT_LABELS.get(feat, feat)
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel(f"SHAP value" if j == 0 else "", fontsize=11)
        ax.set_title(label, fontsize=12, fontweight="bold")

    fig.suptitle(f"SHAP Dependence — {TARGETS[target]['label']} (LightGBM)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"shap_dep_{target}.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → shap_dep_{target}.png")

# 5e. Permutation importance (best model per target)
for target, info in best_models.items():
    model_name = info["model_name"]
    perm_df = all_perm_imp[target][model_name].head(15).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 7))
    bar_colors = [GROUP_COLORS.get(g, "#999") for g in perm_df["group"]]
    labels = [FEAT_LABELS.get(f, f) for f in perm_df["feature"]]
    ax.barh(range(len(perm_df)), perm_df["importance_mean"],
            xerr=perm_df["importance_std"], color=bar_colors,
            edgecolor="white", height=0.7)
    ax.set_yticks(range(len(perm_df)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Permutation Importance (ΔR²)", fontsize=12)
    ax.set_title(f"Feature Importance — {TARGETS[target]['label']}\n({model_name}, permutation-based)",
                 fontsize=14, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=g) for g, c in GROUP_COLORS.items()
                       if g in perm_df["group"].values]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"perm_importance_{target}.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → perm_importance_{target}.png")

# 5f. PDP + ICE for top 3 features (from best model)
for target, info in best_models.items():
    model_name = info["model_name"]
    model = info["model"]
    X_train = info["X_train"]
    perm_df = all_perm_imp[target][model_name]
    top3 = perm_df.head(3)["feature"].tolist()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for j, feat in enumerate(top3):
        feat_idx = FEATURES.index(feat)
        try:
            PartialDependenceDisplay.from_estimator(
                model, X_train, [feat_idx],
                kind="both", subsample=100, ax=axes[j], random_state=42,
                ice_lines_kw={"color": "#4C78A8", "alpha": 0.05, "linewidth": 0.5},
                pd_line_kw={"color": "#E45756", "linewidth": 3},
            )
        except Exception as e:
            PartialDependenceDisplay.from_estimator(
                model, X_train, [feat_idx],
                kind="average", ax=axes[j], random_state=42,
                pd_line_kw={"color": "#E45756", "linewidth": 3},
            )
        label = FEAT_LABELS.get(feat, feat)
        axes[j].set_title(label, fontsize=12, fontweight="bold")
        axes[j].set_ylabel(f"Partial Dependence" if j == 0 else "")

    fig.suptitle(f"PDP + ICE — {TARGETS[target]['label']} ({model_name})",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"pdp_ice_{target}.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → pdp_ice_{target}.png")

# 5g. Lasso coefficients
for target in TARGETS:
    lasso_model = make_lasso()
    mask_t = df[target].notna()
    for yr in TARGETS[target]["exclude_years"]:
        mask_t &= (df["year"] != yr)
    sub_t = df.loc[mask_t]
    lasso_model.fit(sub_t[FEATURES], sub_t[target].astype(float))
    coefs = lasso_model.named_steps["lasso"].coef_

    coef_df = pd.DataFrame({
        "feature": FEATURES, "coefficient": coefs,
        "group": [feat_to_group.get(f, "Other") for f in FEATURES],
    })
    coef_df["abs_coef"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [GROUP_COLORS.get(g, "#999") for g in coef_df["group"]]
    labels = [FEAT_LABELS.get(f, f) for f in coef_df["feature"]]
    ax.barh(range(len(coef_df)), coef_df["coefficient"], color=colors,
            edgecolor="white", height=0.7)
    ax.set_yticks(range(len(coef_df)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Standardized Coefficient (β)", fontsize=12)
    ax.set_title(f"Lasso Coefficients — {TARGETS[target]['label']}\n(standardized, top 15 by |β|)",
                 fontsize=14, fontweight="bold")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=g) for g, c in GROUP_COLORS.items()
                       if g in coef_df["group"].values]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"lasso_coef_{target}.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → lasso_coef_{target}.png")

# ======================================================================
# 6. Save CSV + report
# ======================================================================
print(f"\n{'='*70}")
print("SAVING OUTPUTS")
print("=" * 70)

# Feature importance CSV (SHAP-based from LightGBM)
shap_imp_rows = []
for target, sv in all_shap_vals.items():
    mean_abs = np.abs(sv.values).mean(axis=0)
    for i, feat in enumerate(FEATURES):
        shap_imp_rows.append({
            "target": target, "feature": feat,
            "group": feat_to_group.get(feat, "Other"),
            "mean_abs_shap": mean_abs[i],
        })
shap_imp_df = pd.DataFrame(shap_imp_rows).sort_values(
    ["target", "mean_abs_shap"], ascending=[True, False])
shap_imp_df.to_csv(os.path.join(OUT, "feature_importance.csv"), index=False)
print(f"  → feature_importance.csv")

# Model results report
lines = []
lines.append(f"# Baseline Model Results — Cook County High Schools")
lines.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
lines.append("## Model Comparison\n")
lines.append("| Target | Model | Train R² | Test R² | Test RMSE | Test MAE | CV R² (5-fold) |")
lines.append("|--------|-------|----------|---------|-----------|---------|----------------|")
for _, r in results_df.iterrows():
    lines.append(
        f"| {r['target']} | {r['model']} | {r['train_r2']:.3f} | "
        f"{r['test_r2']:.3f} | {r['test_rmse']:.2f} | {r['test_mae']:.2f} | "
        f"{r['cv_r2_mean']:.3f} ± {r['cv_r2_std']:.3f} |"
    )
lines.append("")

lines.append("## Best Models\n")
for target in TARGETS:
    sub = results_df[results_df["target"] == target]
    best = sub.loc[sub["test_r2"].idxmax()]
    lines.append(f"- **{target}**: {best['model']} (Test R² = {best['test_r2']:.3f})")
lines.append("")

lines.append("## Top 10 Features by Mean |SHAP| (LightGBM)\n")
for target in TARGETS:
    lines.append(f"### {target}\n")
    sub_imp = shap_imp_df[shap_imp_df["target"] == target].head(10)
    lines.append("| Rank | Feature | Group | Mean |SHAP| |")
    lines.append("|------|---------|-------|-------------|")
    for rank, (_, row) in enumerate(sub_imp.iterrows(), 1):
        label = FEAT_LABELS.get(row["feature"], row["feature"])
        lines.append(f"| {rank} | {label} | {row['group']} | {row['mean_abs_shap']:.3f} |")
    lines.append("")

lines.append("## Plots\n")
lines.append("### Model Comparison\n![Model Comparison](model_comparison.png)\n")
for target in TARGETS:
    tl = TARGETS[target]["label"]
    mn = best_models[target]["model_name"]
    lines.append(f"### {tl}\n")
    lines.append(f"![SHAP Beeswarm](shap_beeswarm_{target}.png)\n")
    lines.append(f"![SHAP Bar](shap_bar_{target}.png)\n")
    lines.append(f"![SHAP Dependence](shap_dep_{target}.png)\n")
    lines.append(f"![Permutation Importance](perm_importance_{target}.png)\n")
    lines.append(f"![PDP + ICE](pdp_ice_{target}.png)\n")
    lines.append(f"![Lasso Coefficients](lasso_coef_{target}.png)\n")

lines.append("## Methodology\n")
lines.append("### Train/Test Split\n")
lines.append("- **Temporal split**: Train = school years 2016–2023, Test = 2024–2025")
lines.append("- Rationale: simulates real-world use case — predict future school performance from historical data")
lines.append("- This is a **time-based out-of-sample** test, which is stricter and more realistic than random splits for panel data\n")
lines.append("### Cross-Validation\n")
lines.append("- 5-fold **GroupKFold** grouped by `school_id`")
lines.append("- Prevents data leakage: the same school never appears in both train and validation fold")
lines.append("- This controls for school-level autocorrelation\n")
lines.append("### Models\n")
lines.append("- **Lasso**: L1-regularized linear regression (median imputation + standardization)")
lines.append("- **RandomForest**: 300 trees, max_depth=12 (median imputation)")
lines.append("- **GradientBoosting**: sklearn HistGradientBoostingRegressor (handles NaN natively)")
lines.append("- **LightGBM**: gradient boosting (handles NaN natively, SHAP source)\n")
lines.append("### Interpretability\n")
lines.append("- **SHAP** (from LightGBM): TreeExplainer gives exact Shapley values")
lines.append("- **Permutation Importance**: model-agnostic, from best model")
lines.append("- **PDP/ICE**: shows marginal effect of top features")
lines.append("- **Lasso coefficients**: standardized β gives linear effect direction + magnitude")

report = "\n".join(lines)
with open(os.path.join(OUT, "model_results.md"), "w") as f:
    f.write(report)
print(f"  → model_results.md")

# ======================================================================
print(f"\n{'='*70}")
print("PIPELINE COMPLETE")
print("=" * 70)
print(f"\nOutputs saved to {OUT}/")
for fn in sorted(os.listdir(OUT)):
    print(f"  - {fn}")
