"""
ElasticNet Pipeline — Cook County High Schools
================================================
Trains one ElasticNet model per target column, runs post-selection OLS
for inference, and generates coefficient-path plots.

Exports:
  outputs/elasticnet_coefficients.json   → consumed by app.py (Streamlit)
  outputs/coef_paths_<target>.png        → coefficient-path plots

Run:  python ml_pipeline.py              (inside conda env dap)
"""

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "panel_yx_highschools.csv"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Targets to model ────────────────────────────────────────────────
# Add extra non-y_ targets here if needed (e.g. ["x_dropout_rate"]).
EXTRA_TARGETS = []


# ═════════════════════════════════════════════════════════════════════
# FUNCTIONS
# ═════════════════════════════════════════════════════════════════════

def run_panel_elasticnet(
    df,
    y_var,
    feature_cols,
    group_var="school_name",
    alpha_range=None,
    l1_ratio_range=None,
    n_splits=5,
    random_state=1,
):
    """Train an ElasticNet for *y_var* using only *feature_cols* as X.

    Returns a dict with model diagnostics, coefficients (both scaled
    and original-unit), plus scaler parameters.
    """
    if alpha_range is None:
        alpha_range = np.logspace(-4, 0, 50)
    if l1_ratio_range is None:
        l1_ratio_range = np.linspace(0.1, 0.9, 9)

    subset = df.dropna(subset=[y_var])
    X = subset[feature_cols]
    y = subset[y_var]
    groups = subset[group_var]

    # Single outer train/test split via GroupKFold
    gkf = GroupKFold(n_splits=n_splits)
    train_idx, test_idx = next(gkf.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]

    # Pipeline: Impute → Scale → ElasticNet
    pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        ElasticNet(random_state=random_state, max_iter=10000),
    )

    param_grid = {
        "elasticnet__alpha": alpha_range,
        "elasticnet__l1_ratio": l1_ratio_range,
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=GroupKFold(n_splits=n_splits).split(X_train, y_train, groups_train),
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    scaler_step = best_model.named_steps["standardscaler"]
    enet_step = best_model.named_steps["elasticnet"]

    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    coef_scaled = enet_step.coef_
    coef_original = enet_step.coef_ / scaler_step.scale_

    coef_df = pd.DataFrame({
        "feature": feature_cols,
        "coefficient_scaled": coef_scaled,
        "abs_scaled": np.abs(coef_scaled),
        "coefficient_original_units": coef_original,
    }).sort_values("abs_scaled", ascending=False)

    return {
        "best_alpha": grid_search.best_params_["elasticnet__alpha"],
        "best_l1_ratio": grid_search.best_params_["elasticnet__l1_ratio"],
        "best_cv_mse": -grid_search.best_score_,
        "train_mse": mean_squared_error(y_train, y_pred_train),
        "test_mse": mean_squared_error(y_test, y_pred_test),
        "r2_test": r2_score(y_test, y_pred_test),
        "intercept": float(enet_step.intercept_),
        "scaler_mean": scaler_step.mean_.tolist(),
        "scaler_scale": scaler_step.scale_.tolist(),
        "coef_df": coef_df,
    }


def run_post_elasticnet_ols(
    df,
    elasticnet_res,
    y_var,
    top_k=10,
    use_nonzero=False,
    cluster_var=None,
):
    """Re-fit OLS on the features selected by ElasticNet for inference
    (p-values, confidence intervals, clustered SEs)."""
    coef_df = elasticnet_res["coef_df"].copy()

    if use_nonzero:
        selected = coef_df[coef_df["coefficient_scaled"] != 0]
    else:
        selected = coef_df.head(top_k)

    selected_features = selected["feature"].tolist()

    sub = df.dropna(subset=[y_var] + selected_features)
    X = sm.add_constant(sub[selected_features])
    y = sub[y_var]

    if cluster_var is not None:
        model = sm.OLS(y, X).fit(
            cov_type="cluster",
            cov_kwds={"groups": sub[cluster_var]},
        )
    else:
        model = sm.OLS(y, X).fit()

    return model, selected_features


def plot_elasticnet_paths(
    X, y, feature_names=None, l1_ratio=0.9, top_n=10, title=None, save_path=None,
):
    """Plot how ElasticNet coefficients change across a range of alphas."""
    if feature_names is None and hasattr(X, "columns"):
        feature_names = X.columns.tolist()
    elif feature_names is None:
        feature_names = [f"x{i}" for i in range(X.shape[1])]

    alphas = np.logspace(-4, 0, 50)
    coefs = []
    for a in alphas:
        enet = ElasticNet(alpha=a, l1_ratio=l1_ratio, max_iter=10000)
        enet.fit(X, y)
        coefs.append(enet.coef_)
    coefs = np.array(coefs)

    # Highlight only top_n variables (by |coef| at smallest alpha)
    if top_n is not None:
        top_idx = set(np.argsort(np.abs(coefs[0, :]))[-top_n:])
    else:
        top_idx = set(range(coefs.shape[1]))

    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(coefs.shape[1]):
        if i in top_idx:
            ax.plot(alphas, coefs[:, i], linewidth=2, label=feature_names[i])
        else:
            ax.plot(alphas, coefs[:, i], color="grey", alpha=0.3)

    ax.set_xscale("log")
    ax.set_xlabel("Alpha (log scale)")
    ax.set_ylabel("Coefficient value")
    ax.set_title(title or f"Elastic Net Coefficient Paths (l1_ratio={l1_ratio})")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    if top_n is not None:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved → {save_path}")
    plt.close(fig)


def get_feature_cols(x_cols, target):
    """Return feature columns, excluding the target if it's an x_ column."""
    return [c for c in x_cols if c != target]


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Load data ────────────────────────────────────────────────────
    data = pd.read_csv(DATA_PATH)

    y_cols = sorted([c for c in data.columns if c.startswith("y_")])
    x_cols = sorted([c for c in data.columns if c.startswith("x_")])

    all_targets = y_cols + [t for t in EXTRA_TARGETS if t in data.columns]

    print(f"Targets  ({len(all_targets)}): {all_targets}")
    print(f"Features ({len(x_cols)}): {x_cols}")

    # ── Loop over targets ────────────────────────────────────────────
    dashboard_data = {}

    for target in all_targets:
        non_null = data[target].notna().sum()
        if non_null < 100:
            print(f"\nSKIPPING {target} (only {non_null} non-null rows)")
            continue

        print(f"\n{'='*60}")
        print(f"TARGET: {target}  ({non_null} observations)")
        print(f"{'='*60}")

        # Feature set: all x_ columns, minus the target if it's x_
        feat_cols = get_feature_cols(x_cols, target)

        # ── 1. ElasticNet ────────────────────────────────────────────
        res = run_panel_elasticnet(data, y_var=target, feature_cols=feat_cols)

        print(f"  Best α={res['best_alpha']:.6f}  L1={res['best_l1_ratio']:.2f}")
        print(f"  Train MSE={res['train_mse']:.2f}  "
              f"Test MSE={res['test_mse']:.2f}  R²={res['r2_test']:.3f}")
        print("  Top 5 (by |standardised coef|):")
        top5 = res["coef_df"].head(5)
        for _, r in top5.iterrows():
            print(f"    {r['feature']:40s}  scaled={r['coefficient_scaled']:+8.4f}  "
                  f"raw={r['coefficient_original_units']:+10.4f}")

        # ── 2. Post-selection OLS ────────────────────────────────────
        ols_model, ols_features = run_post_elasticnet_ols(
            data, res, y_var=target, use_nonzero=True, cluster_var="school_name",
        )
        print(ols_model.summary())

        # ── 3. Coefficient-path plot ─────────────────────────────────
        sub = data.dropna(subset=[target])
        # Impute X for plotting (same strategy as pipeline)
        X_plot = pd.DataFrame(
            SimpleImputer(strategy="median").fit_transform(sub[feat_cols]),
            columns=feat_cols,
        )
        y_plot = sub[target].values
        plot_elasticnet_paths(
            X_plot, y_plot,
            feature_names=feat_cols,
            l1_ratio=res["best_l1_ratio"],
            top_n=10,
            title=f"Coefficient Paths — {target} (l1_ratio={res['best_l1_ratio']:.2f})",
            save_path=OUTPUT_DIR / f"coef_paths_{target}.png",
        )

        # ── 4. Serialise for the dashboard ───────────────────────────
        cdf = res["coef_df"]
        dashboard_data[target] = {
            "intercept": res["intercept"],
            "train_mse": res["train_mse"],
            "test_mse": res["test_mse"],
            "r2_test": res["r2_test"],
            "best_alpha": res["best_alpha"],
            "best_l1_ratio": res["best_l1_ratio"],
            "scaler_mean": dict(zip(feat_cols, res["scaler_mean"])),
            "scaler_scale": dict(zip(feat_cols, res["scaler_scale"])),
            "coefficients": cdf[
                ["feature", "coefficient_scaled", "coefficient_original_units"]
            ].to_dict(orient="records"),
        }

    # ── Export JSON ──────────────────────────────────────────────────
    out_path = OUTPUT_DIR / "elasticnet_coefficients.json"
    with open(out_path, "w") as f:
        json.dump(dashboard_data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"All results saved → {out_path}")
    print(f"Targets processed: {list(dashboard_data.keys())}")
