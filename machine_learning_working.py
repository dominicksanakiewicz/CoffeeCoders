import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ─────────────────────────────────────────────────────────────
path = "C:/Users/amand/student30538-w26/CoffeeCoders"
os.chdir(path)

DATA_PATH = Path(path) / "panel_yx_highschools.csv"
OUTPUT_DIR = Path(path) / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

data = pd.read_csv(DATA_PATH)

# ── Identify column groups ─────────────────────────────────────────────
y_cols = sorted([c for c in data.columns if c.startswith("y_")])
x_cols = sorted([c for c in data.columns if c.startswith("x_")])

print(f"Targets ({len(y_cols)}): {y_cols}")
print(f"Features ({len(x_cols)}): {x_cols}")

# ── Optional custom transformer to skip zeros if needed ────────────────
class SkipZeroImputer(BaseEstimator, TransformerMixin):
    """Convert zeros to NaN so that they are ignored by imputer/scaler."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_ = X.copy()
        X_[X_ == 0] = np.nan
        return X_

# ── ElasticNet with GroupKFold ─────────────────────────────────────────
def run_panel_elasticnet(df, y_var, feature_cols, group_var="school_name",
                         alpha_range=None, l1_ratio_range=None, n_splits=5, random_state=1):

    if alpha_range is None:
        alpha_range = np.logspace(-4, 0, 50)
    if l1_ratio_range is None:
        l1_ratio_range = np.linspace(0.1, 0.9, 9)

    # Drop rows where y is missing
    subset = df.dropna(subset=[y_var])
    X = subset[feature_cols].copy()
    y = subset[y_var].copy()
    groups = subset[group_var].copy()

    # Ensure numeric
    X = X.apply(pd.to_numeric, errors='coerce')

    # Outer train/test split
    gkf = GroupKFold(n_splits=n_splits)
    train_idx, test_idx = next(gkf.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]

    # Pipeline: Imputer -> Scaler -> ElasticNet
    pipeline = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler(),
        ElasticNet(random_state=random_state, max_iter=10000)
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
    coef_original = coef_scaled / scaler_step.scale_

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

# ── Post-ElasticNet OLS ─────────────────────────────────────────────────
def run_post_elasticnet_ols(data, elasticnet_res, y_var, top_k=10, use_nonzero=False, cluster_var=None):
    coef_df = elasticnet_res['coef_df'].copy()
    selected = coef_df[coef_df['coefficient_scaled'] != 0] if use_nonzero else coef_df.head(top_k)
    selected_features = selected['feature'].tolist()

    # Use only rows where target is not NaN
    subset = data.dropna(subset=[y_var])
    X = subset[selected_features].copy()
    X = X.fillna(X.median())  # fill missing values same as ElasticNet
    X = sm.add_constant(X)
    y = subset[y_var]

    model = sm.OLS(y, X).fit(
        cov_type='cluster',
        cov_kwds={'groups': subset[cluster_var]} if cluster_var else None
    )
    return model, selected_features

# ── Run ElasticNet for all y_ targets ───────────────────────────────────
dashboard_data = {}
for target in y_cols:
    non_null = data[target].notna().sum()
    if non_null < 100:
        print(f"SKIPPING {target} ({non_null} non-null rows)")
        continue

    print(f"\n{'='*60}\nTARGET: {target}  ({non_null} observations)\n{'='*60}")
    res = run_panel_elasticnet(data, y_var=target, feature_cols=x_cols)

    print(f"Best α={res['best_alpha']:.6f}  L1={res['best_l1_ratio']:.2f}")
    print(f"Train MSE={res['train_mse']:.2f}  Test MSE={res['test_mse']:.2f}  R²={res['r2_test']:.3f}")
    print("Top 5 coefficients (by |scaled|):")
    print(res['coef_df'].head(5))

    dashboard_data[target] = {
        "intercept": res["intercept"],
        "train_mse": res["train_mse"],
        "test_mse": res["test_mse"],
        "r2_test": res["r2_test"],
        "best_alpha": res["best_alpha"],
        "best_l1_ratio": res["best_l1_ratio"],
        "scaler_mean": dict(zip(x_cols, res["scaler_mean"])),
        "scaler_scale": dict(zip(x_cols, res["scaler_scale"])),
        "coefficients": res["coef_df"][["feature", "coefficient_scaled", "coefficient_original_units"]].to_dict(orient="records"),
    }

    model, features = run_post_elasticnet_ols(data, res, y_var=target, use_nonzero=True, cluster_var='school_name')
    print(model.summary())

# ── Save dashboard JSON ────────────────────────────────────────────────
out_path = OUTPUT_DIR / "elasticnet_coefficients.json"
with open(out_path, "w") as f:
    json.dump(dashboard_data, f, indent=2)

print(f"\nAll results saved → {out_path}")

# ── Optional: ElasticNet coefficient paths ─────────────────────────────
def plot_elasticnet_paths(X, y, feature_names=None, l1_ratio=0.9, top_n=None):
    if feature_names is None and hasattr(X, 'columns'):
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

    top_idx = range(coefs.shape[1])
    if top_n is not None:
        top_idx = np.argsort(np.abs(coefs[-1, :]))[-top_n:]

    plt.figure(figsize=(10,6))
    for i in range(coefs.shape[1]):
        plt.plot(alphas, coefs[:, i], color='grey', alpha=0.3 if i not in top_idx else 1.0, linewidth=2 if i in top_idx else 1)
    for i in top_idx:
        plt.plot(alphas, coefs[:, i], label=feature_names[i], linewidth=2)
    plt.xscale('log')
    plt.xlabel('Alpha (log scale)')
    plt.ylabel('Coefficient value')
    plt.title(f'Elastic Net Coefficient Paths (l1_ratio={l1_ratio})')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    if top_n is not None:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

# ── Prepare clean numeric feature dataframe ───────────────────────────
# Only include numeric columns for modeling and plotting
# Keep school_name for grouping
data_clean = data[x_cols + ['school_name']].copy()
# Ensure numeric
for col in x_cols:
    data_clean[col] = pd.to_numeric(data_clean[col], errors='coerce')

# Math proficiency
X = data_clean.drop(columns=['school_name']).copy()
y = data['y_math_prof'].copy()

# Fill missing values in X
X = X.fillna(X.median())

# Optionally drop rows where y is NaN
mask = y.notna()
X = X.loc[mask]
y = y.loc[mask]

fig_math = plot_elasticnet_paths(X, y, l1_ratio=res['best_l1_ratio'], top_n=10)

# Example for ELA
X = data_clean.drop(columns=['school_name']).copy()
y = data['y_ela_prof'].copy()
X = X.fillna(X.median())
mask = y.notna()
X = X.loc[mask]
y = y.loc[mask]
fig_ela = plot_elasticnet_paths(X, y, l1_ratio=res['best_l1_ratio'], top_n=10)

# ── ElasticNet Coefficient Path Plots for All Targets ────────────────

plots = {}  # store figures

targets = ['y_math_prof', 'y_ela_prof', 'y_grad_4yr', 'x_dropout_rate']

for target in targets:
    X = data_clean.drop(columns=['school_name']).copy()
    y = data[target].copy()
    
    # Fill missing values in predictors
    X = X.fillna(X.median())
    
    # Drop rows where target is NaN
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]
    
    # Plot and return figure
    fig = plot_elasticnet_paths(X, y, l1_ratio=res['best_l1_ratio'], top_n=10)
    
    # Store figure in dictionary for later use or saving
    plots[target] = fig
    print(f"ElasticNet coefficient path plot created for {target}")

# ── Example: Save figures if desired ────────────────────────────────
# plots['y_math_prof'].savefig("math_coef_paths.png")
# plots['y_ela_prof'].savefig("ela_coef_paths.png")
# plots['y_grad_4yr'].savefig("grad4yr_coef_paths.png")
# plots['x_dropout_rate'].savefig("dropout_coef_paths.png")

print("\nAll coefficient path plots created and stored in 'plots' dictionary.")

# List of targets
targets = ['y_math_prof', 'y_ela_prof', 'y_grad_4yr', 'x_dropout_rate']

# Dictionary to store results
elasticnet_feature_summary = {}

for target in targets:
    # Assume you ran run_panel_elasticnet for this target and have `res` for it
    # If not, rerun ElasticNet for that target first
    # For example:
    # res = run_panel_elasticnet(data_clean, y_var=target, feature_cols=x_cols, group_var='school_name')
    
    coef_df = res['coef_df']
    
    num_total = len(coef_df)
    num_zero = (coef_df['coefficient_scaled'] == 0).sum()
    num_nonzero = num_total - num_zero
    
    elasticnet_feature_summary[target] = {
        'total_features': num_total,
        'cut_to_zero': num_zero,
        'retained': num_nonzero
    }

# Print nicely
for target, summary in elasticnet_feature_summary.items():
    print(f"{target}: {summary['retained']} retained, {summary['cut_to_zero']} cut to zero, total {summary['total_features']}")