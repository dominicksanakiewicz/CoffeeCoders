import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, GroupKFold
import statsmodels.api as sm

path = "C:/Users/amand/student30538-w26/CoffeeCoders"
os.chdir(path)
data = pd.read_csv('panel_yx_highschools.csv')
data_clean = data.drop(columns=['school_id', 'year', 'district', 'county', 'school_type', 'grades_served', 'TRACTID', 'LAT', 'LON', 'acs_end_year'])
data_clean.dtypes
len(data_clean)
data_clean = data_clean.fillna(0)

def run_panel_elasticnet(data, y_var, group_var='school_name', 
                         alpha_range=None, l1_ratio_range=None, n_splits=5, random_state=1):

    if alpha_range is None:
        alpha_range = np.logspace(-4, 0, 50)
    if l1_ratio_range is None:
        l1_ratio_range = np.linspace(0.1, 0.9, 9)
    

    X = data.drop(columns=[y_var, group_var])
    y = data[y_var]
    feature_names = X.columns
    
    groups = data[group_var]
    
    gkf = GroupKFold(n_splits=n_splits)
    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train, groups_test = groups.iloc[train_idx], groups.iloc[test_idx]
        break  

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    param_grid = {
        'elasticnet__alpha': alpha_range,
        'elasticnet__l1_ratio': l1_ratio_range
    }
    
    pipeline = make_pipeline(
        StandardScaler(),
        ElasticNet(random_state=random_state, max_iter=10000)
    )
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=gkf.split(X_train, y_train, groups_train),
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    enet_step = best_model.named_steps['elasticnet']
    
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    
    coef_original = enet_step.coef_ / scaler.scale_
    
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient_scaled': enet_step.coef_,
        'coefficient_original_units': coef_original
    })
    coef_df['abs_coef'] = coef_df['coefficient_original_units'].abs()
    coef_df = coef_df.sort_values(by='abs_coef', ascending=False)
    
    results = {
        'best_alpha': grid_search.best_params_['elasticnet__alpha'],
        'best_l1_ratio': grid_search.best_params_['elasticnet__l1_ratio'],
        'best_cv_mse': -grid_search.best_score_,
        'train_mse': mse_train,
        'test_mse': mse_test,
        'coef_df': coef_df
    }
    
    return results


##############################################

def run_post_elasticnet_ols(
    data,
    elasticnet_res,
    y_var,
    top_k=10,
    use_nonzero=False,
    cluster_var=None
):

    coef_df = elasticnet_res['coef_df'].copy()

    # Choose features
    if use_nonzero:
        selected = coef_df[coef_df['coefficient_scaled'] != 0]
    else:
        selected = coef_df.head(top_k)

    selected_features = selected['feature'].tolist()

    # Prepare regression data
    X = data[selected_features]
    y = data[y_var]

    X = sm.add_constant(X)

    # Fit model
    if cluster_var is not None:
        model = sm.OLS(y, X).fit(
            cov_type='cluster',
            cov_kwds={'groups': data[cluster_var]}
        )
    else:
        model = sm.OLS(y, X).fit()

    return model, selected_features

#############################################################
res = run_panel_elasticnet(data_clean, y_var='y_ela_prof', group_var='school_name')

print(f"Best alpha: {res['best_alpha']}")
print(f"Best l1_ratio: {res['best_l1_ratio']}")
print(f"Train MSE: {res['train_mse']:.2f}")
print(f"Test MSE: {res['test_mse']:.2f}")

print("\nTop coefficients:")
print(res['coef_df'].head(10))

model, features = run_post_elasticnet_ols(
    data_clean,
    res,
    y_var='y_ela_prof',
    use_nonzero=True,
    cluster_var='school_name'
)

print(model.summary())

###############################################
res = run_panel_elasticnet(data_clean, y_var='y_grad_4yr', group_var='school_name')

print(f"Best alpha: {res['best_alpha']}")
print(f"Best l1_ratio: {res['best_l1_ratio']}")
print(f"Train MSE: {res['train_mse']:.2f}")
print(f"Test MSE: {res['test_mse']:.2f}")

print("\nTop coefficients:")
print(res['coef_df'].head(10))

model, features = run_post_elasticnet_ols(
    data_clean,
    res,
    y_var='y_grad_4yr',
    use_nonzero=True,
    cluster_var='school_name'
)

print(model.summary())


##############################################

res = run_panel_elasticnet(data_clean, y_var='y_math_prof', group_var='school_name')

print(f"Best alpha: {res['best_alpha']}")
print(f"Best l1_ratio: {res['best_l1_ratio']}")
print(f"Train MSE: {res['train_mse']:.2f}")
print(f"Test MSE: {res['test_mse']:.2f}")

print("\nTop coefficients:")
print(res['coef_df'].head(10))

model, features = run_post_elasticnet_ols(
    data_clean,
    res,
    y_var='y_math_prof',
    use_nonzero=True,
    cluster_var='school_name'
)

print(model.summary())

##############################################
res = run_panel_elasticnet(data_clean, y_var='x_dropout_rate', group_var='school_name')

print(f"Best alpha: {res['best_alpha']}")
print(f"Best l1_ratio: {res['best_l1_ratio']}")
print(f"Train MSE: {res['train_mse']:.2f}")
print(f"Test MSE: {res['test_mse']:.2f}")

print("\nTop coefficients:")
print(res['coef_df'].head(10))

model, features = run_post_elasticnet_ols(
    data_clean,
    res,
    y_var='x_dropout_rate',
    use_nonzero=True,
    cluster_var='school_name'
)

print(model.summary())
######################################################
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import ElasticNet

def plot_elasticnet_paths(X, y, feature_names=None, l1_ratio=0.9, top_n=None):
    """
    Plots Elastic Net coefficient paths across a range of alphas.
    
    Parameters
    ----------
    X : pd.DataFrame or np.array
        Predictor matrix (exclude y)
    y : pd.Series or np.array
        Outcome variable
    feature_names : list of str, optional
        Names of features. Default: X.columns if X is DataFrame
    l1_ratio : float
        Elastic Net l1_ratio (0=L2/Ridge, 1=L1/Lasso)
    top_n : int or None
        If specified, only highlights top_n largest coefficients at smallest alpha
    """

    # Feature names
    if feature_names is None and hasattr(X, 'columns'):
        feature_names = X.columns.tolist()
    elif feature_names is None:
        feature_names = [f"x{i}" for i in range(X.shape[1])]

    # Alpha grid
    alphas = np.logspace(-4, 0, 50)

    # Fit Elastic Net across alphas
    coefs = []
    for a in alphas:
        enet = ElasticNet(alpha=a, l1_ratio=l1_ratio, max_iter=10000)
        enet.fit(X, y)
        coefs.append(enet.coef_)
    coefs = np.array(coefs)

    # If top_n specified, select top_n variables by absolute coefficient at smallest alpha
    if top_n is not None:
        final_coefs = np.abs(coefs[0, :])
        top_idx = np.argsort(final_coefs)[-top_n:]
    else:
        top_idx = range(coefs.shape[1])

    # Plot
    plt.figure(figsize=(10,6))
    for i in range(coefs.shape[1]):
        if i in top_idx:
            plt.plot(alphas, coefs[:, i], linewidth=2, label=feature_names[i])
        else:
            plt.plot(alphas, coefs[:, i], color='grey', alpha=0.3)

    plt.xscale('log')
    plt.xlabel('Alpha (log scale)')
    plt.ylabel('Coefficient value')
    plt.title(f'Elastic Net Coefficient Paths (l1_ratio={l1_ratio})')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    if top_n is not None:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

################################################################
X = data_clean.drop(columns=['y_math_prof', 'school_name'])
y = data_clean['y_math_prof']

# Call the function
plot_elasticnet_paths(X, y, l1_ratio=res['best_l1_ratio'], top_n=10)

###############################################################

X = data_clean.drop(columns=['y_ela_prof', 'school_name'])
y = data_clean['y_ela_prof']

# Call the function
plot_elasticnet_paths(X, y, l1_ratio=res['best_l1_ratio'], top_n=10)

###############################################################

X = data_clean.drop(columns=['y_grad_4yr', 'school_name'])
y = data_clean['y_grad_4yr']

# Call the function
plot_elasticnet_paths(X, y, l1_ratio=res['best_l1_ratio'], top_n=10)

###############################################################

X = data_clean.drop(columns=['x_dropout_rate', 'school_name'])
y = data_clean['x_dropout_rate']

# Call the function
plot_elasticnet_paths(X, y, l1_ratio=res['best_l1_ratio'], top_n=10)
