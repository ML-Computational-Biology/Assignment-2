import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif,f_regression
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import (matthews_corrcoef, roc_auc_score, balanced_accuracy_score,
                             f1_score, fbeta_score, recall_score, precision_score,
                             confusion_matrix, average_precision_score)
from optuna.integration import OptunaSearchCV
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# =========================
# GLOBAL CONSTANTS
# =========================

N_ITER = 1000
RANDOM_STATE = 42
CI=0.95



class RepeatedNestedCV:
    def __init__(self,
                 estimators: dict,
                 param_grids: dict,
                 R: int = 10,
                 N: int = 5,
                 K: int = 3,
                 random_state: int = RANDOM_STATE,
                 scoring: str = 'balanced_accuracy'):
        self.estimators = estimators
        self.param_grids = param_grids
        self.R = R
        self.N = N
        self.K = K
        self.random_state = random_state
        self.scoring = scoring

    def evaluate_model(self, y_true, y_pred, y_proba):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return {
            'MCC': matthews_corrcoef(y_true, y_pred),
            'AUC': roc_auc_score(y_true, y_proba),
            'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
        }

    def run(self, X, y):
        results = []
        rskf = RepeatedStratifiedKFold(n_splits=self.N, n_repeats=self.R, random_state=self.random_state)

        for estimator_name, pipeline in self.estimators.items():
            print(f"\nRunning nCV for: {estimator_name}")
            param_grid = self.param_grids[estimator_name]

            for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
                print(f"  Fold {fold_idx + 1}/{self.R * self.N}")
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                inner_cv = StratifiedKFold(n_splits=self.K, shuffle=True, random_state=self.random_state)

                optuna_search = OptunaSearchCV(
                    estimator=pipeline,
                    param_distributions=param_grid,
                    cv=inner_cv,
                    scoring=self.scoring,
                    n_trials=20,
                    random_state=self.random_state,
                    refit=True
                )
                optuna_search.fit(X_train, y_train)

                best_model = optuna_search.best_estimator_
                y_pred = best_model.predict(X_test)
                y_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else y_pred

                metrics = self.evaluate_model(y_test, y_pred, y_proba)
                metrics.update({
                    'Estimator': estimator_name,
                    'Best Params': optuna_search.best_params_,
                    'Fold': fold_idx + 1
                })
                results.append(metrics)

        return pd.DataFrame(results)


def bootstrap_ci(data, func=np.median, n_bootstraps=N_ITER, ci=CI):
    bootstraps = [func(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_bootstraps)]
    lower = np.percentile(bootstraps, (1 - ci) / 2 * 100)
    upper = np.percentile(bootstraps, (1 + ci) / 2 * 100)
    return lower, upper


def summarize_with_ci(df, metrics=None, ci=CI, n_bootstraps=N_ITER):
    if metrics is None:
        metrics = ['MCC', 'AUC', 'Balanced Accuracy']

    summary = {}

    for metric in metrics:
        print(f"\n {metric} Summary with {int(ci*100)}% CI:")

        rows = []
        for model in df['Estimator'].unique():
            values = df[df['Estimator'] == model][metric].values
            median = np.median(values)
            mean = np.mean(values)
            std = np.std(values)
            ci_low, ci_high = bootstrap_ci(values, func=np.median, n_bootstraps=n_bootstraps, ci=ci)
            rows.append({
                'Estimator': model,
                'Median': median,
                'Mean': mean,
                'Std': std,
                'CI Lower': ci_low,
                'CI Upper': ci_high
            })

        metric_df = pd.DataFrame(rows).sort_values(by='Median', ascending=False).set_index('Estimator')
        summary[metric] = metric_df

        print(metric_df)

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='Estimator', y=metric)
        plt.title(f'{metric} Distribution per Estimator')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    return summary



def train_and_save_final_model(X, y, estimators, param_grids, winner,
                               save_path="../models/final_model.pkl",
                               cv_folds=5, scoring='balanced_accuracy'):

    print(f"\n[Step 1] Selecting best hyperparameters for: {winner}")

    pipeline = estimators[winner]
    param_grid = param_grids[winner]
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring=scoring)
    grid_search.fit(X, y)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print(f"\nBest hyperparameters for {winner} on full dataset:")
    print(best_params)

    print("\n[Step 2] Training final model and saving to disk...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(best_model, f)

    print(f" Final model saved to: {save_path}")
    return best_model, best_params




    
############################################################
# BONUS
############################################################



def compare_top_metrics(*dfs, method_names=None, top_metrics=['MCC', 'AUC', 'Balanced Accuracy']):
    """
    Compare top 3 metrics across multiple methods (baseline, FS, etc.).

    This function was developed to facilitate the visual comparison of key evaluation metrics
    (e.g., MCC, AUC, Balanced Accuracy) across multiple model variants or preprocessing strategies.
    Although it was not ultimately used in the final analysis, it remains a useful utility for 
    future experiments involving method benchmarking or comparative performance analysis.

    Parameters:
    - dfs: list of pandas DataFrames (each from a different FS or baseline)
    - method_names: optional list of method names to label each DataFrame
    - top_metrics: list of top metrics to compare

    Returns:
    - Combined DataFrame with all results and method labels
    """
    combined = []
    if method_names is None:
        method_names = [f'Method_{i+1}' for i in range(len(dfs))]

    for df, name in zip(dfs, method_names):
        df_copy = df.copy()
        df_copy['Method'] = name
        combined.append(df_copy)

    all_data = pd.concat(combined, ignore_index=True)

    for metric in top_metrics:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=all_data, x='Method', y=metric)
        plt.title(f'Comparison of {metric} across Methods')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    return all_data


