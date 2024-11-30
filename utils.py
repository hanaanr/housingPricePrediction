import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

def handle_infinite_values(df):
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df[col] = df[col].fillna(df[col].median())
    return df

def calculate_vif(X):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

def remove_low_variance_features(X, threshold=0.01):
    selector = VarianceThreshold(threshold)
    selector.fit(X)
    return X[X.columns[selector.get_support(indices=True)]]
