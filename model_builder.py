from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectFromModel
import numpy as np
import pandas as pd

def select_features(X, y):
    # Use RandomForestRegressor to select important features
    selector = SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42), threshold='median')
    selector.fit(X, y)
    return selector.transform(X), selector.get_support()

def tune_random_forest(X, y):
    param_dist = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [10, 20, 30, 40, 50, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]  # Remove 'auto'
    }
    rf = RandomForestRegressor(random_state=42)
    random_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=20, cv=5, random_state=42, error_score='raise')
    random_search.fit(X, y)
    return random_search.best_estimator_


def build_models(X, y):
    # Feature selection
    X_selected, feature_mask = select_features(X, y)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    
    # Add more sophisticated cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Decision Tree Model
    dt_model = DecisionTreeRegressor(random_state=42)
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)
    dt_rmse = np.sqrt(mean_squared_error(y_test, dt_pred))
    dt_r2 = r2_score(y_test, dt_pred)
    dt_scores = cross_val_score(dt_model, X_selected, y, cv=cv, scoring='r2')
    
    # Random Forest Model with tuning
    rf_model = tune_random_forest(X_selected, y)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    rf_r2 = r2_score(y_test, rf_pred)
    rf_scores = cross_val_score(rf_model, X_selected, y, cv=cv, scoring='r2')

        # Remove NaN values from scores
    dt_scores = dt_scores[~np.isnan(dt_scores)]
    rf_scores = rf_scores[~np.isnan(rf_scores)]
    
    # Get feature names after selection
    feature_names = X.columns[feature_mask]
    
    return {
        'Decision Tree': {
            'model': dt_model,
            'rmse': dt_rmse,
            'r2': dt_r2,
            'cv_scores': dt_scores,
            'feature_importance': pd.DataFrame({
                'Feature': feature_names,
                'Importance': dt_model.feature_importances_
            }).sort_values('Importance', ascending=False)
        },
        'Random Forest': {
            'model': rf_model,
            'rmse': rf_rmse,
            'r2': rf_r2,
            'cv_scores': rf_scores,
            'feature_importance': pd.DataFrame({
                'Feature': feature_names,
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=False)
        }
    }
