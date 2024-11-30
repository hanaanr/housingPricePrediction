import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from utils import handle_infinite_values, remove_low_variance_features

def load_and_clean_data(file_path):
    # Load data
    df = pd.read_csv(file_path)
    
    # Remove duplicate properties
    df = df.drop_duplicates()

    # Outttttttttliers
    df = handle_outliers(df)
    
    # Create new features
    df['Age'] = 2024 - df['YearBuilt']
    
    
    # Handle Vacant Land properties
    df = df[df['PropertyType'] != 'Vacant Land']
    
    # Create property type categories
    df['PropertyTypeCategory'] = df['PropertyType'].map({
        'Single Family Residential': 'Single Family',
        'Condo/Co-op': 'Condo',
        'Townhouse': 'Townhouse',
        'Mobile/Manufactured Home': 'Other',
        'Unknown': 'Other'
    })
    
    # Create ZIP code price categories more safely
    try:
        zip_price_medians = df.groupby('ZipCode')['Price'].median()
        if len(zip_price_medians.unique()) >= 4:
            df['ZipCodePriceCategory'] = pd.qcut(
                zip_price_medians[df['ZipCode']].values,
                q=3,
                labels=['Low', 'Medium', 'High'],
                duplicates='drop'
            )
        else:
            df['ZipCodePriceCategory'] = pd.cut(
                zip_price_medians[df['ZipCode']].values,
                bins=2,
                labels=['Low', 'High'],
                include_lowest=True
            )
    except Exception as e:
        logging.warning(f"Could not create ZIP code price categories: {e}")
        df['ZipCodePriceCategory'] = 'Medium'
    
    return df

def handle_outliers(df):
    # Define columns to check for outliers
    numeric_cols = ['Price', 'SquareFeet']
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.01)  # Even less aggressive
        Q3 = df[col].quantile(0.99)  # Even less aggressive
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Instead of removing, cap the values
        df[col] = df[col].clip(lower_bound, upper_bound)
    
    return df

def handle_multicollinearity(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=min(0.95, X.shape[1] - 1))  # Ensure we don't exceed the number of features
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca

def prepare_features(df):

    # Add new features
    df['Age'] = 2024 - df['YearBuilt']
    df['TotalRooms'] = df['Bedrooms'] + df['Bathrooms']

    # Log transform skewed features
    df['LogPrice'] = np.log1p(df['Price'])
    df['LogSquareFeet'] = np.log1p(df['SquareFeet'])

    # Create interaction terms with error handling
    df['BedroomBathroomRatio'] = np.where(df['Bathrooms'] > 0, df['Bedrooms'] / df['Bathrooms'], np.nan)
    df['RoomsPerSqFt'] = np.where(df['SquareFeet'] > 0, df['TotalRooms'] / df['SquareFeet'], np.nan)

    # One-hot encode PropertyType
    property_type_dummies = pd.get_dummies(df['PropertyType'], prefix='PropType')

     # Select features for modeling
    feature_cols = ['Bedrooms', 'Bathrooms', 'LogSquareFeet', 'Age', 'TotalRooms', 
                    'BedroomBathroomRatio', 'RoomsPerSqFt']
    
    # Combine features
    X = pd.concat([df[feature_cols], property_type_dummies], axis=1)

    X = handle_infinite_values(X)
    X = remove_low_variance_features(X)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # Handle multicollinearity
    X, pca = handle_multicollinearity(X_scaled)
    
    return X_scaled, df['LogPrice']




