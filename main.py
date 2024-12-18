import requests
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import time
import random
warnings.filterwarnings('ignore')

def get_rotating_headers():
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Edge/120.0.0.0',
        'Mozilla/5.0 (X11; Linux x86_64) Chrome/120.0.0.0 Safari/537.36'
    ]
    return {
        'Accept': '*/*',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive',
        'Host': 'www.redfin.com',
        'User-Agent': random.choice(user_agents),
        'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120"',
        'sec-ch-ua-mobile': '?0',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin'
    }

def exponential_backoff(retry_count):
    wait_time = min(300, (2 ** retry_count) + random.uniform(0, 1))
    time.sleep(wait_time)


def get_housing_data():
    # List to store all properties
    all_properties = []
    
    # Multiple cities in the Chicago metro area
    cities = [
        ('Chicago', '17426'),
        ('Aurora', '17357'),
        ('Naperville', '17459'),
        ('Joliet', '17428'),
        ('Evanston', '17413'),
        ('Oak Park', '17478'),
        ('Schaumburg', '17499'),
        ('Skokie', '17503'),
        ('Des Plaines', '17394'),
        ('Arlington Heights', '17358')
    ]
    
    property_types = ['', 'house', 'condo', 'townhouse']
    price_ranges = [
        (0, 200000),
        (200000, 400000),
        (400000, 600000),
        (600000, 800000),
        (800000, 1000000),
        (1000000, 1500000),
        (1500000, 2000000),
        (2000000, 3000000),
        (3000000, 5000000)
    ]
    
    pages_per_search = 5
    max_retries = 3
    total_collected = 0
    
    for city, region_id in cities:
        print(f"\nCollecting data for {city}...")
        time.sleep(random.uniform(2, 4))  # Random delay between cities
        
        for prop_type in property_types:
            for min_price, max_price in price_ranges:
                for page in range(1, pages_per_search + 1):
                    url = "https://www.redfin.com/stingray/api/gis-csv"
                    params = {
                        'al': '1',
                        'market': city.lower(),
                        'num_homes': '100',
                        'ord': 'redfin-recommended-asc',
                        'page_number': str(page),
                        'region_id': region_id,
                        'region_type': '6',
                        'sf': '1,2,3,5,6,7',
                        'status': '9',
                        'uipt': '1,2,3,4,5,6,7,8',
                        'v': '8',
                        'min_price': str(min_price),
                        'max_price': str(max_price)
                    }
                    
                    if prop_type:
                        params['property_type'] = prop_type
                        
                    download_url = f"{url}?{requests.compat.urlencode(params)}"
                    
                    for retry in range(max_retries):
                        try:
                            print(f"\nFetching {city} data (Page {page}):")
                            print(f"Property type: {prop_type or 'all'}")
                            print(f"Price range: ${min_price:,}-${max_price:,}")
                            
                            headers = get_rotating_headers()
                            session = requests.Session()
                            session.headers.update(headers)
                            
                            main_page = f"https://www.redfin.com/city/{region_id}/IL/{city}"
                            session.get(main_page, verify=False)
                            
                            exponential_backoff(retry)
                            
                            response = session.get(download_url, verify=False)
                            
                            if response.status_code == 200:
                                content = response.text
                                if content.startswith('{}&&'):
                                    content = content[4:]
                                
                                try:
                                    df = pd.read_csv(StringIO(content))
                                    if not df.empty:
                                        if 'ADDRESS' not in df.columns:
                                            print("Invalid data format received")
                                            continue
                                        df = df.dropna(how='all')
                                        if len(df) > 0:
                                            all_properties.append(df)
                                            total_collected += len(df)
                                            print(f"Successfully collected {len(df)} properties")
                                            print(f"Running total: {total_collected}")
                                            break
                                except pd.errors.EmptyDataError:
                                    print("Empty CSV received")
                                    continue
                                except Exception as e:
                                    print(f"Error parsing CSV: {str(e)}")
                                    continue
                            else:
                                print(f"Failed to fetch data. Status code: {response.status_code}")
                                
                        except Exception as e:
                            print(f"Error on attempt {retry + 1}: {e}")
                            exponential_backoff(retry)
                            continue
                    
                    time.sleep(random.uniform(1, 3))  # Random delay between pages
                
                time.sleep(random.uniform(2, 4))  # Random delay between price ranges
    
    if all_properties:
        combined_df = pd.concat(all_properties, ignore_index=True)
        print(f"\nTotal properties before deduplication: {len(combined_df)}")
        
        combined_df = combined_df.drop_duplicates(
            subset=['ADDRESS', 'PRICE', 'BEDS', 'BATHS', 'SQUARE FEET'],
            keep='first'
        )
        print(f"Total unique properties after deduplication: {len(combined_df)}")
        
        print("\nSample of collected data:")
        print(combined_df[['ADDRESS', 'PRICE', 'BEDS', 'BATHS', 'SQUARE FEET']].head())
        
        return combined_df
    else:
        print("No data collected")
        return None




def clean_data(df):
    if df is None or df.empty:
        return None
    
    df = df.copy()
    
    # Select relevant columns
    columns_to_keep = [
        'PRICE', 'BEDS', 'BATHS', 'SQUARE FEET', 
        'YEAR BUILT', 'PROPERTY TYPE', 'ZIP OR POSTAL CODE'
    ]
    
    # Keep only columns that exist in the DataFrame
    columns_to_keep = [col for col in columns_to_keep if col in df.columns]
    df = df[columns_to_keep]
    
    # Rename columns
    df.columns = ['Price', 'Bedrooms', 'Bathrooms', 'SquareFeet', 
                 'YearBuilt', 'PropertyType', 'ZipCode']
    
    # Convert numeric columns with imputation
    numeric_columns = ['Price', 'Bedrooms', 'Bathrooms', 'SquareFeet', 
                      'YearBuilt', 'ZipCode']
    
    for col in numeric_columns:
        if col in df.columns:
            # First convert to numeric, keeping NaN values
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Impute missing values
            if col == 'Price':
                # Use median for price
                df[col].fillna(df[col].median(), inplace=True)
            elif col in ['Bedrooms', 'Bathrooms']:
                # Use mode (most common value) for discrete features
                df[col].fillna(df[col].mode()[0], inplace=True)
            elif col == 'SquareFeet':
                # Use median for square footage
                df[col].fillna(df[col].median(), inplace=True)
            elif col == 'YearBuilt':
                # Use median for year built
                df[col].fillna(df[col].median(), inplace=True)
            elif col == 'ZipCode':
                # Use most common zipcode in the area
                df[col].fillna(df[col].mode()[0], inplace=True)
    
    # For PropertyType, fill missing values with 'Unknown'
    if 'PropertyType' in df.columns:
        df['PropertyType'].fillna('Unknown', inplace=True)
    
    # Round imputed values for certain columns
    df['Bedrooms'] = df['Bedrooms'].round()
    df['Bathrooms'] = df['Bathrooms'].round(1)  # Allow for half baths
    df['YearBuilt'] = df['YearBuilt'].round()
    df['ZipCode'] = df['ZipCode'].round()
    
    return df



def build_decision_tree(df):
    # Prepare features and target
    features = ['Bedrooms', 'Bathrooms', 'SquareFeet', 'YearBuilt']
    X = df[features]
    y = df['Price']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train model
    model = DecisionTreeRegressor(random_state=42, min_samples_split=5)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return model, rmse, r2, feature_importance

def main():
    print("Starting housing price prediction project...")
    
    # Get data
    df = get_housing_data()
    
    if df is not None and not df.empty:
        print(f"\nSuccessfully collected {len(df)} properties")
        
        # Clean data
        df_clean = clean_data(df)
        
        if df_clean is not None and not df_clean.empty:
            print(f"\nAfter cleaning: {len(df_clean)} properties")
            
            # Save data
            df_clean.to_csv('housing_data.csv', index=False)
            print("\nData saved to housing_data.csv")
            
            # Display basic statistics
            print("\nBasic Statistics:")
            print(f"Median Price: ${df_clean['Price'].median():,.2f}")
            print(f"Average Square Footage: {df_clean['SquareFeet'].mean():,.2f}")
            print(f"Average Bedrooms: {df_clean['Bedrooms'].mean():.1f}")
            
            if len(df_clean) >= 5:
                # Build and evaluate model
                model, rmse, r2, feature_importance = build_decision_tree(df_clean)
                
                print("\nModel Performance:")
                print(f"Root Mean Square Error: ${rmse:,.2f}")
                print(f"R² Score: {r2:.4f}")
                
                print("\nFeature Importance:")
                print(feature_importance)
            else:
                print("\nNot enough data for modeling (need at least 5 properties)")
        else:
            print("Failed to clean data")
    else:
        print("Failed to collect data")

if __name__ == "__main__":
    main()
