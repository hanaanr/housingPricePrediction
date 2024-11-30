import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def analyze_housing_data(df):
    print("\nData Summary:")
    print(f"Total properties: {len(df)}")
    print(f"Property types:\n{df['PropertyType'].value_counts()}")
    print(f"\nPrice statistics:")
    print(df['Price'].describe())
    
    # Create visualizations
    plt.figure(figsize=(15, 5))
    
    # Price distribution
    plt.subplot(131)
    sns.histplot(df['Price'], bins=30)
    plt.title('Price Distribution')
    
    # Price vs Square Feet
    plt.subplot(132)
    sns.scatterplot(data=df, x='SquareFeet', y='Price')
    plt.title('Price vs Square Feet')
    
    # Price by Property Type
    plt.subplot(133)
    sns.boxplot(data=df, x='PropertyType', y='Price')
    plt.xticks(rotation=45)
    plt.title('Price by Property Type')
    
    plt.tight_layout()
    plt.savefig('housing_analysis.png')
    plt.close()

def analyze_model_results(results):
    print("\nModel Performance Comparison:")
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"RMSE: ${result['rmse']:,.2f}")
        print(f"R² Score: {result['r2']:.4f}")
        print(f"Cross-validation R² scores: {result['cv_scores'].mean():.4f} (+/- {result['cv_scores'].std() * 2:.4f})")
        print("\nTop 10 Important Features:")
        print(result['feature_importance'].head(10))
