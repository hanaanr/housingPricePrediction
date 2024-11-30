from data_processing import load_and_clean_data, prepare_features
from model_builder import build_models
from analyze_results import analyze_housing_data, analyze_model_results
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    try:
        logging.info("Loading and cleaning data...")
        df = load_and_clean_data('housing_data.csv')
        logging.info("Data loaded and cleaned successfully")
    except Exception as e:
        logging.error(f"Error in data loading and cleaning: {e}")
        return

    try:
        logging.info("Analyzing housing data...")
        analyze_housing_data(df)

        logging.info("Preparing features...")
        X, y = prepare_features(df)

        logging.info("Building models...")
        results = build_models(X, y)

        logging.info("Analyzing model results...")
        analyze_model_results(results)
    except Exception as e:
        logging.error(f"Error in model building or analysis: {e}")

if __name__ == "__main__":
    main()
