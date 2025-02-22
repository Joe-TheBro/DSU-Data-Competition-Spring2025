import argparse
import pickle
import pandas as pd
from model import DateFeatureEngineer, FeatureRemoval, categorical_selector_function

def run_inference(data_path):
    # Read the csv file
    df = pd.read_csv(data_path)
    
    if 'High Cost Claim' in df.columns:
        df.drop(['High Cost Claim'], axis=1)

    # Run predictions
    preds = model.predict(df)

    # Save results
    df['High Cost Claim predictions'] = preds
    output_path = data_path.replace('.csv', '_predictions.csv')
    df.to_csv(output_path)

    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":

    # Load the model
    model = pickle.load(open('train_pipeline.pkl', 'rb'))
    parser = argparse.ArgumentParser(description="Run inference on a dataset.")
    parser.add_argument("data_path", type=str, help="Path to the input parquet file.")
    args = parser.parse_args()

    run_inference(args.data_path)
