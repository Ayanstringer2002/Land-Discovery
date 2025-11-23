import argparse
import pandas as pd
import joblib
import os

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    return joblib.load(model_path)


def run_inference(model, input_data):
    """
    input_data: pandas DataFrame
    """
    predictions = model.predict(input_data)
    return predictions


def load_input_data(input_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    return df


def main(model_path, input_path, output_path):
    print("Loading model...")
    model = load_model(model_path)

    print("Loading input data...")
    input_df = load_input_data(input_path)

    print("Running inference...")
    preds = run_inference(model, input_df)

    # Save predictions
    input_df["prediction"] = preds
    input_df.to_csv(output_path, index=False)

    print("Inference complete ✅")
    print(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run inference on trained model")
    
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model file (.pkl)")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output_path", type=str, default="predictions.csv", help="Path to save predictions")

    args = parser.parse_args()

    main(args.model_path, args.input_path, args.output_path)
