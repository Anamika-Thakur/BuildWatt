"""
Stage 1: Data Preparation & Splitting
Splits the UCI Energy Efficiency dataset into train, calibration, and test sets.
"""
import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURE_NAMES = [
    'Relative_Compactness', 'Surface_Area', 'Wall_Area', 'Roof_Area',
    'Overall_Height', 'Orientation', 'Glazing_Area', 'Glazing_Area_Distribution'
]
TARGET_NAMES = ['Heating_Load', 'Cooling_Load']


def prepare_data(data_path="ENB2012_data.xlsx", output_dir="data", random_state=42):
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Input file not found: {data_path}")
        
    df = pd.read_excel(data_path)
    
    column_mapping = {
        'X1': 'Relative_Compactness', 'X2': 'Surface_Area', 'X3': 'Wall_Area',
        'X4': 'Roof_Area', 'X5': 'Overall_Height', 'X6': 'Orientation',
        'X7': 'Glazing_Area', 'X8': 'Glazing_Area_Distribution',
        'Y1': 'Heating_Load', 'Y2': 'Cooling_Load'
    }
    if all(k in df.columns for k in ['X1', 'X2', 'Y1', 'Y2']):
        df = df.rename(columns=column_mapping)
    else:
        df.columns = FEATURE_NAMES + TARGET_NAMES
        
    # Split: Train (60%), Calibration (20%), Test (20%)
    train_df, temp_df = train_test_split(df, test_size=0.40, random_state=random_state)
    cal_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=random_state)
    
    train_path = os.path.join(output_dir, "train.csv")
    cal_path = os.path.join(output_dir, "cal.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    train_df.to_csv(train_path, index=False)
    cal_df.to_csv(cal_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Data prepared successfully:")
    print(f"  Train: {train_df.shape} -> {train_path}")
    print(f"  Cal:   {cal_df.shape}   -> {cal_path}")
    print(f"  Test:  {test_df.shape}  -> {test_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="ENB2012_data.xlsx")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    prepare_data(args.data_path, args.output_dir, args.random_state)

