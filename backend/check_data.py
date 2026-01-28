
import json
import pandas as pd
import numpy as np

try:
    with open('d:/character_recognition/backend/app_meta.json', 'r') as f:
        meta = json.load(f)
    print(f"Meta pred_emnist len: {len(meta['pred_emnist'])}")
except Exception as e:
    print(f"Error loading meta: {e}")

try:
    # Read only first few rows to check columns quickly, but we need total length
    # Getting total length of CSV can be slow if huge, but let's try reading just column 0
    df = pd.read_csv('d:/character_recognition/backend/A_Z Handwritten Data.csv', usecols=[0])
    print(f"CSV Total Rows: {len(df)}")
    print(f"First 5 labels: {df.iloc[:5, 0].tolist()}")
except Exception as e:
    print(f"Error loading CSV: {e}")
