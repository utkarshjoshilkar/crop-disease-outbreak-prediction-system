import pandas as pd
try:
    df = pd.read_csv('synthetic_crop_disease_dataset_v2.csv')
    crops = sorted(df['crop_type'].unique().tolist())
    print("UNIQUE_CROPS:" + ",".join(crops))
except Exception as e:
    print("ERROR:" + str(e))
