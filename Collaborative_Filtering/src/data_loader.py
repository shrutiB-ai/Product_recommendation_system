import pandas as pd
import yaml

def load_config(path='config.yaml'):
    with open(path,'r') as f:
        return yaml.safe_load(f)

def load_interactions(path):
    df= pd.read_csv(path)
    required_cols = ['user', 'item']
    if 'rating' not in df.columns:
        df['rating'] =1.0

    return df[['user','item','rating']]

