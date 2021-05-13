import pandas as pd


def load_tourism():
    df = pd.read_csv('data/Tourism.csv')
    return df