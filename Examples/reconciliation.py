import pandas as pd
import sys
sys.path.append('../')
from pyhts.hierarchy import Hierarchy
from pyhts.HFModel import HFModel
import pickle as pkl


def example1():
    df = pd.read_csv('data/Tourism.csv')
    df_train = df.iloc[:-12, :]
    df_test = df.iloc[-12:, :]
    hierarchy = Hierarchy.from_names(df.columns, chars=[1, 1, 1], period=12)
    model = HFModel(hierarchy,
                    base_forecasters='arima',
                    hf_method='comb',
                    weights='ols')
    print('fitting model...')
    model.fit(df_train)
    print('predicting model....')
    forecast = model.predict(12)
    print('measuring...')
    print(hierarchy.accuracy(df_test, forecast, hist=df_train))


def example2():
    df = pd.read_csv('data/Tourism.csv')
    df_train = df.iloc[:-12, :]
    df_test = df.iloc[-12:, :]
    hierarchy = Hierarchy.from_names(df.columns, [1, 1, 1], 12)
    with open('base.pkl', 'rb') as f:
        forecasters = pkl.load(f)

    print('mint')
    model = HFModel(hierarchy, forecasters, hf_method='comb', comb_method='mint', weights='shrinkage',
                    constrain_level=-1)
    model.fit(df_train)
    forecasts = model.predict(horizon=12)
    print(hierarchy.accuracy(df_test, forecasts, hist=df_train, levels=[0]))

    model = HFModel(hierarchy, forecasters, hf_method='comb', comb_method='ols', constrain_level=0)
    model.fit(df_train)
    forecasts = model.predict(horizon=12)
    print(hierarchy.accuracy(df_test, forecasts, hist=df_train, levels=[0]))

    model = HFModel(hierarchy, forecasters, hf_method='comb', comb_method='ols', constrain_level=1)
    model.fit(df_train)
    forecasts = model.predict(horizon=12)
    print(hierarchy.accuracy(df_test, forecasts, hist=df_train, levels=[0]))

    base_forecasts = model.generate_base_forecast(12)
    print(hierarchy.accuracy_base(df_test, base_forecasts, hist=df_train, levels=[0]))

