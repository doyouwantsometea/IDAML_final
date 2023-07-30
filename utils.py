import numpy as np
import pandas as pd


month_to_idx = {
    'jan': 1,
    'feb': 2,
    'mar': 3,
    'apr': 4,
    'may': 5,
    'jun': 6,
    'jul': 7,
    'aug': 8,
    'sep': 9,
    'oct': 10,
    'nov': 11,
    'dec': 12
}

day_to_idx = {
    'mon': 1,
    'tue': 2,
    'wed': 3,
    'thu': 4,
    'fri': 5,
    'sat': 6,
    'sun': 7
}


def transform(value):
    return np.log(value+1)

def cat_to_num(df):
    df['month'] = df['month'].map(month_to_idx)
    df['day'] = df['day'].map(day_to_idx)
    return df

def preprocess(path: str):
    df = pd.read_csv(path)
    df = cat_to_num(df)
    print(df.columns)
    X = df.drop(['area'], axis=1)
    y = df['area']
    y = y.apply(transform)
    return X, y