import numpy as np
import pandas as pd
# from sklearn.preprocessing import normalize


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

def label(df):
    df['label'] = np.where(df['area'] == 0, 0, 1)
    # print(df[400:430])
    return df

def preprocess(path: str, add_class=False):
    df = pd.read_csv(path)

    # feature engineering
    
    # df = df[df['area'] > 0]
    # normalize(df['FFMC'])
    df.loc[df['FFMC'] < 60, 'FFMC'] = 60
    df.loc[df['ISI'] > 25, 'ISI'] = 25
    df.loc[df['rain'] > 1.8, 'rain'] = 1.8

    df['FFMC_std'] = (df['FFMC'] - df['FFMC'].mean()) / df['FFMC'].std()
    df['ISI_std'] = (df['ISI'] - df['ISI'].mean()) / df['ISI'].std()
    df['rain_std'] = (df['rain'] - df['rain'].mean()) / df['rain'].std()
    # df = df[df['ISI'].apply(normalize)]
    
    df = cat_to_num(df)
    print(df.columns)
    
    if add_class:
        df = label(df)
    X = df.drop(['area', 'FFMC', 'ISI', 'rain'], axis=1)
    # X = df.drop(['area', 'rain'], axis=1)

    
    # y = df['area']
    # print(df.head())
    y = df['area'].apply(transform)
    # print(X[:10])
    # print(y[:10])
    return X, y
