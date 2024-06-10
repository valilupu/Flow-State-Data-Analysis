import pandas as pd
import numpy as np


def preprocessing(file_path):

    df = pd.read_csv(file_path)

    # REMOVE OUTLIERS
    def handle_outliers(df, column):
        df['pct_change'] = df[column].pct_change().abs()
        outlier_mask = df['pct_change'] > 0.5
        df.loc[outlier_mask, column] = pd.NA
        df.drop('pct_change', axis=1, inplace=True)

    for column in ['points', 'time', 'accuracy']:
        handle_outliers(df, column)
        df[column] = df[column].interpolate().apply(np.ceil)

    # FIX TIME
    df['time'] = 60 - df['time']

    # DROP DUPLICATES
    df = df.drop_duplicates(subset=['points'])

    df.to_csv(file_path)
    df = pd.read_csv(file_path)

    df['points_diff'] = df['points'].diff(periods=2)

    drop_rows = []
    drop = False

    for i, points_diff_value in enumerate(df['points_diff']):
        if points_diff_value < 0:
            drop_rows.append(i)

    for i, time_value in enumerate(df['time']):
        if time_value == 60:
            drop = True
        if drop:
            drop_rows.append(i)
        if time_value == -40 and drop:
            drop = False

    df = df.drop(drop_rows)
    df = df[df['time'] != -40]

    columns_to_keep = ['points', 'time', 'accuracy', 'points_diff']
    df = df[columns_to_keep]

    return df

