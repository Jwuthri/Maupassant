import json

import pandas as pd


def cbind(df1, df2):
    return pd.concat([df1, df2], axis=1, ignore_index=True)


def rbind(df1, df2):
    return pd.concat([df1, df2], axis=0, ignore_index=True)


def remove_rows_contains_null(df, column):
    return df[df[column].notnull()]


def keep_rows_contains_null(df, column):
    return df[~df[column].notnull()]


def compare_data_frame(df1, df2, column):
    return df1[~df1[column].isin(df2[column])]


def count_frequency(df, col, column_name="Frequency"):
    df[column_name] = df.groupby(col)[col].transform('count')

    return df


def change_nan_value(df, new_value):
    return df.where((pd.notnull(df)), new_value)


def extract_json(serie, k):
    return serie.map(lambda row: json.loads(row)[k])


def categories_frequency(df, col):
    return df[col].value_counts()
