import pandas as pd
import numpy as np


def _get_column_mean_of_region(data: pd.DataFrame, region_code: int, column_name: str):
    data_without_zeros_in_col = data[[column_name, 'region_code']][data[column_name] != 0]
    available_region_codes = data_without_zeros_in_col['region_code'].unique().tolist()
    if region_code in available_region_codes:
        col_mean = data_without_zeros_in_col[column_name][data_without_zeros_in_col['region_code']==region_code].mean()
        return int(col_mean)
    else:
        return 0


def impute_missing_values_for_continuous_columns(data: pd.DataFrame) -> pd.DataFrame:

    # gps_height
    data['gps_height'] = data['gps_height'].replace(to_replace=0, value=data[
        'gps_height'].median())

    # population
    # Get population mean from region
    data['population'] = data.apply(
        lambda row: _get_column_mean_of_region(data, row['region_code'], 'population') if row['population'] == 0 else row[
            'population'], axis=1)
    # Impute 0 values with median
    data['population'] = data['population'].replace(to_replace=0, value=data[
        'population'].median())

    # longitude
    # Get mean from region
    data['longitude'] = data.apply(
        lambda row: _get_column_mean_of_region(data, row['region_code'], 'longitude') if row['longitude'] == 0 else row[
            'longitude'], axis=1)
    # Impute 0 values with median
    data['longitude'] = data['longitude'].replace(to_replace=0, value=data[
        'longitude'].median())

    # Latitude
    # For all missing longitudes, the corresponding latitude is -2.e-08, replace with 0
    data['latitude'] = data['latitude'].replace(-2.e-08, 0)
    # Then replace 0's with mean latitude of the region
    data['latitude'] = data.apply(
        lambda row: _get_column_mean_of_region(data, row['region_code'], 'latitude') if row['latitude'] == 0 else row[
            'latitude'], axis=1)
    # Impute 0 values with median
    data['latitude'] = data['latitude'].replace(to_replace=0, value=data[
        'latitude'].median())

    return data


def impute_missing_values_for_categorical_columns(data: pd.DataFrame) -> pd.DataFrame:

    # transform all strings to lowercase
    data = data.applymap(lambda s: s.lower() if type(s) == str else s)

    # transform strings like none or nan to np.nan
    data = data.replace({'none': np.nan, 'nan': np.nan, '0': np.nan})

    # replace '[' and ']' with ''
    data = data.replace({'\[': ' ', '\]': ' ', '  ': ''}, regex=True)

    # get categorical columns
    cat_cols = data.select_dtypes(include='object').columns

    # replace NaN values with 'not_available'
    data[cat_cols].fillna(value='not_available', inplace=True)

    return data