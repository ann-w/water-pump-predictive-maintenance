import pandas as pd


def aggregate_categories(data: pd.DataFrame, column: str, threshold: int=50) -> pd.DataFrame:
    # count the number of occurrences of each unique value in column
    value_counts = data[column].value_counts()

    # replace values with 'other' that are not in the top threshold categories
    if len(value_counts) > threshold:
        replace_values = value_counts[threshold:].index
        return data[column].replace(replace_values, 'other')
    else:
        return data[column]

