import pandas as pd
from datetime import date
from impute_missing_values import impute_missing_values_for_continuous_columns, \
    impute_missing_values_for_categorical_columns
from aggregate_categories import aggregate_categories


def pipeline(file_path: str, export_csv: bool = True) -> pd.DataFrame:

    data = pd.read_csv(file_path, parse_dates=['date_recorded'])

    # impute missing values
    data = impute_missing_values_for_continuous_columns(data)
    data = impute_missing_values_for_categorical_columns(data)

    # replace values with 'other' that are not in the top threshold categories
    data['funder'] = aggregate_categories(data, 'funder', threshold=30)
    data['installer'] = aggregate_categories(data, 'installer', threshold=30)

    # Add additional feature: year_recorded
    data['year_recorded'] = data.date_recorded.dt.year

    # drop cols
    columns_to_drop = ['wpt_name', 'subvillage', 'scheme_name', 'ward', 'id', 'date_recorded','gps_height', 'population', 'longitude', 'latitude', 'installer', 'extraction_type', 'extraction_type_class', 'waterpoint_type_group', 'region_code', 'scheme_management', 'management_group', 'payment', 'recorded_by']
    data.drop(columns_to_drop, axis=1, inplace=True)

    # encode dataset
    cat_cols = data.select_dtypes(include=['object']).columns.tolist()

    if 'status_group' in cat_cols:

        # encode target
        target_map = {
            "functional": 0,
            "non functional": 1,
            "functional needs repair": 2
        }

        data['status_group'] = data['status_group'].map(target_map)

        cat_cols.remove('status_group')

    # create dummies
    data = pd.get_dummies(data, columns=cat_cols)

    # Save
    if export_csv:
        today = date.today()
        today_date = today.strftime("%d_%m_%Y")
        data.to_csv(f'../../../data/processed/water_pump_dataset_encoded_{today_date}.csv', index=False)

    return data

# Example
# pipeline("../../../data/external/water_pump_original.csv")
