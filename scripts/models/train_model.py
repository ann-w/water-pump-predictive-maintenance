import logging
import pickle
import pandas as pd
from datetime import date
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO)


def split_data(data_file: str, test_size=0.25):

    data=pd.read_csv(data_file)

    # split into features and target
    feature_cols = list(data.columns)
    feature_cols.remove('status_group')
    features = data[feature_cols]

    target = data['status_group']

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=1)

    return X_train, X_test, y_train, y_test


def train_classifiers(data_file: str, test_size: float = 0.25, save_models: bool = False):

    classifiers = [
        ['decision_tree', DecisionTreeClassifier(splitter='best', random_state=8)],
        ['random_forest', RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=8)],
        ['k_nearest_neighbors', KNeighborsClassifier(n_neighbors=10)],
        ['xgboost', XGBClassifier(objective='multi:softmax', random_state=8)],
    ]

    # split data
    X_train, X_test, y_train, y_test = split_data(data_file, test_size)

    for classifier in classifiers:
        logging.info('=' * 80)
        name, clf = classifier[0], classifier[1]
        logging.info(f'Model: {name}')
        logging.info('-' * 80)

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        logging.info(metrics.classification_report(y_test, y_pred))

        if save_models:
            today = date.today()
            today_date = today.strftime("%d_%m_%Y")
            filename = f'../../models/{name}_{today_date}.sav'
            pickle.dump(clf, open(filename, 'wb'))


# Example
# train_classifiers("../../data/processed/water_pump_dataset_encoded.csv", save_models=True)
