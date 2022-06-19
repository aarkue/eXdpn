"""
.. include:: ./../../docs/_templates/md/data_preprocessing/data_preprocessing.md

"""

from pandas import DataFrame, concat, Series
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from typing import Tuple, List


def data_preprocessing_evaluation(dataframe: DataFrame) -> Tuple[DataFrame, DataFrame, Series, Series]:
    """Preprocessing of datasets before they are used for the machine learning training and testing. This function does some \
    basic preprocessing, such as converting all columns to the correct data type, droping columns with missing values and \
    defining feature attributes and the target attribute. Furthermore, the data is split into train and test datasets.

    Args:
        dataframe (DataFrame): The dataset to be transformed for evaluation of the best model.

    Returns:
        * X_train (DataFrame): The training data without the target attribute.
        * X_test (DataFrame): The test data without the target attribute.
        * y_train (Series): The target attribute values corresponding to the training data.
        * y_test (Series): The test attribute values corresponding to the training data.

    """

    # perform basic preprocessing
    df_X, df_y = basic_data_preprocessing(dataframe)

    # split data
    # use mapping for stratify (map transition to integers)
    transition_int_map = {transition: index for index,
                          transition in enumerate(list(set(df_y)))}
    df_y_transformed = [transition_int_map[transition] for transition in df_y]
    try:
        X_train, X_test, y_train_mapped, y_test_mapped = train_test_split(
            df_X, df_y_transformed, stratify=df_y_transformed)
    except ValueError:
        X_train, X_test, y_train_mapped, y_test_mapped = train_test_split(
            df_X, df_y_transformed)

    # map back to transitions
    y_train = [next(trans for trans, trans_id in transition_int_map.items(
    ) if trans_id == y) for y in y_train_mapped]
    y_test = [next(trans for trans, trans_id in transition_int_map.items(
    ) if trans_id == y) for y in y_test_mapped]

    return X_train, X_test, pd.Series(y_train), pd.Series(y_test)


def basic_data_preprocessing(dataframe: DataFrame) -> Tuple[DataFrame, Series]:
    """Basic preprocessing before datasets, i.e., dropping of columns \
    with only missing values and defining feature attributes and the target attribute.

    Args:
        dataframe (DataFrame): The dataset to be transformed.

    Returns:
        * df_X (DataFrame): The preprocessed dataset of feature attributes.
        * df_y (Series): The preprocessed dataset of the target attribute.

    """

    # don't use case attributes for prediction
    # if any("case::" in cols for cols in dataframe.columns):
    #    idx = [index for index in dataframe.columns if "case::" in index]
    #    dataframe = dataframe.drop(idx, axis = 1)

    # get target and feature names
    target_var = "target"
    df_X = dataframe.copy()
    df_X = df_X.drop(target_var, axis=1)
    df_y = dataframe.copy()
    df_y = dataframe[target_var]

    # drop columns with all NaNs
    df_X = df_X.dropna(how='all', axis=1)

    return df_X, df_y


def fit_scaling(X: DataFrame) -> Tuple[MinMaxScaler, List[str]]:
    """Fits a min-max-scaler on the dataset and returns a scaler for a scaling to [0, 1] as well as the scalable columns.

    Args:
        X (DataFrame): The dataset with the data to fit.

    Returns:
        * scaler (MinMaxScaler): The min-max-scaler fitted on dataset. Scales to [0, 1].
        * scalable_columns (List[str]): The list of names of columns that can be scaled.

    """
    # exclude all columns that cannot be scaled
    scalable_columns = X.select_dtypes(include=[np.number]).columns

    if len(scalable_columns) == 0:
        return None, []

    # define and fit scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(X[scalable_columns])

    return scaler, list(scalable_columns)


def apply_scaling(X: DataFrame, scaler: MinMaxScaler, scalable_columns: List[str]) -> DataFrame:
    """Performs min-max scaling on data with a fitted scaler object on all scalable columns.

    Args:
        X (DataFrame): The dataset with the data to scale.
        scaler (MinMaxScaler): The fitted min-max-scaler.
        scalable_columns (List[str]): The list of names of columns that will be scaled.

    Returns:
        DataFrame: The data where all `scalable_columns` were scaled using the `scaler`.

    """
    # apply scaler on data
    X_scaled = X.copy()

    if len(scalable_columns) == 0:
        return X

    X_scaled[scalable_columns] = scaler.transform(X_scaled[scalable_columns])
    return X_scaled


def fit_ohe(X: DataFrame) -> OneHotEncoder:
    """Fits an one-hot-encoder on all categorical features present in the dataset.

    Args:
        X (DataFrame): The dataset with the data to fit.

    Returns:
        OneHotEncoder: The one-hot-encoder fitted on the dataset.

    """
    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    X_object = X.select_dtypes('object')

    return ohe.fit(X_object)


def apply_ohe(X: DataFrame, ohe: OneHotEncoder) -> DataFrame:
    """Performs one-hot-encoding on all categorical features in the dataset.

    Args:
        X (DataFrame): The dataset with the data to encode.
        ohe (OneHotEncoder): The fitted one-hot-encoder.

    Returns:
        DataFrame: The data where all categorical columns were encoded using the `ohe`.

    """
    X = X.reset_index(drop=True)
    X_object = X.select_dtypes('object')

    X_object_enc = ohe.transform(X_object)
    feature_names = ohe.get_feature_names_out(list(X_object.columns))

    return concat([X.select_dtypes(exclude='object'), DataFrame(X_object_enc, columns=feature_names)], axis=1)
