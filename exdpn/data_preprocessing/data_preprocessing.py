"""
.. include:: ./../../docs/_templates/md/data_preprocessing/data_preprocessing.md

"""

from pandas import DataFrame, concat, Series
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from typing import Tuple, List


def basic_data_preprocessing(dataframe: DataFrame, impute: bool = False) -> Tuple[DataFrame, Series]:
    """Basic preprocessing before datasets, i.e., dropping of columns \
    with only missing values and rows with any NaN value, defining feature attributes and the target attribute.

    Args:
        dataframe (DataFrame): The dataset to be transformed.
        impute (bool): If `True`, missing attribute values will be imputed using constants and an indicator columns will be added. Default is `False`. 

    Returns:
        * df_X (DataFrame): The preprocessed dataset of feature attributes.
        * df_y (Series): The preprocessed dataset of the target attribute.

    """
    # drop columns with all NaNs
    dataframe.dropna(how='all', axis=1, inplace=True)
    # Drop all rows which contain at least one NaN (after NaN Columns are dropped)
    # or impute missing values
    if not impute:
        dataframe.dropna(how='any', axis=0, inplace=True) 

    # get target and feature names
    target_var = "target"
    df_X = dataframe.copy()
    df_X = df_X.drop(target_var, axis=1)
    df_y = dataframe.copy()
    df_y = dataframe[target_var]

    # impute missing values
    if impute:
        numeric_columns = df_X.select_dtypes(include=['number']).columns 
        # use SI on left over object columns since it behaves odd on numerical columns
        si = SimpleImputer(strategy='most_frequent', add_indicator=True)
        idf_X = pd.DataFrame(si.fit_transform(df_X, df_y), columns=si.get_feature_names_out())
        idf_X.index = df_X.index

        df_X = idf_X
        df_X[numeric_columns] = df_X[numeric_columns].replace("missing_value", 0)
        for c in numeric_columns:
            df_X[c] = pd.to_numeric(df_X[c], errors='coerce')

        indicator_cols = [col for col in si.get_feature_names_out() if col not in si.feature_names_in_]
        df_X[indicator_cols] = df_X[indicator_cols].replace({True: 1, False: 0})

    return df_X, df_y


def fit_scaling(X: DataFrame) -> Tuple[StandardScaler, List[str]]:
    """Fits a min-max-scaler on the dataset and returns a scaler for a scaling to [0, 1] as well as the scalable columns.

    Args:
        X (DataFrame): The dataset with the data to fit.

    Returns:
        * scaler (StandardScaler): The standard-scaler fitted on dataset. Scales to mean 0 and standard deviation 1.
        * scalable_columns (List[str]): The list of names of columns that can be scaled.

    """
    # exclude all columns that cannot be scaled
    scalable_columns = X.select_dtypes(include=[np.number]).columns

    if len(scalable_columns) == 0:
        return None, []

    # define and fit scaler
    scaler = StandardScaler()
    scaler.fit(X[scalable_columns])

    return scaler, list(scalable_columns)


def apply_scaling(X: DataFrame, scaler: StandardScaler, scalable_columns: List[str]) -> DataFrame:
    """Performs min-max scaling on data with a fitted scaler object on all scalable columns.

    Args:
        X (DataFrame): The dataset with the data to scale.
        scaler (StandardScaler): The fitted standard-scaler.
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
