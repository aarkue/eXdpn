from pandas import DataFrame, concat, Series
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from typing import Tuple, List


def data_preprocessing_evaluation(dataframe: DataFrame) -> Tuple[DataFrame, DataFrame, Series, Series]:
    """ Data preprocessing for dataframes before they are used for the machine learning model selection. This does some \
    basic preprocessing, such as converting all columns to the correct data type, droping of columns with only NaNs and \
    defining feature variables and target variables. Furthermore, the data is split into a train and test data sets \
    and each numeric feature is scaled with a MinMaxScaler to [0, 1]\
    into training and test sets.
    Args:
        dataframe (DataFrame): Dataframe to be transformed for evaluation of the best model
    Returns: 
        X_train, X_test, y_train, y_test (DataFrame): Preprocessed and splitted data
        #data_scaler (MinMaxScaler]): MinMaxScaler fitted on data set, scales to [0, 1]
        #scalable_columns (pandas.core.indexes.base.Index): List of columns names of all columns that can be scaled
    """

    # perform basic preprocessing
    df_X, df_y = basic_data_preprocessing(dataframe)

    # split data
    # use mapping for stratify (map transition to integers)
    transition_int_map = {transition: index for index, transition in enumerate(list(set(df_y)))}
    df_y_transformed = [transition_int_map[transition] for transition in df_y]
    X_train, X_test, y_train_mapped, y_test_mapped = train_test_split(df_X, df_y_transformed, stratify = df_y_transformed)

    # map back to transitions
    y_train = [next(trans for trans, trans_id in transition_int_map.items() if trans_id == y) for y in y_train_mapped]
    y_test = [next(trans for trans, trans_id in transition_int_map.items() if trans_id == y) for y in y_test_mapped]


    return X_train, X_test, pd.Series(y_train), pd.Series(y_test) 


def basic_data_preprocessing(dataframe: DataFrame) -> Tuple[DataFrame]:
    """ Basic preprocessing before dataframes, i.e., converting all columns to the correct data type, droping of columns \
    with only NaNs and defining feature variables and target variables
    Args:
        dataframe (DataFrame): Dataframe to be transformed
    Returns: 
        df_X (DataFrame): Preprocessed dataframe of feature variables
        df_y (DataFrame): Preprocessed dataframe of target variable
    """

    # TODO define more correct data types?
    # convert timestamp to datatype "datetime"
    if "event::time:timestamp" in dataframe.columns:
        dataframe["event::time:timestamp"] = pd.to_datetime(
            dataframe["event::time:timestamp"])

    # get target and feature names
    target_var = "target"
    df_X = dataframe.copy()
    df_X = df_X.drop(target_var, axis = 1)
    df_y = dataframe.copy()
    df_y = dataframe[target_var]

    # drop columns with all NaNs
    df_X = df_X.dropna(how = 'all', axis = 1)

    # drop case::concept:name in event logs - if existing
    # if "case::concept:name" in df_X.columns:
    #    df_X = df_X.drop(["case::concept:name"], axis = 1)

    return df_X, df_y


def fit_scaling(X: DataFrame) -> Tuple[MinMaxScaler, List[str]]:
    """ Fits a MinMaxScaler on the data and returns a scaler for a scaling t o [0, 1] and the scalable columns 
    Args: 
        X (DataFrame): Dataframe with data to scale
    Returns: 
        scaler (MinMaxScaler): MinMaxScaler fitted on data set, scales to [0, 1]
        scalable_columns (pandas.core.indexes.base.Index): List of columns names of all columns that can be scaled
    """
    # exclude all columns that cannot be scaled
    scalable_columns = X.select_dtypes(include = [np.number]).columns

    if len(scalable_columns) == 0: return None, []

    # define and fit scaler
    scaler = MinMaxScaler(feature_range = (0, 1))
    scaler.fit(X[scalable_columns])

    return scaler, list(scalable_columns)


def apply_scaling(X: DataFrame, scaler: MinMaxScaler, scalable_columns: List[str]) -> DataFrame:
    """ Performs min-max scaling to [0, 1] on data with a fitted scaler on all scalable columns and returns scaled data
    Args: 
        X (DataFrame): Dataframe with data to scale
        scaler (MinMaxScaler): MinMaxScaler fitted on data set, scales to [0, 1]
        scalable_columns (pandas.core.indexes.base.Index): List of columns names of all columns that can be scaled
    Returns: 
        X_scaled (DataFrame): Scaled data, where each feature is scaled to [0, 1]
    """
    # apply scaler on data
    X_scaled = X.copy()

    if len(scalable_columns) == 0: return X

    X_scaled[scalable_columns] = scaler.transform(X_scaled[scalable_columns])
    return X_scaled


def fit_ohe(X: DataFrame) -> Tuple[OneHotEncoder, List[str]]:
    """ Fits anOneHotEncoder on all categorical features in the data set
    Args: 
        X (DataFrame): Dataframe with data to encode
    Returns: 
        OneHotEncoder (OneHotEncoder): Fitted Encoder, used to encode categorical data
        ohe_column_names (List[str]): List of column names of One Hot Encoded dataframe
    """
    ohe = OneHotEncoder(sparse = False, handle_unknown = 'ignore')
    X_object = X.select_dtypes('object')

    return ohe.fit(X_object), list(X_object.columns)


def apply_ohe(X: DataFrame, ohe: OneHotEncoder) -> DataFrame:
    """ Performs One Hot Encoding on all categorical features in the data set. This is necessary for machine learning \
    techniques that cannot handle categorical data, such as Decision Trees, SVMs and Neural Networks
    Args: 
        X (DataFrame): Dataframe with data to encode
        OneHotEncoder (OneHotEncoder): Fitted Encoder, used to encode categorical data
    Returns: 
        X_encoded (DataFrame): Encoded data, if dataframe does not contain categorical data, the original \
        dataframe is returned
    """   
    X = X.reset_index(drop = True)
    X_object = X.select_dtypes('object')

    X_object_enc = ohe.transform(X_object)
    feature_names = ohe.get_feature_names_out(list(X_object.columns))
    return concat([X.select_dtypes(exclude = 'object'), DataFrame(X_object_enc, columns = feature_names)], axis=1)
