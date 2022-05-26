from pandas import DataFrame, concat, Series
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def data_preprocessing_evaluation(dataframe: DataFrame) -> tuple[DataFrame, DataFrame, Series, Series]:
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
    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y)

    return X_train, X_test, y_train, y_test


def basic_data_preprocessing(dataframe: DataFrame) -> tuple[DataFrame]:
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
    df_X = df_X.drop(target_var, axis=1)
    df_y = dataframe.copy()
    df_y = dataframe[target_var]

    # drop columns with all NaNs
    df_X = df_X.dropna(how='all', axis=1)

    # drop case::concept:name in event logs - if existing
    # if "case::concept:name" in df_X.columns:
    #    df_X = df_X.drop(["case::concept:name"], axis = 1)

    return df_X, df_y


def fit_scaling(X: DataFrame) -> tuple[MinMaxScaler, list[str]]:
    """ Fits a MinMaxScaler on the data and returns a scaler for a scaling t o [0, 1] and the scalable columns 
    Args: 
        X (DataFrame): Dataframe with data to scale
    Returns: 
        scaler (MinMaxScaler): MinMaxScaler fitted on data set, scales to [0, 1]
        scalable_columns (pandas.core.indexes.base.Index): List of columns names of all columns that can be scaled
    """
    # exclude all columns that cannot be scaled
    scalable_columns = X.select_dtypes(include=[np.number]).columns

    if len(scalable_columns) == 0: return None, []

    # define and fit scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(X[scalable_columns])

    return scaler, list(scalable_columns)


def apply_scaling(X: DataFrame, scaler: MinMaxScaler, scalable_columns: list[str]) -> DataFrame:
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


def fit_apply_ohe(X: DataFrame, ohe_column_names: pd.core.indexes.base.Index = []) -> tuple[DataFrame, pd.core.indexes.base.Index]:
    """ Performs One Hot Encoding on all categorical features in the data set. This is necessary for machine learning \
    techniques that cannot handle categorical data, such as Decision Trees, SVMs and Neural Networks
    Args: 
        X (DataFrame): Dataframe with data to encode
        ohe_column_names (pd.core.indexes.base.Index): List of column names to make One Hot Encoding persistant if new data is used
    Returns: 
        X_encoded (DataFrame): Encoded data, if dataframe does not contain categorical data, the original \
        dataframe is returned
        ohe_column_names (pd.core.indexes.base.Index): List of column names of current One Hot Encoded dataframe
    """
    X_encoded = X.copy()
    # check if data set contains categorical data, if yes: perform one hot encoding, no: skip
    if len(X.select_dtypes(include=[object]).columns) == 0:
        return X_encoded
    else:
        # split data into categorical and non-categorical features
        categorical_columns = X_encoded.select_dtypes(include=[object]).columns
        X_encoded = pd.get_dummies(X_encoded, columns=categorical_columns)

        # check persistence for new data
        if list(ohe_column_names):
            # check if column names are the same in general
            if set(ohe_column_names) == set(X_encoded.columns):
                # if order is not consistent, make it consistent
                if list(ohe_column_names) != list(X_encoded.columns):
                    X_encoded = X_encoded[ohe_column_names]
            # if the column names in the two data sets are not persistent, throw an error

            # TODO: we shouldnt throw an error but encode categorical cell by 0 in all expanded columns
            else:
                sys.exit("The columns in the Training Data and now used data do not match. Please make sure that \
                    both data sets contain the same columns.")

        # TODO: ohe_column names will remain [] but should not I believe
        return X_encoded, ohe_column_names


def fit_ohe(X: DataFrame) -> tuple[OneHotEncoder, list[str]]:
    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    X_object = X.select_dtypes('object')

    return ohe.fit(X_object), list(X_object.columns)


def apply_ohe(X: DataFrame, ohe: OneHotEncoder) -> DataFrame:
    X = X.reset_index(drop=True)
    X_object = X.select_dtypes('object')

    X_object_enc = ohe.transform(X_object)
    feature_names = ohe.get_feature_names_out(list(X_object.columns))
    return concat([X.select_dtypes(exclude='object'), DataFrame(X_object_enc, columns=feature_names)], axis=1)
