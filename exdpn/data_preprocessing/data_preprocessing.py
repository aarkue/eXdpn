from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd 
import sys
import numpy as np

def data_preprocessing_evaluation(dataframe: DataFrame) -> tuple[DataFrame, MinMaxScaler, pd.core.indexes.base.Index]:
    """ Data preprocessing for dataframes before they are used for the machine learning model selection. This does some \
    basic preprocessing, such as converting all columns to the correct data type, droping of columns with only NaNs and \
    defining feature variables and target variables. Furthermore, the data is split into a train and test data sets \
    and each numeric feature is scaled with a MinMaxScaler to [0, 1\
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

    # define scaler on trainings data only to reduce bias 
    # https://datascience.stackexchange.com/questions/39932/feature-scaling-both-training-and-test-data
    data_scaler, scalable_columns = fit_scaling(X_train) 

    # apply scaling on training and test data
    X_train = apply_scaling(X_train, data_scaler, scalable_columns)
    X_test = apply_scaling(X_test, data_scaler, scalable_columns)

    return X_train, X_test, y_train, y_test #, data_scaler, scalable_columns 

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
        dataframe["event::time:timestamp"] = pd.to_datetime(dataframe["event::time:timestamp"])
    
    # get target and feature names
    target_var = "target"
    df_X = dataframe.copy()
    df_X = df_X.drop(target_var, axis = 1)
    df_y = dataframe.copy()
    df_y = dataframe[target_var]

    # drop columns with all NaNs
    df_X = df_X.dropna(how = 'all', axis = 1)

    # drop case::concept:name in event logs - if existing 
    #if "case::concept:name" in df_X.columns:
    #    df_X = df_X.drop(["case::concept:name"], axis = 1) 

    return df_X, df_y


def fit_scaling(X: DataFrame) -> tuple[MinMaxScaler, pd.core.indexes.base.Index]:
    """ Fits a MinMaxScaler on the data and returns a scaler for a scaling t o [0, 1] and the scalable columns 
    Args: 
        X (DataFrame): Dataframe with data to scale
    Returns: 
        scaler (MinMaxScaler): MinMaxScaler fitted on data set, scales to [0, 1]
        scalable_columns (pandas.core.indexes.base.Index): List of columns names of all columns that can be scaled
    """
    # exclude all columns that cannot be scaled
    scalable_columns = X.select_dtypes(include = [np.number]).columns
    
    # define and fit scaler
    scaler = MinMaxScaler(feature_range = (0, 1))
    scaler.fit(X[scalable_columns])

    return scaler, scalable_columns

def apply_scaling(X: DataFrame, scaler: MinMaxScaler, scalable_columns: pd.core.indexes.base.Index) -> DataFrame:
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
    if len(X.select_dtypes(include = [object]).columns) == 0:
        return X_encoded 
    else: 
        # split data into categorical and non-categorical features
        categorical_columns = X_encoded.select_dtypes(include = [object]).columns
        X_encoded = pd.get_dummies(X_encoded, columns = categorical_columns)

        # check persistence for new data
        if list(ohe_column_names):
            # check if column names are the same in general
            if set(ohe_column_names) == set(X_encoded.columns):
                # if order is not consistent, make it consistent
                if list(ohe_column_names) != list(X_encoded.columns):
                    X_encoded = X_encoded[ohe_column_names]
            # if the column names in the two data sets are not persistent, throw an error 
            else:
                sys.exit("The columns in the Training Data and now used data do not match. Please make sure that \
                    both data sets contain the same columns.")
        
        return X_encoded, ohe_column_names