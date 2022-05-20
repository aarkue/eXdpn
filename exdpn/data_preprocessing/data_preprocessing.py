from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from ml_technique import ML_Technique
from sklearn.preprocessing import OneHotEncoder

def data_preprocessing(dataframe: DataFrame, ml: str):

    # get target and feature names
    target_var = "target"
    df_X = dataframe.copy()
    df_X.loc[:, dataframe.columns != target_var]
    df_y = dataframe.copy()
    df_y = dataframe[target_var]

    # min max normalization to 0 and 1
    # define scaler for trainings data, then use for test data? 
    # https://datascience.stackexchange.com/questions/39932/feature-scaling-both-training-and-test-data
    scalable_columns = [col_name for col_name in df_X.columns if df_X[col_name].dtype != "object"]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df_X[scalable_columns])

    df_X_scaled = df_X.copy()
    df_X_scaled = df_X_scaled.drop(scalable_columns, axis = 1)
    df_X_scaled_temp = DataFrame(scaler.transform(df_X[scalable_columns]), columns = scalable_columns)
    df_X_scaled = df_X_scaled.join(df_X_scaled_temp)


    # one hot encoding for svm and nn

    # split data