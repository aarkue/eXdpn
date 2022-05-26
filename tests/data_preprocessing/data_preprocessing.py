import unittest
from pandas import DataFrame
import pandas as pd 
from exdpn.data_preprocessing import data_preprocessing
from exdpn.data_preprocessing import fit_apply_ohe


# set up test by loading a test dataframe and perform some preprocessing 
def preprocess_data() -> tuple[DataFrame]:
    # load test data frame
    data = pd.read_csv("TestDataFrame.csv")

    # preprocessing and one hot encoding
    X_train, X_test, y_train, y_test = data_preprocessing(data)

    return X_train, X_test, y_train, y_test

def ohc_data(X: DataFrame) -> DataFrame:
    X_ohc = fit_apply_ohe(X)

    return X_ohc

class TestDataPreprocessing(unittest.TestCase):
    def test_simple_preprocessing(self):
        X_train, X_test, y_train, y_test = preprocess_data()
        X_train = ohc_data(X_train)
        X_test = ohc_data(X_test)
        
        # check if all nans droped
        self.assertEqual(X_train.shape, X_train.dropna(how = 'all', axis = 1))

        # check for categorical data after ohe
        self.assertEqual(ohc_data(X_train), "Data does not contain categorical data, no One Hot Encoding is necessary")

        # check normalization
        # TODO


if __name__ == "__main__":
    unittest.main()