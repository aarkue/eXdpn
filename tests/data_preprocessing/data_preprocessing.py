import unittest
from pandas import DataFrame
import os 
from exdpn import petri_net
from exdpn import load_event_log
from exdpn import guard_datasets
from exdpn.data_preprocessing.data_preprocessing import apply_ohe, data_preprocessing_evaluation, apply_scaling, fit_ohe, fit_scaling
import random 
import pm4py 

# set up test by loading a test dataframe and perform some preprocessing 
def preprocess_data() -> tuple[DataFrame]:

    event_log = load_event_log.import_xes(os.path.join(os.getcwd(), 'example.xes'))
    net, im, fm = petri_net.get_petri_net(event_log)
    guard_datasets_per_place = guard_datasets.get_all_guard_datasets(event_log, net, im, fm, event_attributes=pm4py.get_event_attributes(event_log))

    # use data set of decision point p_3
    place_three = [place for place in guard_datasets_per_place.keys() if place.name == "p_3"][0]
    df = guard_datasets_per_place.get(place_three)

    # drop unnedded columns 
    df = df.drop(["event::time:timestamp"], axis = 1) 
    df = df.drop(["event::@@index"], axis = 1) 
    df = df.drop(["event::timestamp"], axis = 1) 
    df = df.drop(["event::case_id"], axis = 1) 

    # preprocessing 
    random.seed(42)

    df_X_train, df_X_test, df_y_train, df_y_test = data_preprocessing_evaluation(df)

    # scaling
    scaler, scalable_columns = fit_scaling(df_X_train)
    df_X_train_scaled = apply_scaling(df_X_train, scaler, scalable_columns)
    df_X_test_scaled = apply_scaling(df_X_test, scaler, scalable_columns)


    # one hot encoding
    ohe, ohe_columns = fit_ohe(df_X_train_scaled)
    df_X_train_scaled_ohe = apply_ohe(df_X_train_scaled, ohe) 
    df_X_test_scaled_ohe = apply_ohe(df_X_test_scaled, ohe) 

    return df_X_train_scaled_ohe, df_X_test_scaled_ohe

class TestDataPreprocessing(unittest.TestCase):
    def test_simple_preprocessing(self):
        df_X_train_scaled_ohe, df_X_test_scaled_ohe = preprocess_data()
        
        # check if all nans droped
        self.assertEqual(df_X_train_scaled_ohe.shape, df_X_train_scaled_ohe.dropna(how = 'all', axis = 1).shape)

        # check normalization 
        self.assertEqual(df_X_train_scaled_ohe.max().max(), 1)
        self.assertEqual(df_X_train_scaled_ohe.min().min(), 0)

        # check if training and test data frame have same columns after one hot encoding 
        self.assertEqual(df_X_train_scaled_ohe.columns.all(), df_X_test_scaled_ohe.columns.all())

if __name__ == "__main__":
    unittest.main()