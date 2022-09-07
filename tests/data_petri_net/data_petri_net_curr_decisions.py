from exdpn.data_petri_net import Data_Petri_Net
from exdpn.util import import_log
from exdpn.guards import ML_Technique

import unittest
import os


def get_p2p_event_log():
    return import_log(os.path.join(
        os.getcwd(), 'datasets', 'p2p_base.xes'))


def get_p2p_event_log_unfit():
    return import_log(os.path.join(
        os.getcwd(), 'datasets', 'p2p_base_unfit.xes'))


class Test_Data_Petri_Net_Current_Decision_Prediction(unittest.TestCase):
    def test_simple(self):
        event_log = get_p2p_event_log()
        event_log_unfit = get_p2p_event_log_unfit()

        dpn = Data_Petri_Net(event_log=event_log, event_level_attributes=[
                             'item_category', 'item_id', 'item_amount', 'supplier', 'total_price'], ml_list=[ML_Technique.SVM], verbose=False)
        preds = dpn.predict_current_decisions(event_log_unfit)

        self.assertEqual(str(list(preds.values())[0]["prediction"][0]), "(request standard approval, 'request standard approval')")


if __name__ == "__main__":
    unittest.main()
