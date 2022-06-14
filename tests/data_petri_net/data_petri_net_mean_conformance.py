from exdpn.data_petri_net import Data_Petri_Net
from exdpn.util import import_log
from exdpn.guards import ML_Technique

import unittest
import os


def get_p2p_event_log():
    return import_log(os.path.join(
        os.getcwd(), 'datasets', 'p2p_base.xes'))


class Test_Data_Petri_Net_Mean_Conformance(unittest.TestCase):
    def test_simple(self):
        event_log = get_p2p_event_log()
        dpn = Data_Petri_Net(event_log, case_level_attributes=['concept:name'], event_level_attributes=[
                             "total_price", "item_id", "item_amount", "supplier", "item_category"], ml_list=[ML_Technique.DT, ML_Technique.LR], verbose=False)

        dpn_bad = Data_Petri_Net(event_log, case_level_attributes=['concept:name'], event_level_attributes=[
                             "item_amount", "supplier"], ml_list=[ML_Technique.DT, ML_Technique.LR], verbose=False)

        self.assertLessEqual(dpn_bad.get_mean_guard_conformance(event_log), dpn.get_mean_guard_conformance(event_log))


if __name__ == "__main__":
    unittest.main()
