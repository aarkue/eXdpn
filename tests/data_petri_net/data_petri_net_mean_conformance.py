from matplotlib.pyplot import eventplot
from exdpn.data_petri_net import Data_Petri_Net
from exdpn.load_event_log import import_xes

import unittest
import os


def get_p2p_event_log():
    return import_xes(os.path.join(
        os.getcwd(), 'datasets', 'p2p_base.xes'))
        #os.getcwd(), 'tests', 'guard_datasets', 'example.xes'))


class Test_Data_Petri_Net_Mean_Conformance(unittest.TestCase):
    def test_simple(self):
        event_log = get_p2p_event_log()
        dpn = Data_Petri_Net(event_log, case_level_attributes=['concept:name'], event_attributes=[
                             "item_id", "item_amount", "supplier", "item_category"], verbose=False)
                             #"costs"], verbose=False)   
        print(dpn.get_mean_guard_conformance(event_log))

        self.assertEqual(1,1)


if __name__ == "__main__":
    unittest.main()
