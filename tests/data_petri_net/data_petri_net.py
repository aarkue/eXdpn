from exdpn.data_petri_net import Data_Petri_Net
from exdpn.load_event_log import import_xes

import unittest
import os


def get_p2p_event_log():
    return import_xes(os.path.join(
        os.getcwd(), 'datasets', 'p2p_base.xes'))


class Test_Data_Petri_Net(unittest.TestCase):
    def test_simple(self):
        # "Checks" that no exceptions are raised
        event_log = get_p2p_event_log()
        dpn = Data_Petri_Net(event_log, case_level_attributes=['concept:name'], event_attributes=[
                             "total_price", "item_id", "item_amount", "supplier", "item_category"], verbose=False)

        self.assertEqual(1,1)


if __name__ == "__main__":
    unittest.main()
