from tabnanny import verbose
from exdpn.decisionpoints import find_decision_points
from exdpn.guard_datasets import extract_all_datasets
from exdpn.util import import_log
from exdpn.petri_net import get_petri_net
from exdpn.guards import Guard_Manager
from exdpn.guards import ML_Technique

import unittest
import os


def get_df():
    event_log = import_log(os.path.join(
        os.getcwd(), 'datasets', 'p2p_base.xes'))

    net, im, fm = get_petri_net(event_log)
    dp = find_decision_points(net)

    place = [place for place in dp.keys() if place.name.startswith(
             "({'create purchase order'}")][0]

    guard_datasets_per_place = extract_all_datasets(
        event_log, net, im, fm, event_level_attributes=["total_price", "item_id", "item_amount", "supplier", "item_category"], tail_length=2)

    df_place = guard_datasets_per_place[place]
    return df_place


class Test_Guard_Manager(unittest.TestCase):
    def test_simple(self):
        df_place = get_df()
        gm = Guard_Manager(df_place, [ML_Technique.XGB])
        _ = gm.train_test()
        technique, guard = gm.get_best()
        self.assertEqual(technique, ML_Technique.XGB,
                         "ML technique should be equal to XGB")


if __name__ == "__main__":
    unittest.main()
