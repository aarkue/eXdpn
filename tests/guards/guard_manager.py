from exdpn.decisionpoints import find_decision_points
from exdpn.guard_datasets import extract_all_datasets
from exdpn.load_event_log import import_xes
from exdpn.petri_net import get_petri_net
from exdpn.guards import Guard_Manager
from exdpn.guards import ML_Technique

import unittest
import os


def get_df():
    event_log = import_xes(os.path.join(
        os.getcwd(), 'tests', 'guard_datasets', 'example.xes'))

    net, im, fm = get_petri_net(event_log)
    dp = find_decision_points(net)

    place_three = [place for place in dp.keys() if place.name == "p_3"][0]

    guard_datasets_per_place = extract_all_datasets(
        event_log, net, im, fm, event_level_attributes=["costs", "resource"], tail_length=2)

    df_place = guard_datasets_per_place[place_three]
    return df_place


class Test_Guard_Manager(unittest.TestCase):
    def test_simple(self):
        df_place = get_df()

        gm = Guard_Manager(df_place, [ML_Technique.DT])
        _ = gm.train_test()
        technique, guard = gm.get_best()
        _ = guard.get_explainable_representation()
        self.assertEqual(technique, ML_Technique.DT,
                         "ML technique should be equal to DT")

    def test_error_on_missing_eval(self):
        df_place = get_df()

        gm = Guard_Manager(df_place, [ML_Technique.DT])
        self.assertRaises(AssertionError, gm.get_best)


if __name__ == "__main__":
    unittest.main()
