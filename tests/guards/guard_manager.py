from exdpn.decisionpoints import find_decision_points
from exdpn.guard_datasets import get_all_guard_datasets
from exdpn.load_event_log import import_xes
from exdpn.petri_net import get_petri_net
from exdpn.guards import Guard_Manager
from exdpn.guards import ML_Technique

import unittest
import os

from pm4py.objects.petri_net.obj import PetriNet
from pm4py.objects.log.obj import EventLog


def get_df():
    event_log = import_xes(os.path.join(
        os.getcwd(), 'tests', 'guard_datasets', 'example.xes'))

    net, im, fm = get_petri_net(event_log)
    dp = find_decision_points(net)

    place_three = [place for place in dp.keys() if place.name == "p_3"][0]
    transition_to_B = [
        transition for transition in dp[place_three] if transition.label == 'B'][0]

    guard_datasets_per_place = get_all_guard_datasets(
        event_log, net, im, fm, event_attributes=["costs"], sliding_window_size=0)
    # TODO: Allow for 'resource' column by OHE nun numerical data
    # same goes for increasing the sliding window size

    df_place = guard_datasets_per_place[place_three]
    return df_place


class Test_Guard_Manager(unittest.TestCase):
    def test_simple(self):
        df_place = get_df()

        gm = Guard_Manager(df_place, [ML_Technique.DT])
        _ = gm.evaluate_guards()
        technique, guard = gm.get_best()
        _ = guard.get_explainable_representation()
        self.assertEqual(technique, ML_Technique.DT,
                         "ML technique should be equal to DT")


if __name__ == "__main__":
    unittest.main()
