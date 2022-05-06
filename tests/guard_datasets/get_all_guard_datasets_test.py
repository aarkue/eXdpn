from exdpn.decisionpoints import find_decision_points
from exdpn.guard_datasets import get_all_guard_datasets
from exdpn.load_event_log import import_xes

import unittest
import os

from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.petri_net.obj import PetriNet
from pm4py.objects.log.obj import EventLog


class TestGetAllGuardDatasets(unittest.TestCase):
    def test_simple(self):
        net, im, fm, event_log, place_three, transition_to_B = get_net_and_log()

        guard_datasets_per_place = get_all_guard_datasets(
            event_log, net, im, fm, event_attributes=["costs", "resource"])
        df_place = guard_datasets_per_place[place_three]
        ins_where_transition_was_taken = df_place.loc[df_place["target"]
                                                      == transition_to_B]

        # check if instances at place_three going to transition_to_B have the desired attributes
        # we use multiple assertions to do so but still consider this an "atomic" test. I.e. good granularity
        self.assertEqual(
            set(ins_where_transition_was_taken["resource"]), set(["Mike", "Pete"]))
        self.assertEqual(
            set(ins_where_transition_was_taken["costs"]), set([100]))
        self.assertEqual(len(ins_where_transition_was_taken), 2)


def get_net_and_log() -> tuple[PetriNet, PetriNet.Place, PetriNet.Place, EventLog, PetriNet.Place, PetriNet.Transition]:
    # TODO: decide on structure with regards to sample event logs
    event_log = import_xes(os.path.join(
        os.getcwd(), 'tests', 'guard_datasets', 'example.xes'))

    # TODO: use internal function of mining Petri nets (see other branch)
    net, im, fm = inductive_miner.apply(event_log)

    dp = find_decision_points(net)

    # this is ugly. We know
    place_three = [place for place in dp.keys() if place.name == "p_3"][0]
    transition_to_B = [trans for trans in dp[place_three]
                       if trans.label == 'B'][0]

    return net, im, fm, event_log, place_three, transition_to_B


if __name__ == "__main__":
    unittest.main()
