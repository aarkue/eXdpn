from exdpn.decisionpoints import find_decision_points
from exdpn.guard_datasets import get_all_guard_datasets
from exdpn.load_event_log import import_xes
from exdpn.petri_net import get_petri_net

import unittest
import os

from pm4py.objects.petri_net.obj import PetriNet
from pm4py.objects.log.obj import EventLog


# set up test by loading test event log and extracting test data
def get_net_and_log() -> tuple[PetriNet, PetriNet.Place, PetriNet.Place, EventLog, PetriNet.Place, PetriNet.Transition]:
    # TODO: decide on structure with regards to sample event logs
    # load test eventlog
    event_log = import_xes(os.path.join(
        os.getcwd(), 'tests', 'guard_datasets', 'example.xes'))

    # get petri net and find decision points 
    net, im, fm = get_petri_net(event_log)
    dp = find_decision_points(net)

    # this is ugly, we know
    # test place: p_3
    place_three = [place for place in dp.keys() if place.name == "p_3"][0]
    # test transition: B 
    transition_to_B = [transition for transition in dp[place_three] if transition.label == 'B'][0]

    return net, im, fm, event_log, place_three, transition_to_B

# test get_all_guard_datasets by comparing the output for a given place and transition with the original data
class TestGetAllGuardDatasets(unittest.TestCase):
    def test_simple(self):
        # get test example
        net, im, fm, event_log, place_three, transition_to_B = get_net_and_log()

        # use function to get guard dataset
        guard_datasets_per_place = get_all_guard_datasets(
            event_log, net, im, fm, event_attributes=["costs", "resource"])
        
        # extract dataframe for place p_3 and instances that passed transition B 
        df_place = guard_datasets_per_place[place_three]
        instances_where_transition_was_taken = df_place.loc[df_place["target"] == transition_to_B]

        # check if instances at place_three going to transition_to_B have the desired attributes
        # we use multiple assertions to do so but still consider this an "atomic" test, i.e., good granularity
        self.assertEqual(
            set(instances_where_transition_was_taken["event::resource"]), set(["Mike", "Pete"]))
        self.assertEqual(
            set(instances_where_transition_was_taken["event::costs"]), set([100]))
        self.assertEqual(len(instances_where_transition_was_taken), 2)

if __name__ == "__main__":
    unittest.main()