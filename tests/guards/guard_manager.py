from exdpn.decisionpoints import find_decision_points
from exdpn.guard_datasets import extract_all_datasets
from exdpn.util import import_log
from exdpn.petri_net import get_petri_net
from exdpn.guards import Guard_Manager
from exdpn.guards import ML_Technique

from pm4py.objects.log.obj import EventLog, Trace

import unittest
import os


def get_df():
    event_log = import_log(os.path.join(
        os.getcwd(), 'tests', 'guard_datasets', 'example.xes'))

    net, im, fm = get_petri_net(event_log, "IM")
    dp = find_decision_points(net)

    place_three = [place for place in dp.keys() if place.name == "p_3"][0]

    guard_datasets_per_place = extract_all_datasets(
        event_log, net, im, fm, event_level_attributes=["costs", "resource"], tail_length=2)

    df_place = guard_datasets_per_place[place_three]
    df_place = df_place.append(df_place)
    return df_place

def get_dummy_log():
    abcd = Trace([{"concept:name": act,  "time:timestamp": i} for i, act in enumerate("abcd")], attributes={"concept:name": "abcd"})
    abce = Trace([{"concept:name": act,  "time:timestamp": i} for i, act in enumerate("abce")], attributes={"concept:name": "acbd"})
    log = EventLog([abcd,abce])

    return log

class Test_Guard_Manager(unittest.TestCase):
    def test_simple(self):
        df_place = get_df()

        gm = Guard_Manager(df_place, [ML_Technique.DT], CV_splits=2)
        _ = gm.train_test()
        technique, guard = gm.get_best()
        _ = guard.get_explainable_representation()
        self.assertEqual(technique, ML_Technique.DT,
                         "ML technique should be equal to DT")

    def test_error_on_missing_eval(self):
        df_place = get_df()

        gm = Guard_Manager(df_place, [ML_Technique.DT], CV_splits=2)
        self.assertRaises(AssertionError, gm.get_best)


    def test_all_techniques_used(self):
        """Ensure that the default ml_list contains every ML Technique"""
        log = get_dummy_log()
        net, im, fm = get_petri_net(log, "IM")
        dfs = extract_all_datasets(log,net,im,fm)

        for place in find_decision_points(net).keys():
            gm = Guard_Manager(dfs[place])
            self.assertEqual(
                set(gm.guards_list.keys()),
                {technique for technique in ML_Technique},
                "The default ml_list of a Guard Manager should contain all ML Techniques!"
            )



if __name__ == "__main__":
    unittest.main()
