import unittest
import os

from pm4py.objects.log.util.sampling import sample_log

from exdpn.decisionpoints import find_decision_points
from exdpn.guard_datasets import extract_all_datasets
from exdpn.util import import_log
from exdpn.petri_net import get_petri_net
from exdpn.guards import Guard_Manager,  ML_Technique

from tests import EVENT_LOG_SAMPLE_SIZE

def get_df():
    event_log = import_log(os.path.join(
        os.getcwd(), 'datasets', 'p2p_base.xes'))
    event_log = sample_log(event_log, no_traces=EVENT_LOG_SAMPLE_SIZE)

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
        gm = Guard_Manager(df_place, [ML_Technique.XGB]) # Create Guard manager using only XGBoost
        _ = gm.train_test()
        technique, _ = gm.get_best()
        self.assertEqual(technique, ML_Technique.XGB,
                         "ML technique should be equal to XGB") # The best technique should obviously be XGBoost


if __name__ == "__main__":
    unittest.main()
