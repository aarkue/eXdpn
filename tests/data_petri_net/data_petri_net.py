import unittest
import os

from pm4py.objects.log.obj import EventLog, Trace
from pm4py.objects.log.util.sampling import sample_log

from exdpn.data_petri_net import Data_Petri_Net
from exdpn.util import import_log
from exdpn.guards import ML_Technique




from tests import EVENT_LOG_SAMPLE_SIZE

def get_p2p_event_log():
    log = import_log(os.path.join(
        os.getcwd(), 'datasets', 'p2p_base.xes'))
    return sample_log(log, no_traces=EVENT_LOG_SAMPLE_SIZE)

def get_dummy_log():
    events = [{"case:concept:name": 1337, "concept:name": act,  "time:timestamp": i} for i, act in enumerate("abcd")]
    trace = Trace(events)
    log = EventLog([trace])

    return log


class Test_Data_Petri_Net(unittest.TestCase):
    def test_simple(self):
        # "Checks" that no exceptions are raised
        event_log = get_p2p_event_log()
        dpn = Data_Petri_Net(event_log, case_level_attributes=['concept:name'], event_level_attributes=[
                             "total_price", "item_id", "item_amount", "supplier", "item_category"], verbose=False)

        self.assertEqual(1,1)

    def test_all_techniques_used(self):
        """Ensure that the default ml_list contains every ML Technique"""
        log = get_dummy_log()
        dpn = Data_Petri_Net(log, miner_type="AM")

        # Use set to ignore ordering
        self.assertEqual(
            set(dpn.ml_list),
            {technique for technique in ML_Technique},
            "The default ml_list of a Data Petri Net should contain all ML Techniques!"
        )



if __name__ == "__main__":
    unittest.main()
