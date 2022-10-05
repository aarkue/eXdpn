import unittest
import os

from pm4py.objects.log.util.sampling import sample_log

from exdpn.data_petri_net import Data_Petri_Net
from exdpn.util import import_log
from exdpn.guards import ML_Technique

from tests import EVENT_LOG_SAMPLE_SIZE

def get_p2p_event_log():
    log = import_log(os.path.join(
        os.getcwd(), 'datasets', 'p2p_base.xes'))
    return sample_log(log, no_traces=EVENT_LOG_SAMPLE_SIZE)


class Test_Data_Petri_Net_Mean_Conformance(unittest.TestCase):
    def test_simple(self):
        event_log = get_p2p_event_log()
        dpn_dt = Data_Petri_Net(event_log, case_level_attributes=['concept:name'], event_level_attributes=[
                             "total_price", "item_id", "item_amount", "supplier", "item_category"], ml_list=[ML_Technique.DT], verbose=False)

        dpn_bad_dt = Data_Petri_Net(event_log, case_level_attributes=['concept:name'], verbose=False)

        self.assertLessEqual(dpn_bad_dt.get_mean_guard_conformance(event_log), dpn_dt.get_mean_guard_conformance(event_log))


        dpn_svm = Data_Petri_Net(event_log, case_level_attributes=['concept:name'], event_level_attributes=[
                             "total_price", "item_id", "item_amount", "supplier", "item_category"], ml_list=[ML_Technique.SVM], verbose=False)

        dpn_bad_svm = Data_Petri_Net(event_log, case_level_attributes=['concept:name'], verbose=False)

        self.assertLessEqual(dpn_bad_svm.get_mean_guard_conformance(event_log), dpn_svm.get_mean_guard_conformance(event_log))

        
        dpn_nn = Data_Petri_Net(event_log, event_level_attributes=[
                             "total_price", "item_id", "item_amount", "supplier", "item_category"], ml_list=[ML_Technique.NN], verbose=False)

        dpn_bad_nn = Data_Petri_Net(event_log, event_level_attributes=[
                             "item_amount", "supplier"], verbose=False)

        self.assertLessEqual(dpn_bad_nn.get_mean_guard_conformance(event_log), dpn_nn.get_mean_guard_conformance(event_log))

        
        dpn_lr = Data_Petri_Net(event_log, case_level_attributes=['concept:name'], event_level_attributes=[
                             "total_price", "item_id", "item_amount", "supplier", "item_category"], ml_list=[ML_Technique.LR], verbose=False)

        dpn_bad_lr = Data_Petri_Net(event_log, case_level_attributes=['concept:name'], verbose=False)

        self.assertLessEqual(dpn_bad_lr.get_mean_guard_conformance(event_log), dpn_lr.get_mean_guard_conformance(event_log))
        


if __name__ == "__main__":
    unittest.main()
