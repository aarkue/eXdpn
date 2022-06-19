"""
.. include:: ./../../docs/_templates/md/data_petri_net/data_petri_net.md

"""

from exdpn.data_preprocessing.data_preprocessing import basic_data_preprocessing
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.log.obj import EventLog
import pm4py.util.xes_constants as xes
from pm4py.statistics.attributes.log.get import get_trace_attribute_values

from typing import Dict, List, Any

from exdpn.decisionpoints import find_decision_points
from exdpn.guard_datasets import extract_all_datasets
from exdpn.guards.guard import Guard
from exdpn.petri_net import get_petri_net
from exdpn.guards import Guard_Manager
from exdpn.guards import ML_Technique


class Data_Petri_Net():
    def __init__(self,
                 event_log: EventLog,
                 petri_net: PetriNet = None,
                 initial_marking: Marking = None,
                 final_marking: Marking = None,
                 miner_type: str = "AM",
                 case_level_attributes: List[str] = [],
                 event_level_attributes: List[str] = [],
                 tail_length: int = 3,
                 activityName_key: str = xes.DEFAULT_NAME_KEY,
                 ml_list: List[ML_Technique] = [
                    ML_Technique.DT,
                    ML_Technique.LR,
                    ML_Technique.SVM,
                    ML_Technique.NN],
                 hyperparameters: Dict[ML_Technique, Dict[str, Any]] = {ML_Technique.NN: {'hidden_layer_sizes': (10, 10)},
                                                                        ML_Technique.DT: {'min_samples_split': 0.1, 
                                                                                          'min_samples_leaf': 0.1, 
                                                                                          'ccp_alpha': 0.2},
                                                                        ML_Technique.LR: {"C": 0.5},
                                                                        ML_Technique.SVM: {"C": 0.5}},
                 guard_threshold: float = 0.0,
                 verbose: bool = True) -> None:
        """Initializes a data Petri net based on the event log provided.

        Args:
            event_log (EventLog): Event log to be used as a basis for the data Petri net
            petri_net (PetriNet): Petri net corresponding to the event log. Does not have to be supplied
            initial_marking (Marking): Initial marking of the Petri net corresponding to the event log. Does not have to be supplied
            final_marking (Marking): Final marking of the Petri net corresponding to the event log. Does not have to be supplied
            miner_type (str): Spezifies type of mining algorithm, either inductive miner ("IM") or alpha miner ("AM", default)
            case_level_attributes (List[str]): Attribute list on the level of cases to be considered for each instance in the datasets
            event_level_attributes (List[str]): Attribute list on the level of events to be considered for each instance in the datasets
            tail_length (int): Number of events lookback to extract executed activity. Defaults to 3.
            activityName_key (str): Event level attribute name corresponding to the name of an event. Defaults to "concept:name"
            ml_list (List[ML_Technique]): List of all machine learning techniques that should be evaluated, default is all \
            implemented techniques
            hyperparameters (Dict[ML_Technique, Dict[str, Any]]): Hyperparameter that should be used for the machine learning techniques, \
            if not specified default parameters are used
            guard_threshold (float): Threshold (between 0 and 1) that determines if guard is added to the data petri net or not, if the guard performance \
            is smaller than the threshold the guard is not added. Default is 0 (no threshold) 
            verbose (bool): Specifies if the execution of all methods should print status-esque messages or not
            
        Examples:
            Use event log and mine petri net based on it
            ```python 
            >>> from exdpn.util import import_log
            >>> from exdpn.data_petri_net import data_petri_net
            >>> from exdpn.guards import ML_Technique
            >>> event_log = import_log('p2p_base.xes')
            >>> dpn = data_petri_net.Data_Petri_Net(event_log = event_log, 
            ...                                     case_level_attributes = ["concept:name"],
            ...                                     event_level_attributes = ['item_category','item_id','item_amount','supplier','total_price'],
            ...                                     ml_list = [ML_Technique.SVM, ML_Technique.DT],
            ...                                     verbose = False)

            ``` 
            
            Use a mined petri net (based on event log)
            ```python
            >>> from exdpn.util import import_log
            >>> from exdpn.data_petri_net import data_petri_net
            >>> from exdpn.guards import ML_Technique
            >>> from exdpn import petri_net
            >>> event_log = import_log('p2p_base.xes')
            >>> net, im, fm = petri_net.get_petri_net(event_log)
            >>> dpn = data_petri_net.Data_Petri_Net(event_log = event_log, 
            ...                                     petri_net = net,
            ...                                     initial_marking = im,
            ...                                     final_marking = fm,
            ...                                     case_level_attributes = ["concept:name"],
            ...                                     event_level_attributes = ['item_category','item_id','item_amount','supplier','total_price'],
            ...                                     ml_list = [ML_Technique.SVM, ML_Technique.DT],
            ...                                     verbose = False)

            ```

            Costumize data petri net with personal hyperparameters and guard threshold
            ```python 
            >>> from exdpn.util import import_log
            >>> from exdpn.data_petri_net import data_petri_net
            >>> from exdpn.guards import ML_Technique
            >>> event_log = import_log('p2p_base.xes')
            >>> dpn = data_petri_net.Data_Petri_Net(event_log = event_log, 
            ...                                     case_level_attributes = ["concept:name"],
            ...                                     event_level_attributes = ['item_category','item_id','item_amount','supplier','total_price'],
            ...                                     ml_list = [ML_Technique.SVM, ML_Technique.DT],
            ...                                     hyperparameters = {ML_Technique.SVM: {"C": 0.5}, ML_Technique.DT: {'max_depth': 2}},
            ...                                     guard_threshold = 0.7,
            ...                                     verbose = False)

            ```
            
        """
        self.verbose = verbose
        if petri_net is None or initial_marking is None or final_marking is None:
            self.petri_net, self.im, self.fm = get_petri_net(event_log, miner_type)
        else:
            self.petri_net = petri_net
            self.im = initial_marking
            self.fm = final_marking

        self.decision_points = find_decision_points(self.petri_net)
        self.print_if_verbose("-> Mining guard datasets... ", end="")
        self.guard_ds_per_place = extract_all_datasets(
            event_log, self.petri_net, self.im, self.fm, case_level_attributes, event_level_attributes, tail_length, activityName_key
        )
        self.print_if_verbose("Done")

        self.case_level_attributes = case_level_attributes
        self.event_level_attributes = event_level_attributes
        self.tail_length = tail_length
        self.activityName_key = activityName_key

        # initialize all gms
        self.guard_manager_per_place = {place: Guard_Manager(self.guard_ds_per_place[place], 
                                                             ml_list = ml_list,
                                                             hyperparameters = hyperparameters) 
                                        for place in self.decision_points.keys()}

        # evaluate all guards for all guard managers
        for place, guard_manager in self.guard_manager_per_place.items():
            self.print_if_verbose(
                f"-> Evaluating guards at decision point '{place.name}'... ", end=''
            )
            guard_manager.train_test()
            self.print_if_verbose("Done")

        self.guard_per_place = None
        self.ml_technique_per_place = {}
        self.performance_per_place = {}
        self.guard_threshold = guard_threshold


    def print_if_verbose(self, string: str, end: str = '\n'):
        """Internal method used as a shortcut for printing messages only if self.verbose is set to True."""
        if self.verbose:
            print(string, end=end)

    def get_best(self) -> Dict[PetriNet.Place, Guard]:
        """Returns the best guard for each decision point in the data Petri net.

        Returns:
            Dict[PetriNet.Place, Guard]: The best performing guard for each decision point with respect to the F1-score
        
        Examples:
            ```python
            >>> from exdpn.util import import_log
            >>> from exdpn.data_petri_net import data_petri_net
            >>> from exdpn.guards import ML_Technique
            >>> event_log = import_log('p2p_base.xes')
            >>> dpn = data_petri_net.Data_Petri_Net(event_log = event_log, 
            ...                                     case_level_attributes = ["concept:name"],
            ...                                     event_level_attributes = ['item_category','item_id','item_amount','supplier','total_price'],
            ...                                     ml_list = [ML_Technique.SVM, ML_Technique.DT],
            ...                                     verbose = False)
            >>> best_guards = dpn.get_best()

            ``` 
        """
        
        if self.guard_per_place != None:
            return

        self.guard_per_place = {}

        for place, guard_manager in self.guard_manager_per_place.items():
            ml_technique, guard = guard_manager.get_best()
            if max(guard_manager.guards_results.values()) < self.guard_threshold:
                max_performance = max(guard_manager.guards_results.values())
                self.print_if_verbose(
                    f"-> Guard at decision point '{place.name}': was dropped because performance {max_performance} is below threshold {self.guard_threshold}")
                continue
            self.guard_per_place[place] = guard 
            self.ml_technique_per_place[place] = ml_technique
            self.performance_per_place[place] = self.guard_manager_per_place[place].guards_results[ml_technique]
            self.print_if_verbose(
                f"-> Best machine learning technique at decision point '{place.name}': {ml_technique} w/ performance {self.performance_per_place[place]}")

        return self.guard_per_place


    def get_guard_at_place(self, place: PetriNet.Place) -> Guard:
        """Returns the best guard for given decision point.

        Args:
            place (PetriNet.Place): The decision point to looked up

        Returns:
            Guard: The best guard for given decision point

        Examples:
            ```python
            >>> from exdpn.util import import_log
            >>> from exdpn.data_petri_net import data_petri_net
            >>> from exdpn.guards import ML_Technique
            >>> from exdpn.decisionpoints import find_decision_points
            >>> event_log = import_log('p2p_base.xes')
            >>> dpn = data_petri_net.Data_Petri_Net(event_log = event_log, 
            ...                                     case_level_attributes = ["concept:name"], 
            ...                                     event_level_attributes = ['item_category','item_id','item_amount','supplier','total_price'],
            ...                                     ml_list = [ML_Technique.SVM, ML_Technique.DT],
            ...                                     verbose = False)
            >>> all_decision_points = find_decision_points(dpn.petri_net).keys()
            >>> my_decision_point = list(all_decision_points)[0]
            >>> my_guard = dpn.get_guard_at_place(my_decision_point)

            ```
        """
        
        if self.guard_per_place == None:
            self.get_best()

        return self.guard_per_place[place]


    def get_mean_guard_conformance(self, test_event_log: EventLog) -> float:
        """Returns the mean conformance for the given event log, i.e., the percentage of traces (which fit on the mined model) where all guards were respected.

        Args:
            test_event_log (EventLog): The event log used to test the performance of the data Perti net

        Returns:
            float: Fraction of traces that respected all decision point guards passed during token based replay. \
                Respecting a decision point guard means moving to the transition predicted by the guard at the corresponding place
        
        Examples:
            ```python
            >>> from exdpn.util import import_log
            >>> from exdpn.data_petri_net import data_petri_net
            >>> from exdpn.guards import ML_Technique
            >>> #event_log = import_log('p2p_base.xes')
            >>> event_log = import_log(os.path.join(os.getcwd(), 'datasets', 'p2p_base.xes'))        
            >>> dpn = data_petri_net.Data_Petri_Net(event_log = event_log, 
            ...                                     case_level_attributes = ["concept:name"],
            ...                                     event_level_attributes = ['item_category','item_id','item_amount','supplier','total_price'],
            ...                                     ml_list = [ML_Technique.SVM, ML_Technique.DT],
            ...                                     verbose = False)
            >>> print("Mean guard conformance:", dpn.get_mean_guard_conformance(event_log))
            Mean guard conformance: 0.949
            
            ```
        """
        
        if self.guard_per_place == None:
            self.get_best()


        self.print_if_verbose("-> Computing guard datasets for replay")
        guard_datasets = extract_all_datasets(
                            test_event_log,
                            self.petri_net, self.im, self.fm,
                            self.case_level_attributes,
                            self.event_level_attributes,
                            self.tail_length,
                            self.activityName_key
                        )
        all_trace_ids: Dict[Any, int] = get_trace_attribute_values(test_event_log, xes.DEFAULT_TRACEID_KEY)

        # `prediction_result` keeps track for every case, if the case has fit on all guards thus far. As soon as a guard is violated, the cases entry in the dictionary is set to 0.
        prediction_result = {i: 1  for i in all_trace_ids}

        # Seen trace ids might be different from all trace ids
        # as unfit cases are ignored and do not produce instances in the datasets.
        # The mean-guard-conformance metric must respect the potentially reduced number of cases
        # for which the guard conformance can be checked.
        seen_trace_ids = set()
        for decision_point, dp_dataset in guard_datasets.items():
            if len(dp_dataset) == 0:
                continue
            # Ignore guard if not added to data petri net
            if decision_point not in self.guard_per_place.keys():
                continue

            # Extract data for prediction 
            trace_ids = dp_dataset.index.get_level_values(xes.DEFAULT_TRACEID_KEY) # preserves order, duplicates not deleted
            seen_trace_ids.update(trace_ids)
            X, y_raw = basic_data_preprocessing(dp_dataset)
            y = y_raw.tolist()


            # Check if prediction is correct.
            prediction = self.guard_per_place[decision_point].predict(X)
            for caseid, pred, target in zip(trace_ids, prediction, y):
                if pred != target:
                    prediction_result[caseid] = 0


        return sum([prediction_result[trace_id] for trace_id in seen_trace_ids]) / len(seen_trace_ids)


# tests implemented examples
if __name__ == "__main__":
    import doctest
    doctest.testmod()
# run python .\exdpn\data_petri_net\data_petri_net.py from eXdpn file 