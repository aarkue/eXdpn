"""
.. include:: ./data_petri_net.md

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
            verbose (bool): Specifies if the execution of all methods should print status-esque messages or not"""
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
        """
        
        if self.guard_per_place == None:
            self.get_best()

        return self.guard_per_place[place]


    def get_mean_guard_conformance(self, test_event_log: EventLog) -> float:
        """Returns the mean conformance (percentage of traces where all guards were respected) for the given event log. \
            Note that the XES defined standard case identifier attribute must be present in the event log.

        Args:
            test_event_log (EventLog): The event log used to test the performance of the data Perti net

        Returns:
            float: Fraction of traces that respected all decision points passed during token based replay. \
                Respecting a decision point means moving to the transition predicted by the guard at the corresponding place
        """
        
        if self.guard_per_place == None:
            self.get_best()

        # pm4py.statistics.attributes.log.get.get_all_trace_attributes_from_log can't be used since it removes xes.DEFAULT_TRACEID_KEY ...
        assert xes.DEFAULT_TRACEID_KEY in test_event_log[0].attributes.keys(), \
            f"Error: case identifier missing. Expected case level attribute with name '{xes.DEFAULT_TRACEID_KEY}' to be present"

        case_level_attribute_set = set(self.case_level_attributes)
        case_level_attribute_set.add(xes.DEFAULT_TRACEID_KEY)

        self.print_if_verbose("-> Computing guard datasets for replay")
        guard_datasets = extract_all_datasets(
                            test_event_log,
                            self.petri_net, self.im, self.fm,
                            list(case_level_attribute_set),
                            self.event_level_attributes,
                            self.tail_length,
                            self.activityName_key
                        )
        
        # initialize dict for results, assume all guards were respected for each trace
        all_trace_ids = list(get_trace_attribute_values(test_event_log, xes.DEFAULT_TRACEID_KEY).keys())
        prediction_result = {i: 1 for i in all_trace_ids}

        # seen trace ids might be different from all trace ids
        # since unfit traces are ignored and do not produce instances in the datasets.
        # the mean guard conformance metric must respect the potentially reduced number of traces
        # for which the guard conformance can be checked.
        seen_trace_ids = set()

        for decision_point, dp_dataset in guard_datasets.items():
            if len(dp_dataset) == 0:
                continue

            # ignore guard if not added to data petri net
            if decision_point not in self.guard_per_place.keys():
                continue
            # extract data for prediction 
            cols_to_keep = [col for col in dp_dataset.columns
                            if any(feature.startswith(col) for feature in self.guard_per_place[decision_point].feature_names)]
            trace_ids = dp_dataset[f'case::{xes.DEFAULT_TRACEID_KEY}']
            seen_trace_ids.update(trace_ids) # keep track of seen traces
            X_raw, y_raw = basic_data_preprocessing(dp_dataset)
            X = X_raw[cols_to_keep]
            y = list(y_raw)

            # check if prediction with current guard is correct 
            prediction = self.guard_per_place[decision_point].predict(X)
            for j in range(len(y)):
                if y[j] != prediction[j]:
                    prediction_result[trace_ids[j]] = 0

        return sum([prediction_result[trace_id] for trace_id in seen_trace_ids]) / len(seen_trace_ids)
