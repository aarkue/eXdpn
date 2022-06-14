from exdpn.data_preprocessing.data_preprocessing import basic_data_preprocessing
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.log.obj import EventLog
import pm4py.util.xes_constants as xes

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
                 case_level_attributes: List[str] = [],
                 event_level_attributes: List[str] = [],
                 tail_length: int = 3,
                 activityName_key: str = xes.DEFAULT_NAME_KEY,
                 ml_list: List[ML_Technique] = [
                    ML_Technique.DT,
                    ML_Technique.LR,
                    ML_Technique.SVM,
                    ML_Technique.NN
                ],
                 hyperparameters: Dict[ML_Technique, Dict[str, Any]] = {ML_Technique.NN: {'hidden_layer_sizes': (10, 10)},
                                                                        ML_Technique.DT: {'min_samples_split': 0.1, 
                                                                                          'min_samples_leaf': 0.1, 
                                                                                          'ccp_alpha': 0.2},
                                                                        ML_Technique.LR: {"C": 0.5},
                                                                        ML_Technique.SVM: {"C": 0.5}},
                 guard_threshold: float = 0.6,
                 verbose: bool = True) -> None:
        """Initializes a data Petri net based on the event log provided.
        Args:
            event_log (EventLog): Event log to be used as a basis for the data Petri net
            petri_net (PetriNet): Petri net corresponding to the event log. Does not have to be supplied
            initial_marking (Marking): Initial marking of the Petri net corresponding to the event log. Does not have to be supplied
            final_marking (Marking): Final marking of the Petri net corresponding to the event log. Does not have to be supplied
            case_level_attributes (List[str]): Attribute list on the level of cases to be considered for each instance in the datasets
            event_level_attributes (List[str]): Attribute list on the level of events to be considered for each instance in the datasets
            tail_length (int): Number of events lookback to extract executed activity. Defaults to 3.
            activityName_key (str): Event level attribute name corresponding to the name of an event. Defaults to "concept:name"
            ml_list (List[ML_Technique]): List of all machine learning techniques that should be evaluated, default is all \
            implemented techniques
            hyperparameter (Dict[ML_Technique, Dict[str, Any]]): Hyperparameter that should be used for the machine learning techniques, \
            if not specified default parameters are used
            guard_threshold (float): Threshold (between 0 and 1) that determines if guard is added to the data petri net or not, if the guard performance \
            is smaller than the threshold the guard is not added. Default is 0.6 
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
                                                             hyperparameter = hyperparameter) 
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
        """ Returns "best" guard for each decision point in the data Petri net.
        Returns:
            best_guards (Dict[PetriNet.Place, Guard]): Best performing guard for each decision point with respect to the \
                chosen metric (F1 score)
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
            self.guard_per_place[place] = guard[1] # use model based on all data
            self.ml_technique_per_place[place] = ml_technique
            self.performance_per_place[place] = self.guard_manager_per_place[place].guards_results[ml_technique]
            self.print_if_verbose(
                f"-> Best machine learning technique at decision point '{place.name}': {ml_technique} w/ performance {self.performance_per_place[place]}")
            self.print_if_verbose(
                guard[0].get_explainable_representation()) # use "training" model for representation

        return self.guard_per_place


    def get_guard_at_place(self, place: PetriNet.Place) -> Guard:
        """ Returns "best" guard for given decision point.
        Args:
            place (PetriNet.Place): Decision point to be evaluated 
        Returns:
            guard (Guard): "Best" guard for given decision point
        """
        
        if self.guard_per_place == None:
            self.get_best()

        return self.guard_per_place[place]


    def get_mean_guard_conformance(self, test_event_log: EventLog) -> float:
        """Returns the mean conformance (percentage of traces where ALL guards were respected) from the given event log.
        Args:
            test_event_log (EventLog): The event log used to test the performance of the data Perti net
        Returns:
            mean conformace (float): Fraction of traces that respected ALL decision points passed during token based replay. \
                Respecting a decision point means moving to the transition predicted by the guard at the corresponding place
        """
        
        if self.guard_per_place == None:
            self.get_best()

        # remember which entry in a guard dataset belongs to which trace
        TRACE_NUMBER_ATTR_NAME = '__trace_number__'
        assert TRACE_NUMBER_ATTR_NAME not in self.case_level_attributes, \
            f"Error: case level attribute name '{TRACE_NUMBER_ATTR_NAME}' is reserved for internal purposes"

        for i in range(len(test_event_log)):
            test_event_log[i].attributes[TRACE_NUMBER_ATTR_NAME] = i

        self.print_if_verbose("-> Computing guard datasets for replay")
        guard_datasets = extract_all_datasets(
                            test_event_log,
                            self.petri_net, self.im, self.fm,
                            self.case_level_attributes +
                            [TRACE_NUMBER_ATTR_NAME],
                            self.event_level_attributes,
                            self.tail_length,
                            self.activityName_key
                        )
        
        # initialize dict for results, assume all guards were respected for each trace
        prediction_result = {i: 1 for i in range(len(test_event_log))}
        for decision_point, dp_dataset in guard_datasets.items():
            if len(dp_dataset) == 0:
                continue

            # ignore guard if not added to data petri net
            if decision_point not in self.guard_per_place.keys():
                continue
            # extract data for prediction 
            cols_to_keep = [col for col in dp_dataset.columns
                            if any(feature.startswith(col) for feature in self.guard_per_place[decision_point].feature_names)]
            trace_nums = dp_dataset[f'case::{TRACE_NUMBER_ATTR_NAME}']
            X_raw, y_raw = basic_data_preprocessing(dp_dataset)
            X = X_raw[cols_to_keep]
            y = list(y_raw)

            # check if prediction with current guard is correct 
            prediction = self.guard_per_place[decision_point].predict(X)
            for j in range(len(y)):
                if y[j] != prediction[j]:
                    prediction_result[trace_nums[j]] = 0
        
        return sum(list(prediction_result.values())) / len(test_event_log)
