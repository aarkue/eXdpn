from pandas import DataFrame
from exdpn.data_preprocessing.data_preprocessing import basic_data_preprocessing
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.log.obj import EventLog
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from typing import Dict, List 
from tqdm import tqdm


from exdpn.decisionpoints import find_decision_points
from exdpn.guard_datasets import get_all_guard_datasets
from exdpn.guards.guard import Guard
from exdpn.petri_net import get_petri_net
from exdpn.guards import Guard_Manager
from exdpn.guards import ML_Technique
from exdpn.guard_datasets import get_all_guard_datasets


class Data_Petri_Net():
    def __init__(self,
                 event_log: EventLog,
                 case_level_attributes: list[str],
                 event_attributes: list[str],
                 numeric_attributes: list[str] = [],
                 petri_net: PetriNet = None,
                 initial_marking: Marking = [],
                 final_marking: Marking = [],
                 sliding_window_size: int = 3,
                 act_name_attr: str = "concept:name",
                 ml_list: list[ML_Technique] = [ML_Technique.NN,
                                                ML_Technique.DT,
                                                ML_Technique.LR,
                                                ML_Technique.SVM],
                 verbose: bool = True) -> None:
        """Initializes a data Petri net based on the event log provided.
        Args:
            event_log (EventLog): Event log to be used as a basis for the data Petri net
            case_level_attributes (list[str]): Attribute list on the level of cases to be considered for each instance in the datasets
            event_attributes (list[str]): Attribute list on the level of events to be considered for each instance in the datasets
            numeric_attributes (list[str]): Attribute list to be converted to float, optional
            petri_net (PetriNet): Petri net corresponding to the event log. Does not have to be supplied
            initial_marking (PetriNet.Place): Initial marking of the Petri net corresponding to the event log. Does not have to be supplied
            final_marking (PetriNet.Place): Final marking of the Petri net corresponding to the event log. Does not have to be supplied
            sliding_window_size (int): Size of the sliding window recording the last sliding_window_size events, default is last 3 events
            act_name_attr (str): Event level attribute name corresponding to the name of an event
            ml_list (list[ML_technique]): List of all machine learning techniques that should be evaluated, default is all \
                implemented techniques
            verbose (bool): Specifies if the execution of all methods should print status-esque messages or not, default is true
        """
        
        self.verbose = verbose
        if petri_net == None or (len(initial_marking) == 0) or (len(final_marking) == 0):
            self.petri_net, self.im, self.fm = get_petri_net(event_log)
        else:
            self.petri_net = petri_net
            self.im = initial_marking
            self.fm = final_marking

        self.decision_points = find_decision_points(self.petri_net)
        self.print_if_verbose("-> Mining guard datasets... ", end="")
        self.guard_ds_per_place = get_all_guard_datasets(
            event_log, self.petri_net, self.im, self.fm, case_level_attributes, event_attributes, sliding_window_size, act_name_attr)
        self.print_if_verbose("Done")

        self.case_level_attributes = case_level_attributes
        self.event_attributes = event_attributes
        self.numeric_attributes = numeric_attributes
        self.sliding_window_size = sliding_window_size
        self.act_name_attr = act_name_attr
        # TODO: remove default values in get all guard ds -> DONE, also removed
        # the default value for ml_list in guard manager 

        # initialize all gms
        # TODO: add support for custom parameters per ml technique
        self.guard_manager_per_place = {place: Guard_Manager(
            self.guard_ds_per_place[place], numeric_attributes=numeric_attributes, ml_list=ml_list) for place in self.decision_points.keys()}

        # evaluate all guards for all guard managers
        for place, guard_manager in self.guard_manager_per_place.items():
            self.print_if_verbose(
                f"-> Evaluating guards at decision point '{place.name}'... ", end='')
            guard_manager.evaluate_guards()
            self.print_if_verbose("Done") 

        self.guard_per_place = None
        self.ml_technique_per_place = {}
        self.performance_per_place = {}


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
        
        # TODO: add support for taking no guard if the best performance is bad / below threshold
        if self.guard_per_place != None:
            return

        self.guard_per_place = {}

        for place, guard_manager in self.guard_manager_per_place.items():
            ml_technique, guard = guard_manager.get_best()
            self.guard_per_place[place] = guard[1] # use model based on all data
            self.ml_technique_per_place[place] = ml_technique
            self.performance_per_place[place] = self.guard_manager_per_place[place].guards_results[ml_technique]
            self.print_if_verbose(
                f"-> Best machine learning technique at decision point '{place.name}': {ml_technique.name} w/ performance {self.performance_per_place[place]}")
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
        guard_datasets = get_all_guard_datasets(test_event_log,
                                                self.petri_net, self.im, self.fm,
                                                self.case_level_attributes +
                                                [TRACE_NUMBER_ATTR_NAME],
                                                self.event_attributes,
                                                self.sliding_window_size,
                                                self.act_name_attr)
        
        # initialize dict for results, assume all guards were respected for each trace
        prediction_result = {i: 1 for i in range(len(test_event_log))}
        for decision_point, dp_dataset in guard_datasets.items():
            if len(dp_dataset) == 0:
                continue

            # extract data for prediction 
            cols_to_keep = [col for col in dp_dataset.columns
                            if any(feature.startswith(col) for feature in self.guard_per_place[decision_point].feature_names)]
            trace_nums = dp_dataset[f'case::{TRACE_NUMBER_ATTR_NAME}']
            X_raw, y_raw = basic_data_preprocessing(dp_dataset, self.numeric_attributes)
            X = X_raw[cols_to_keep]
            y = list(y_raw)

            # check if prediction with current guard is correct 
            prediction = self.guard_per_place[decision_point].predict(X)
            for j in range(len(y)):
                if y[j] != prediction[j]:
                    prediction_result[trace_nums[j]] = 0
        
        return sum(list(prediction_result.values())) / len(test_event_log)
