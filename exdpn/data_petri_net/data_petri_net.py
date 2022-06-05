from pm4py.objects.petri_net.obj import PetriNet
from pm4py.objects.log.obj import EventLog
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from typing import Dict
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
                 petri_net: PetriNet = None,
                 initial_marking: PetriNet.Place = None,
                 final_marking: PetriNet.Place = None,
                 case_level_attributes: list[str] = [],
                 event_attributes: list[str] = [],
                 sliding_window_size: int = 3,
                 act_name_attr: str = "concept:name",
                 ml_list: list[ML_Technique] = [ML_Technique.DT],
                 verbose: bool = True) -> None:
        """Initializes a data Petri net based on the event log provided.
        Args:
            event_log (EventLog): Event log to be used as a basis for the data Petri net
            petri_net (PetriNet): Petri net corresponding to the event log. Does not have to be supplied
            initial_marking (PetriNet.Place): Initial marking of the Petri net corresponding to the event log. Does not have to be supplied
            final_marking (PetriNet.Place): Final marking of the Petri net corresponding to the event log. Does not have to be supplied
            case_level_attributes (list[str]): Attribute list on the level of cases to be considered for each instance in the datasets
            event_attributes (list[str]): Attribute list on the level of events to be considered for each instance in the datasets
            sliding_window_size (int): Size of the sliding window recording the last sliding_window_size events
            act_name_attr (str): Event level attribute name corresponding to the name of an event
            ml_list (list[ML_technique]): List of all machine learning techniques that should be evaluated, default is all \
                implemented techniques
            verbose (bool): Specifies if the execution of all methods should print status-esque messages or not"""
        self.verbose = verbose
        if petri_net == None or initial_marking == None or final_marking == None:
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
        self.sliding_window_size = sliding_window_size
        self.act_name_attr = act_name_attr
        # TODO: remove default values in get all guard ds

        # initialize all gms
        # TODO: add support for custom parameters per ml technique
        self.guard_manager_per_place = {place: Guard_Manager(
            self.guard_ds_per_place[place], ml_list=ml_list) for place in self.decision_points.keys()}

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
        """Internal method used as a shortcut for printing messages only if self.verbose is set to True"""
        if self.verbose:
            print(string, end=end)

    def get_best(self) -> Dict[PetriNet.Place, Guard]:
        """ Returns "best" guard for each decision point in the data Petri net
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
            self.guard_per_place[place] = guard
            self.ml_technique_per_place[place] = ml_technique
            self.performance_per_place[place] = self.guard_manager_per_place[place].guards_results[ml_technique]
            self.print_if_verbose(
                f"-> Best machine learning technique at decision point '{place.name}': {ml_technique.name} w/ performance {self.performance_per_place[place]}")
            self.print_if_verbose(
                guard.get_explainable_representation(), end="\n")

        return self.guard_per_place

    def get_guard_at_place(self, place: PetriNet.Place) -> Guard:
        if self.guard_per_place == None:
            self.get_best()

        return self.guard_per_place[place]

    def get_mean_guard_conformance(self, test_event_log: EventLog) -> float:
        """Returns the mean conformance (percentage of traces where ALL guards were respected) from the given event log.
        Args:
            test_event_log (EventLog): The event log used to test the performance of the data Perti net"""
        if self.guard_per_place == None:
            self.get_best()

        for i in range(len(test_event_log)):
            test_event_log[i].attributes['__trace_number__'] = i

        self.print_if_verbose("-> Computing guard datasets for replay")
        guard_datasets = get_all_guard_datasets(test_event_log,
                                                self.petri_net, self.im, self.fm,
                                                self.case_level_attributes +
                                                ['__trace_number__'],
                                                self.event_attributes,
                                                self.sliding_window_size,
                                                self.act_name_attr)
        # TODO: add support for bringing your own ds with you

        prediction_result = {dp: {i:1 for i in range(len(test_event_log))} for dp in guard_datasets.keys()}

        for decision_point, dp_dataset in guard_datasets.items():
            if len(dp_dataset) == 0:
                    continue

            cols_to_keep = [col for col in dp_dataset.columns
                            if any(feature.startswith(col) for feature in self.guard_per_place[decision_point].feature_names)]
            trace_nums = dp_dataset['case::__trace_number__']
            X = dp_dataset[cols_to_keep]
            y = list(dp_dataset["target"])

            prediction = self.guard_per_place[decision_point].predict(X)
    
            for j in range(len(y)):
                if y[j] != prediction[j]: 
                    prediction_result[decision_point][trace_nums[j]] = 0

        trace_performance = 0
        for i in tqdm(range(len(test_event_log))):
            conform = 1
            for conform_at_dp in prediction_result.values():
                if conform_at_dp[i] == 0:
                    conform = 0
                    break
            trace_performance = trace_performance + conform
            

        #trace_performance = 0

        #for i in tqdm(range(len(test_event_log))):
        #    conformance = 1 # no guards passed leads to a conformance value of 1
        #    for decision_point, dp_dataset in guard_datasets.items():
        #        dp_dataset = dp_dataset[dp_dataset['case::__trace_number__'] == i]

        #        if len(dp_dataset) == 0:
        #            continue

        #        cols_to_keep = [col for col in dp_dataset.columns
        #                        if any(feature.startswith(col) for feature in self.guard_per_place[decision_point].feature_names)]
        #        X = dp_dataset[cols_to_keep]
        #        y = list(dp_dataset["target"])

        #        respected = 0  # count the number of instances where this trace respected the guard
        #        prediction = self.guard_per_place[decision_point].predict(X)

        #        for j in range(len(y)):
        #            if y[j] == prediction[j]:
        #                respected = respected + 1

        #        conformance = conformance * \
        #            (respected / len(y))
        #    trace_performance = trace_performance + conformance

        return trace_performance / len(test_event_log)
