"""
.. include:: ./../../docs/_templates/md/data_petri_net/data_petri_net.md

"""

from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.log.obj import EventLog
import pm4py.util.xes_constants as xes
from pm4py.statistics.attributes.log.get import get_trace_attribute_values

from typing import Dict, Iterator, List, Any, Tuple
from pandas import DataFrame
from matplotlib.figure import Figure

from exdpn.data_preprocessing.data_preprocessing import basic_data_preprocessing
from exdpn.decisionpoints import find_decision_points
from exdpn.guard_datasets import extract_all_datasets, extract_current_decisions
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
                 ml_list: List[ML_Technique] = [ML_Technique.DT,
                                                ML_Technique.LR,
                                                ML_Technique.SVM,
                                                ML_Technique.NN,
                                                ML_Technique.XGB,
                                                ML_Technique.RF],
                 hyperparameters: Dict[ML_Technique, Dict[str, Any]] = {ML_Technique.NN: {'max_iter': 20000, 'learning_rate_init': 0.0001, 'hidden_layer_sizes': (10, 5), 'alpha': 0.0001},
                                                                        ML_Technique.DT: {'min_impurity_decrease': 0.0075},
                                                                        ML_Technique.LR: {'C': 0.1375, 'tol': 0.001},
                                                                        ML_Technique.SVM: {'C': 0.3, 'tol': 0.001},
                                                                        ML_Technique.XGB: {'max_depth': 2, 'n_estimators': 50},
                                                                        ML_Technique.RF: {'max_depth': 5}},
                 CV_splits: int = 5,
                 CV_shuffle: bool = False,
                 random_state: int = None,
                 guard_threshold: float = 0.0,
                 impute: bool = False,
                 verbose: bool = True) -> None:
        """Initializes a data Petri net based on the event log provided.

        Args:
            event_log (EventLog): The event log to be used as a basis for the data Petri net.
            petri_net (PetriNet, optional): The Petri net corresponding to the event log. If not supplied, a Petri net is mined from `event_log`.
            initial_marking (Marking, optional): The initial marking of the Petri net corresponding to the event log. Does not have to be supplied.
            final_marking (Marking, optional): The final marking of the Petri net corresponding to the event log. Does not have to be supplied.
            miner_type (str, optional): Specifies the type of Petri net mining algorithm to be used when `petri_net` is `None`. \
                Either inductive miner ("IM") or alpha miner ("AM", default).
            case_level_attributes (List[str], optional): The attribute list on the level of cases to be considered for each instance in the datasets.
            event_level_attributes (List[str], optional): The attribute list on the level of events to be considered for each instance in the datasets.
            tail_length (int, optional): The number of preceding events to record. Defaults to 3.
            activityName_key (str, optional): The key of the activity name in the event log. Defaults to `pm4py.util.xes_constants.DEFAULT_NAME_KEY` ("concept:name").
            ml_list (List[ML_Technique], optional): The list of all machine learning techniques that should be evaluated. Default includes all \
                implemented techniques.
            hyperparameters (Dict[ML_Technique, Dict[str, Any]], optional): The hyperparameters that should be used for the machine learning techniques. \
                If not specified, standard/generic parameters are used.
            CV_splits (int): Number of folds to use in stratified corss-validation, defaults to 5.
            CV_shuffle (bool): Shuffle samples before splitting, defaults to False. 
            random_state (int, optional): The random state to be used for algorithms wherever possible. Defaults to None.
            guard_threshold (float, optional): The performance threshold (between 0 and 1) that determines if a guard is added to the data Petri net or not. If the guard performance \
                is smaller than the threshold the guard is not added (see `exdpn.guards.guard_manager.Guard_Manager.train_test`). Default is 0. 
            impute (bool): If `True`, missing attribute values in the guard datasets will be imputed using constants and an indicator columns will be added. Default is `False`.
            verbose (bool, optional): Specifies if the execution should print status-esque messages or not.

        Examples:
            Use an event log to mine a Petri net based on it:
            >>> from exdpn.util import import_log
            >>> from exdpn.data_petri_net import Data_Petri_Net
            >>> from exdpn.guards import ML_Technique
            >>> event_log = import_log('./datasets/p2p_base.xes')
            >>> dpn = Data_Petri_Net(event_log = event_log,
            ...                      event_level_attributes = ['item_category','item_id','item_amount','supplier','total_price'],
            ...                      ml_list = [ML_Technique.SVM, ML_Technique.DT],
            ...                      verbose = False)

            Providing an already mined Petri net:
            >>> from exdpn.util import import_log
            >>> from exdpn.data_petri_net import Data_Petri_Net
            >>> from exdpn.petri_net import get_petri_net
            >>> event_log = import_log('./datasets/p2p_base.xes')
            >>> net, im, fm = get_petri_net(event_log)
            >>> dpn = Data_Petri_Net(event_log = event_log, 
            ...                      petri_net = net,
            ...                      initial_marking = im,
            ...                      final_marking = fm,
            ...                      event_level_attributes = ['item_category','item_id','item_amount','supplier','total_price'],
            ...                      verbose = False)

            Customize a data Petri net with personal hyperparameters and a guard threshold:
            >>> from exdpn.util import import_log
            >>> from exdpn.data_petri_net import Data_Petri_Net
            >>> from exdpn.guards import ML_Technique
            >>> event_log = import_log('./datasets/p2p_base.xes')
            >>> dpn = Data_Petri_Net(event_log = event_log,
            ...                      event_level_attributes = ['item_category','item_id','item_amount','supplier','total_price'],
            ...                      ml_list = [ML_Technique.SVM, ML_Technique.DT],
            ...                      hyperparameters = {ML_Technique.SVM: {"C": 0.5}, ML_Technique.DT: {'max_depth': 2}},
            ...                      guard_threshold = 0.7,
            ...                      verbose = False)

            .. include:: ../../docs/_templates/md/example-end.md

        Note: 
            For a full example please check [Data Petri Net Example](https://github.com/aarkue/eXdpn/blob/main/docs/dpn_example.ipynb).

        """
        self.verbose = verbose
        self.ml_list = ml_list
        if petri_net is None or initial_marking is None or final_marking is None:
            self.petri_net, self.im, self.fm = get_petri_net(
                event_log, miner_type)
        else:
            self.petri_net = petri_net
            self.im = initial_marking
            self.fm = final_marking

        self.decision_points = find_decision_points(self.petri_net)
        self._print_if_verbose("-> Mining guard datasets... ", end="")
        self.guard_ds_per_place = extract_all_datasets(
            event_log, self.petri_net, self.im, self.fm, case_level_attributes, event_level_attributes, tail_length, activityName_key
        )
        self._print_if_verbose("Done")

        self.case_level_attributes = case_level_attributes
        self.event_level_attributes = event_level_attributes
        self.tail_length = tail_length
        self.activityName_key = activityName_key
        self.random_state = random_state

        # initialize all gms
        self.guard_manager_per_place = {place: Guard_Manager(self.guard_ds_per_place[place],
                                                             ml_list=ml_list,
                                                             hyperparameters=hyperparameters,
                                                             CV_splits=CV_splits,
                                                             CV_shuffle=CV_shuffle,
                                                             random_state=self.random_state,
                                                             impute=impute)
                                        for place in self.decision_points.keys()}

        # evaluate all guards for all guard managers
        for place, guard_manager in self.guard_manager_per_place.items():
            self._print_if_verbose(
                f"-> Evaluating guards at decision point '{place.name}'... ", end=''
            )
            guard_manager.train_test()
            self._print_if_verbose("Done")

        self.guard_per_place = None
        self.ml_technique_per_place = {}
        self.performance_per_place = {}
        self.guard_threshold = guard_threshold
        self.impute = impute

    def _print_if_verbose(self, string: str, end: str = '\n'):
        """Internal method used as a shortcut for printing messages only if self.verbose is set to True."""
        if self.verbose:
            print(string, end=end)

    def get_best(self) -> Dict[PetriNet.Place, Guard]:
        """Returns the best guard for each decision point in the data Petri net.

        Returns:
            Dict[PetriNet.Place, Guard]: The best performing guard for each decision point with respect to the F1-score.


        Examples:

            >>> from exdpn.util import import_log
            >>> from exdpn.data_petri_net import Data_Petri_Net
            >>> from exdpn.guards import ML_Technique
            >>> event_log = import_log('./datasets/p2p_base.xes')
            >>> dpn = Data_Petri_Net(event_log = event_log,
            ...                      event_level_attributes = ['item_category','item_id','item_amount','supplier','total_price'],
            ...                      ml_list = [ML_Technique.SVM, ML_Technique.DT],
            ...                      verbose = False)
            >>> best_guards = dpn.get_best()


            .. include:: ../../docs/_templates/md/example-end.md

        """

        self.guard_per_place = {}

        for place, guard_manager in self.guard_manager_per_place.items():
            ml_technique, guard = guard_manager.get_best()
            if max(guard_manager.f1_mean_test.values()) < self.guard_threshold:
                max_performance = max(guard_manager.guards_results.values())
                self._print_if_verbose(
                    f"-> Guard at decision point '{place.name}': was dropped because performance {max_performance} is below threshold {self.guard_threshold}")
                continue
            self.guard_per_place[place] = guard
            self.ml_technique_per_place[place] = ml_technique
            self.performance_per_place[place] = self.guard_manager_per_place[place].f1_mean_test[ml_technique]
            self._print_if_verbose(
                f"-> Best machine learning technique at decision point '{place.name}': {ml_technique} w/ performance {self.performance_per_place[place]}")

        return self.guard_per_place

    def get_guard_at_place(self, place: PetriNet.Place) -> Guard:
        """Returns the best guard for given decision point.

        Args:
            place (PetriNet.Place): The decision point to look up.

        Returns:
            Guard: The best guard at `place`.

        Examples:

            >>> from exdpn.util import import_log
            >>> from exdpn.data_petri_net import Data_Petri_Net
            >>> from exdpn.guards import ML_Technique
            >>> from exdpn.decisionpoints import find_decision_points
            >>> event_log = import_log('./datasets/p2p_base.xes')
            >>> dpn = Data_Petri_Net(event_log = event_log,
            ...                      event_level_attributes = ['item_category','item_id','item_amount','supplier','total_price'],
            ...                      ml_list = [ML_Technique.SVM, ML_Technique.DT],
            ...                      verbose = False)
            >>> all_decision_points = list(find_decision_points(dpn.petri_net).keys())
            >>> my_decision_point = all_decision_points[0]
            >>> my_guard = dpn.get_guard_at_place(my_decision_point)

            .. include:: ../../docs/_templates/md/example-end.md

        """
        if self.guard_per_place == None:
            self.get_best()

        return self.guard_per_place[place]

    def get_mean_guard_conformance(self, test_event_log: EventLog) -> float:
        """Returns the mean conformance for the given event log, i.e., the percentage of traces (which fit on the underlying Petri net) where all guards were respected. \
            Respecting a guard means moving from the corresponding place to the transition predicted by the guard.

        Args:
            test_event_log (EventLog): The event log used to test the performance of the data Perti net.

        Returns:
            float: Fraction of traces that respected all decision point guards passed during token based replay.

        Examples:

            >>> from exdpn.util import import_log
            >>> from exdpn.data_petri_net import Data_Petri_Net
            >>> from exdpn.guards import ML_Technique
            >>> event_log = import_log('./datasets/p2p_base.xes')        
            >>> dpn = Data_Petri_Net(event_log = event_log,
            ...                      event_level_attributes = ['item_category','item_id','item_amount','supplier','total_price'],
            ...                      ml_list = [ML_Technique.SVM, ML_Technique.DT],
            ...                      verbose = False)
            >>> dpn.get_mean_guard_conformance(event_log) # value may deviate
            0.949

            .. include:: ../../docs/_templates/md/example-end.md

        """
        if self.guard_per_place == None:
            self.get_best()

        self._print_if_verbose("-> Computing guard datasets for replay")
        guard_datasets = extract_all_datasets(
            test_event_log,
            self.petri_net, self.im, self.fm,
            self.case_level_attributes,
            self.event_level_attributes,
            self.tail_length,
            self.activityName_key
        )
        all_trace_ids: Dict[Any, int] = get_trace_attribute_values(
            test_event_log, xes.DEFAULT_TRACEID_KEY)

        # `prediction_result` keeps track for every case, if the case has fit on all guards thus far. As soon as a guard is violated, the cases entry in the dictionary is set to 0.
        prediction_result = {i: 1 for i in all_trace_ids}

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
            trace_ids = dp_dataset.index.get_level_values(
                xes.DEFAULT_TRACEID_KEY)  # preserves order, duplicates not deleted
            seen_trace_ids.update(trace_ids)
            X, y_raw = basic_data_preprocessing(dp_dataset, impute=self.impute)
            y = y_raw.tolist()

            # Check if prediction is correct.
            prediction = self.guard_per_place[decision_point].predict(X)
            for caseid, pred, target in zip(trace_ids, prediction, y):
                if pred != target:
                    prediction_result[caseid] = 0

        return sum([prediction_result[trace_id] for trace_id in seen_trace_ids]) / len(seen_trace_ids)

    def predict_current_decisions(self, log: EventLog) -> Dict[PetriNet.Place, DataFrame]:
        """Returns a dictionary mapping places to a current decision and their next-transition-predictions for the given event log. \
            Current decisions of an unfit trace arise at those places which have enabled transitions in the token based replay-marking. \
            The current decisions of an event log are all current decisons of unfit traces with respect to token based replay (see `exdpn.guard_datasets.extract_current_decisions`).

        Args:
            log (EventLog): The event log used to compute current decisions.

        Returns:
            Dict[PetriNet.Place, DataFrame]: A mapping of decision points to current decision instances and their predicted next transition.

        Examples:

            >>> from exdpn.util import import_log
            >>> from exdpn.data_petri_net import Data_Petri_Net
            >>> from exdpn.guards import ML_Technique
            >>> event_log = import_log('./datasets/p2p_base.xes')
            >>> event_log_unfit = import_log('./datasets/p2p_base_unfit.xes')        
            >>> dpn = Data_Petri_Net(event_log = event_log,
            ...                      event_level_attributes = ['item_category','item_id','item_amount','supplier','total_price'],
            ...                      ml_list = [ML_Technique.SVM],
            ...                      verbose = False)
            >>> preds = dpn.predict_current_decisions(event_log_unfit)
            >>> list(preds.values())[0]["prediction"][0]
            (request standard approval, 'request standard approval')

            .. include:: ../../docs/_templates/md/example-end.md

        """
        current_decisions = extract_current_decisions(
            log,
            self.petri_net, self.im, self.fm,
            self.case_level_attributes, self.event_level_attributes,
            self.tail_length, self.activityName_key
        )

        for place, decision_dataset in current_decisions.items():
            if len(decision_dataset) == 0:
                continue

            X, _ = basic_data_preprocessing(
                decision_dataset, impute=self.impute)
            decision_dataset.drop(['target'], axis=1, inplace=True)
            guard = self.get_guard_at_place(place)
            try:
                predicted_transitions = guard.predict(X)
            except:
                self._print_if_verbose(
                    f'Prediction of transitions following place {place} failed')
            else:
                decision_dataset["prediction"] = predicted_transitions

        return current_decisions

    def explain_current_decision_predictions_for_trace(
        self,
        curr_decision_preds: Dict[PetriNet.Place, DataFrame],
        trace_id: str,
        base_sample_size: int = 10
    ) -> Iterator[Tuple[PetriNet.Place, PetriNet.Transition, Dict[str, Figure]]]:
        """Yields pairs of the form: (decision point, predicted next transition, local explanation) for the given current decision pedictions.

        Args:
            curr_decision_preds (Dict[PetriNet.Place, DataFrame]): A mapping of decision points to current decision instances and their predicted next transition. \
                (see `exdpn.data_petri_net.Data_Petri_Net.predict_current_decisions`)
            trace_id (str): The trace id of the trace for which the local explanations are computed.
            base_sample_size (int, optional): The number of instances used to compute the local explanations. Defaults to 10.

        Yields:
            Tuple[PetriNet.Place, PetriNet.Transition, Dict[str, Figure]]: A decision point, predicted next transition, local explanation pair.

        """
        explanations_todo = []

        for place, predictions in curr_decision_preds.items():
            for prediction_instance in predictions.loc[[trace_id]].iterrows():
                explanations_todo.append((
                    place,
                    prediction_instance[1]["prediction"],
                    DataFrame([prediction_instance[1]]).drop("prediction", axis=1)))

        for (place, pred, prediction_instance) in explanations_todo:
            guard = self.get_guard_at_place(place)
            gm = self.guard_manager_per_place[place]
            yield (
                place,
                pred,
                guard.get_local_explanations(prediction_instance, gm.df_X.sample(base_sample_size, replace=True)))


# tests implemented examples
if __name__ == "__main__":
    import doctest
    doctest.testmod()
# run python .\exdpn\data_petri_net\data_petri_net.py from eXdpn file
