"""
.. include:: ./../../docs/_templates/md/guards/guard_manager.md

"""

import warnings
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from pandas import DataFrame
import numpy as np 
import pandas as pd 

from typing import Dict, List, Tuple, Any, final
from exdpn import guards

from exdpn.guards import ML_Technique  # imports all guard classes
from exdpn.guards import Guard
from exdpn.data_preprocessing import basic_data_preprocessing
from exdpn.guards.model_builder import model_builder
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold


class Guard_Manager():
    def __init__(self,
                 dataframe: DataFrame,
                 ml_list: List[ML_Technique] = [ML_Technique.DT,
                                                ML_Technique.LR,
                                                ML_Technique.SVM,
                                                ML_Technique.NN],
                 hyperparameters: Dict[ML_Technique, Dict[str, Any]] = {ML_Technique.NN: {'hidden_layer_sizes': (10, 10)},
                                                                        ML_Technique.DT: {'min_samples_split': 0.1,
                                                                                          'min_samples_leaf': 0.1,
                                                                                          'ccp_alpha': 0.2},
                                                                        ML_Technique.LR: {"C": 0.5},
                                                                        ML_Technique.SVM: {"C": 0.5}},
                 CV_splits: int = 5,
                 impute: bool = False) -> None:
        """Initializes all information needed for the calculation of the best guard for each decision point and /
        returns a dictionary with the list of all guards for each machine learning technique.

        Args:
            dataframe (DataFrame): Dataset used to evaluate the guard.
            ml_list (List[ML_Technique]): List of all machine learning techniques that should be evaluated, default is all implemented.
            hyperparameters (Dict[ML_Technique, Dict[str, Any]]): Hyperparameters that should be used for the machine learning techniques, \
                if not specified, standard/generic parameters are used.
            CV_splits (int): Number of folds to use in stratified corss-validation, defaults to 5.
            impute (bool): If `True`, missing attribute values in the guard datasets will be imputed using constants and an indicator columns will be added. Default is `False`.

        Examples:
            
            >>> from exdpn.util import import_log
            >>> from exdpn.petri_net import get_petri_net
            >>> from exdpn.guard_datasets import extract_all_datasets
            >>> from exdpn import guards
            >>> event_log = import_log('./datasets/p2p_base.xes')
            >>> pn, im, fm = get_petri_net(event_log)
            >>> dp_dataset_map = extract_all_datasets(event_log, pn, im, fm,
            ...                                       event_level_attributes = ['item_category','item_id','item_amount','supplier','total_price'], 
            ...                                       activityName_key = "concept:name")
            >>> # select a certrain decision point and the corresponding data set 
            >>> dp = list(dp_dataset_map.keys())[0]
            >>> dp_dataset = dp_dataset_map[dp]
            >>> # create a guard manager for that decision point
            >>> guard_manager = guards.Guard_Manager(dataframe = dp_dataset)

            .. include:: ../../docs/_templates/md/example-end.md

        """
        df_X, df_y = basic_data_preprocessing(dataframe, impute=impute)
        self.df_X = df_X
        self.df_y = df_y
        self.hyperparameters = hyperparameters

        self.CV_splits = CV_splits 
        self.impute = impute

        # set up cross validation for model evaluation
        try:
            self.skf = StratifiedKFold(n_splits = CV_splits, shuffle = True, random_state = 2022)
            self.skf.get_n_splits(self.df_X, self.df_y)
        except TypeError:
            raise TypeError(
                "Invalid number of splits for cross validation.")

        # create list of all needed machine learning techniques to evaluate the guards
        self.guards_list = {technique: model_builder(
            technique, hyperparameters[technique]) for technique in ml_list}


    def train_test(self) -> Dict[str, Any]:
        """Calculates for a given decision point all selected guards and returns the precision of the machine learning model, \
            using the specified machine learning techniques.

        Returns:
            Dict[str, Any]: Returns a mapping of all selected machine learning techniques \
                to the achieved F1-score.

        Examples:
             
            >>> from exdpn.util import import_log
            >>> from exdpn.petri_net import get_petri_net
            >>> from exdpn.guard_datasets import extract_all_datasets
            >>> from exdpn import guards
            >>> event_log = import_log('./datasets/p2p_base.xes')
            >>> pn, im, fm = get_petri_net(event_log)
            >>> dp_dataset_map = extract_all_datasets(event_log, pn, im, fm,
            ...                                       event_level_attributes = ['item_category','item_id','item_amount','supplier','total_price'], 
            ...                                       activityName_key = "concept:name")
            >>> # select a certrain decision point and the corresponding data set 
            >>> dp = list(dp_dataset_map.keys())[0]
            >>> dp_dataset = dp_dataset_map[dp]
            >>> # create a guard manager for that decision point
            >>> guard_manager = guards.Guard_Manager(dataframe = dp_dataset)
            >>> guard_manager_results = guard_manager.train_test()

            
            .. include:: ../../docs/_templates/md/example-end.md
        """
        
        # use mapping for target column (map transition to integers)
        transition_int_map = {transition: index for index,
                          transition in enumerate(list(set(self.df_y)))}
        df_y_transformed = [transition_int_map[transition] for transition in self.df_y]
        
        self.guards_results_mean = {}
        guards_results = {guard_name: [] for guard_name in self.guards_list.keys()}
        failing_guards = list()
        # evaluate all selected ml techniques for all guards of the given decision point
        for train_idx, test_idx in self.skf.split(self.df_X, df_y_transformed):
            # get training and test data for current cv split
            X_train = self.df_X.iloc[train_idx, :]
            X_test = self.df_X.iloc[test_idx, :]
            y_train_mapped = list(df_y_transformed[i] for i in train_idx)
            y_test_mapped = list(df_y_transformed[i] for i in test_idx)
            
            # map back to transitions
            y_train = pd.Series([next(trans for trans, trans_id in transition_int_map.items() if trans_id == y) for y in y_train_mapped])
            y_test = pd.Series([next(trans for trans, trans_id in transition_int_map.items() if trans_id == y) for y in y_test_mapped])
            
            for guard_name, guard_models in self.guards_list.items():
                try:
                    # train model for current cv split 
                    guard_models.train(X_train, y_train)
                    y_prediction = guard_models.predict(X_test)

                    # convert Transition objects to integers so that sklearn's F1 score doesn't freak out
                    # this is ugly, we know
                    transition_int_map = {transition: index for index, transition in enumerate(
                        list(set(y_prediction + y_test.tolist())))}
                    y_prediction_transformed = [
                        transition_int_map[transition] for transition in y_prediction]
                    y_test_transformed = [transition_int_map[transition]
                                        for transition in y_test.tolist()]

                    # get f1 score for current cv split
                    guards_results_temp = f1_score(
                        y_test_transformed, y_prediction_transformed, average="weighted")
                    guards_results[guard_name] += [guards_results_temp] 
                except Exception as e:
                    failing_guards.append(guard_name)
                    warnings.warn(f"Warning: Technique {guard_name} failed to train/test on the provided data: {e}. Removing technique from consideration.")
            for failing_guard in failing_guards:
                self.guards_list.pop(failing_guard)
            
            # get mean f1 score for each machine learning technique
            self.guards_results_mean = {technique: np.mean(results) for technique, results in guards_results.items()}

            # retrain models on whole data 
            for guard_models in self.guards_list.values():
                guard_models.train(self.df_X, self.df_y)
        return self.guards_results_mean

    def get_best(self) -> Tuple[str, Guard]:
        """Returns "best" guard for a decision point (see `train_test`).

        Returns:
            * best_guard_name (str): The name of the best performing guard.
            * best_guard (Guard): The corresponding guard object with the best performance.

        Raises:
            AssertionError: If `train_test` has not been called yet.

        Examples:
            
            >>> from exdpn.util import import_log
            >>> from exdpn.petri_net import get_petri_net
            >>> from exdpn.guard_datasets import extract_all_datasets
            >>> from exdpn import guards
            >>> event_log = import_log('./datasets/p2p_base.xes')
            >>> pn, im, fm = get_petri_net(event_log)
            >>> dp_dataset_map = extract_all_datasets(event_log, pn, im, fm,
            ...                                       event_level_attributes = ['item_category','item_id','item_amount','supplier','total_price'], 
            ...                                       activityName_key = "concept:name")
            >>> # select a certrain decision point and the corresponding data set 
            >>> dp_key = [k for k in dp_dataset_map.keys()][0]
            >>> dp_dataset = dp_dataset_map[dp_key]
            >>> # create a guard manager for that decision point
            >>> guard_manager = guards.Guard_Manager(dataframe = dp_dataset)
            >>> guard_manager_results = guard_manager.train_test()
            >>> best_guard = guard_manager.get_best()
            >>> print("Name of best guard:", best_guard[0])
            Name of best guard: Decision Tree

            
            .. include:: ../../docs/_templates/md/example-end.md
        """
        assert self.guards_results_mean != None, "Guards must be evaluated first"
        best_guard_name = max(self.guards_results_mean, key=self.guards_results_mean.get)

        return best_guard_name, self.guards_list[best_guard_name]

    def get_comparison_plot(self) -> Figure:
        """Constructs a comparison bar plot of the F1 scores for all trained techniques.

        Returns:
            Figure: The bar plot figure.

        Examples:

            >>> from exdpn.util import import_log
            >>> from exdpn.petri_net import get_petri_net
            >>> from exdpn.guard_datasets import extract_all_datasets
            >>> from exdpn import guards
            >>> event_log = import_log('./datasets/p2p_base.xes')
            >>> pn, im, fm = get_petri_net(event_log)
            >>> dp_dataset_map = extract_all_datasets(event_log, pn, im, fm,
            ...                                       event_level_attributes = ['item_category','item_id','item_amount','supplier','total_price'], 
            ...                                       activityName_key = "concept:name")
            >>> # select a certrain decision point and the corresponding data set 
            >>> dp = list(dp_dataset_map.keys())[0]
            >>> dp_dataset = dp_dataset_map[dp]
            >>> # create a guard manager for that decision point
            >>> guard_manager = guards.Guard_Manager(dataframe = dp_dataset)
            >>> guard_manager_results = guard_manager.train_test()
            >>> # return comparision plot
            >>> fig = guard_manager.get_comparison_plot()
            
            <img src="../../images/comparision-plot.svg" alt="Comparision plot of the performance of the used machine learning techniques" style="max-height: 350px;"/>

            .. include:: ../../docs/_templates/md/example-end.md

        Note: 
            For an example of the explainable representations of all machine learning techniques please check [Data Petri Net Example](https://github.com/aarkue/eXdpn/blob/main/docs/dpn_example.ipynb).


        """
        guard_results = {(str(technique)): result for technique,
                         result in self.guards_results_mean.items()}
        fig = plt.figure(figsize=(6, 3))
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        axis = plt.gca()
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        plt.ylabel('F1 score')
        plt.title('Comparison of Techniques')

        colors = {
            'Decision Tree': '#478736',
            'Logistic Regression': '#e26f8f',
            'Support Vector Machine': '#e1ad01',
            'Neural Network': '#263488',
            'Random Forest': '#A2B5CD'
        }
        keys = list(guard_results.keys())
        values = [guard_results[key] for key in keys]
        colors = [colors[technique] for technique in keys]
        plt.bar(keys, values, color=colors)
        return fig


# tests implemented examples
if __name__ == "__main__":
    import doctest
    doctest.testmod()
# run python .\exdpn\guards\guard_manager.py from eXdpn file
