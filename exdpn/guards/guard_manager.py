"""
.. include:: ./../../docs/_templates/md/guards/guard_manager.md

"""

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from pandas import DataFrame
from exdpn.data_preprocessing.data_preprocessing import basic_data_preprocessing

from typing import Dict, List, Tuple, Any 

from exdpn.guards import ML_Technique # imports all guard classes
from exdpn.guards import Guard
from exdpn.data_preprocessing import data_preprocessing_evaluation 
from exdpn.guards.model_builder import model_builder
from sklearn.metrics import f1_score


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
                                                                        ML_Technique.SVM: {"C": 0.5}}) -> None:
        """Initializes all information needed for the calculation of the best guard for each decision point and /
        returns a dictionary with the list of all guards for each machine learning technique.
        
        Args:
            dataframe (DataFrame): Dataset used to evaluate the guard  
            ml_list (List[ML_Technique]): List of all machine learning techniques that should be evaluated, default is all implemented 
            hyperparameters (Dict[ML_Technique, Dict[str, Any]]): Hyperparameter that should be used for the machine learning techniques, \
            if not specified default parameters are used
        
        Examples:
            ```python
            >>> import os 
            >>> from exdpn.util import import_log
            >>> from exdpn.petri_net import get_petri_net
            >>> from exdpn.guard_datasets import extract_all_datasets
            >>> from exdpn import guards
            >>> event_log = import_log(os.path.join(os.getcwd(), 'datasets', 'p2p_base.xes'))
            >>> pn, im, fm = get_petri_net(event_log)
            >>> dp_dataset_map = extract_all_datasets(event_log, pn, im, fm,
            ...                                       case_level_attributes =["concept:name"], 
            ...                                       event_level_attributes = ['item_category','item_id','item_amount','supplier','total_price'], 
            ...                                       activityName_key = "concept:name")
            >>> # select a certrain decision point and the corresponding data set 
            >>> dp_key = [k for k in dp_dataset_map.keys()][1]
            >>> dp_dataset = dp_dataset_map[dp_key]
            >>> # create a guard manager for that decision point
            >>> guard_manager = guards.Guard_Manager(dataframe = dp_dataset)

            ```
        """
        
        # TODO: refactor data_preprocessing so that it does not do more than one thing
        # or does all the things

        # TODO: think about persistence of the encoders so that new unseen instances can still be encoded

        X_train, X_test, y_train, y_test = data_preprocessing_evaluation(dataframe)
        
        self.dataframe = dataframe
        self.X_train = X_train
        self.X_test  = X_test
        self.y_train = y_train
        self.y_test  = y_test

        # create list of all needed machine learning techniques to evaluate the guards
        self.guards_list = {technique: model_builder(technique, hyperparameters[technique]) for technique in ml_list}
        
        self.guards_results = None


    def train_test(self) -> Dict[str, Any]:
        """ Calculates for a given decision point all selected guards and returns the precision of the machine learning model, \
        using the specified machine learning techniques.
        
        Returns:
            guards_results (Dict[str, Any]): Returns a mapping of all selected machine learning techniques \
            to the achieved F1-score and two trained guard models: the "training" guard (position 0) and final guard (position 1)
        
        Examples:
            ```python
            >>> import os 
            >>> from exdpn.util import import_log
            >>> from exdpn.petri_net import get_petri_net
            >>> from exdpn.guard_datasets import extract_all_datasets
            >>> from exdpn import guards
            >>> event_log = import_log(os.path.join(os.getcwd(), 'datasets', 'p2p_base.xes'))
            >>> pn, im, fm = get_petri_net(event_log)
            >>> dp_dataset_map = extract_all_datasets(event_log, pn, im, fm,
            ...                                       case_level_attributes =["concept:name"], 
            ...                                       event_level_attributes = ['item_category','item_id','item_amount','supplier','total_price'], 
            ...                                       activityName_key = "concept:name")
            >>> # select a certrain decision point and the corresponding data set 
            >>> dp_key = [k for k in dp_dataset_map.keys()][1]
            >>> dp_dataset = dp_dataset_map[dp_key]
            >>> # create a guard manager for that decision point
            >>> guard_manager = guards.Guard_Manager(dataframe = dp_dataset)
            >>> guard_manager_results = guard_manager.train_test()
            
            ```
        
        """
        
        self.guards_results = {}
        # evaluate all selected ml techniques for all guards of the given decision point
        for guard_name, guard_models in self.guards_list.items():
            guard_models.train(self.X_train, self.y_train)
            y_prediction = guard_models.predict(self.X_test)
             
            # convert Transition objects to integers so that sklearn's F1 score doesn't freak out
            # this is ugly, we know
            transition_int_map = {transition: index for index, transition in enumerate(list(set(y_prediction + self.y_test.tolist())))}
            y_prediction_transformed = [transition_int_map[transition] for transition in y_prediction]
            y_test_transformed = [transition_int_map[transition] for transition in self.y_test.tolist()]

            self.guards_results[guard_name] = f1_score(y_test_transformed, y_prediction_transformed, average="weighted")
        
        return self.guards_results


    def get_best(self) -> Tuple[str, List[Guard]]:
        """ Returns "best" guard for a decision point.
        
        Returns:
            best_guard (Tuple[str, Guard]): Returns "best" guard for a decision point with respect to the \
            chosen metric (F1 score), the returned tuple contains the machine learning technique and the corresponding guard
        
        Examples:
            ```python
            >>> import os 
            >>> from exdpn.util import import_log
            >>> from exdpn.petri_net import get_petri_net
            >>> from exdpn.guard_datasets import extract_all_datasets
            >>> from exdpn import guards
            >>> event_log = import_log(os.path.join(os.getcwd(), 'datasets', 'p2p_base.xes'))
            >>> pn, im, fm = get_petri_net(event_log)
            >>> dp_dataset_map = extract_all_datasets(event_log, pn, im, fm,
            ...                                       case_level_attributes =["concept:name"], 
            ...                                       event_level_attributes = ['item_category','item_id','item_amount','supplier','total_price'], 
            ...                                       activityName_key = "concept:name")
            >>> # select a certrain decision point and the corresponding data set 
            >>> dp_key = [k for k in dp_dataset_map.keys()][1]
            >>> dp_dataset = dp_dataset_map[dp_key]
            >>> # create a guard manager for that decision point
            >>> guard_manager = guards.Guard_Manager(dataframe = dp_dataset)
            >>> guard_manager_results = guard_manager.train_test()
            >>> best_guard = guard_manager.get_best()
            >>> print("Name of best guard:", best_guard[0])
            Name of best guard: Decision Tree

            ```

        """
        
        assert self.guards_results != None, "Guards must be evaluated first"
        best_guard_name = max(self.guards_results, key=self.guards_results.get)
        
        return best_guard_name, self.guards_list[best_guard_name]

    def get_comparison_plot(self) -> Figure:        
        """ Constructs a comparison bar plot of the F1 scores for all trained techniques for a place.

        Returns:
            Figure: The bar plot figure

        Examples:
            ```python
            >>> import os 
            >>> from exdpn.util import import_log
            >>> from exdpn.petri_net import get_petri_net
            >>> from exdpn.guard_datasets import extract_all_datasets
            >>> from exdpn import guards
            >>> event_log = import_log(os.path.join(os.getcwd(), 'datasets', 'p2p_base.xes'))
            >>> pn, im, fm = get_petri_net(event_log)
            >>> dp_dataset_map = extract_all_datasets(event_log, pn, im, fm,
            ...                                       case_level_attributes =["concept:name"], 
            ...                                       event_level_attributes = ['item_category','item_id','item_amount','supplier','total_price'], 
            ...                                       activityName_key = "concept:name")
            >>> # select a certrain decision point and the corresponding data set 
            >>> dp_key = [k for k in dp_dataset_map.keys()][1]
            >>> dp_dataset = dp_dataset_map[dp_key]
            >>> # create a guard manager for that decision point
            >>> guard_manager = guards.Guard_Manager(dataframe = dp_dataset)
            >>> guard_manager_results = guard_manager.train_test()
            >>> guard_manager.get_comparison_plot()
            >>> # add plot here 

            ```

        """
        guard_results = {(str(technique)): result for technique,result in self.guards_results.items()}
        fig = plt.figure(figsize=(6,3))
        plt.xticks(rotation=45,ha='right')
        plt.ylim(0,1)
        axis = plt.gca()
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        plt.ylabel('F1 score')
        plt.title('Comparison of Techniques')
        
        colors = {
            'Decision Tree': '#478736',
            'Logistic Regression': '#e26f8f',
            'Support Vector Machine': '#e1ad01',
            'Neural Network': '#263488'
        }   
        keys = list(guard_results.keys())
        values = [guard_results[key] for key in keys]
        colors = [colors[technique] for technique in keys]
        plt.bar(keys,values,color=colors)
        return fig

# tests implemented examples
if __name__ == "__main__":
    import doctest
    doctest.testmod()
# run python .\exdpn\guards\guard_manager.py from eXdpn file 