"""
.. include:: ./guard.md

"""

from sklearn.tree import DecisionTreeClassifier, plot_tree
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from exdpn.data_preprocessing.data_preprocessing import apply_ohe
from exdpn.guards import Guard
from exdpn.data_preprocessing import fit_ohe

from pandas import DataFrame
from pm4py.objects.petri_net.obj import PetriNet
from typing import Dict, List, Any 


class Decision_Tree_Guard(Guard):
    def __init__(self, hyperparameters: Dict[str, Any] = {'min_samples_split': 0.1, 
                                                          'min_samples_leaf': 0.1, 
                                                          'ccp_alpha': 0.2}) -> None:
        """Initializes a decision tree based guard with the provided hyperparameters.
        
        Args:
            hyperparameters (Dict[str, Any]): Hyperparameters used for the classifier
        
        Raises:
            TypeError: If supplied hyperparameters are invalid

        Examples:
            ```python
            >>> from exdpn.guards import Decision_Tree_Guard
            >>> guard = Decision_Tree_Guard()

            ```
        """
        
        super().__init__(hyperparameters)
        # possible hyperparameters: max_depth, min_samples_split, min_samples_leaf
        try:
            self.model = DecisionTreeClassifier(**hyperparameters)
        except TypeError:
            raise TypeError(
                "Wrong hyperparameters were supplied to the decision tree guard")

        self.transition_int_map = None
        self.feature_names = None
        self.ohe = None
        self.ohe_columns = None


    def train(self, X: DataFrame, y: DataFrame) -> None:
        """Trains the decision tree guard using the dataframe and the specified hyperparameters.
        Args:
            X (DataFrame): Feature variables of the provided dataset, used to train the classifier behind the guard 
            y (DataFrame): Target variable of the provided dataset, is to be predicted using X

        Examples:
            ```python
            >>> import os 
            >>> from exdpn.util import import_log
            >>> from exdpn.petri_net import get_petri_net
            >>> from exdpn.guard_datasets import extract_all_datasets
            >>> from exdpn import guards
            >>> from exdpn.guards import Decision_Tree_Guard
            >>> from exdpn.data_preprocessing import data_preprocessing_evaluation
            >>> #event_log = import_log('p2p_base.xes')
            >>> event_log = import_log(os.path.join(os.getcwd(), 'datasets', 'p2p_base.xes'))
            >>> pn, im, fm = get_petri_net(event_log)
            >>> dp_dataset_map = extract_all_datasets(event_log, pn, im, fm,
            ...                                       case_level_attributes =["concept:name"], 
            ...                                       event_level_attributes = ['item_category','item_id','item_amount','supplier','total_price'], 
            ...                                       activityName_key = "concept:name")
            >>> # select a certrain decision point and the corresponding data set 
            >>> dp_key = [k for k in dp_dataset_map.keys()][1]
            >>> dp_dataset = dp_dataset_map[dp_key]
            >>> X_train, X_test, y_train, y_test = data_preprocessing_evaluation(dp_dataset)
            >>> guard = Decision_Tree_Guard()
            >>> guard.train(X_train, y_train)

            ```
        """
        
        # one hot encoding for categorical data
        self.ohe, self.ohe_columns = fit_ohe(X)
        X = apply_ohe(X, self.ohe)

        # store feature names for the explainable representation
        self.feature_names = list(X.columns)

        # make transition to integer (i.e. ID) map
        self.transition_int_map = {
            transition: index for index, transition in enumerate(list(set(y)))}
        y_transformed = [self.transition_int_map[transition]
                         for transition in y]

        self.model = self.model.fit(X, y_transformed)


    def predict(self, input_instances: DataFrame) -> List[PetriNet.Transition]:
        """Predicts the next transition based on the input instances.
        
        Args:
            input_instances (DataFrame): Dataset of input instances used to predict the target variable, i.e., the next transition
        
        Returns:
            predicted_transitions (List[PetriNet.Transition]): Predicted transitions

        Examples:
            ```python
            >>> import os 
            >>> from exdpn.util import import_log
            >>> from exdpn.petri_net import get_petri_net
            >>> from exdpn.guard_datasets import extract_all_datasets
            >>> from exdpn import guards
            >>> from exdpn.guards import Decision_Tree_Guard
            >>> from exdpn.data_preprocessing import data_preprocessing_evaluation
            >>> #event_log = import_log('p2p_base.xes')
            >>> event_log = import_log(os.path.join(os.getcwd(), 'datasets', 'p2p_base.xes'))
            >>> pn, im, fm = get_petri_net(event_log)
            >>> dp_dataset_map = extract_all_datasets(event_log, pn, im, fm,
            ...                                       case_level_attributes =["concept:name"], 
            ...                                       event_level_attributes = ['item_category','item_id','item_amount','supplier','total_price'], 
            ...                                       activityName_key = "concept:name")
            >>> # select a certrain decision point and the corresponding data set 
            >>> dp_key = [k for k in dp_dataset_map.keys()][1]
            >>> dp_dataset = dp_dataset_map[dp_key]
            >>> X_train, X_test, y_train, y_test = data_preprocessing_evaluation(dp_dataset)
            >>> guard = Decision_Tree_Guard()
            >>> guard.train(X_train, y_train)
            >>> y_prediction = guard.predict(X_test)

            ```
        """
        
        # one hot encoding for categorical data
        input_instances = apply_ohe(input_instances, self.ohe)

        predicted_transition_ids = self.model.predict(input_instances)
        # ty stackoverflow
        # finds the key (transition) where the value (transition integer / id) corresponds to the predicted integer / id
        # for all predicted integers
        return [next(trans for trans, trans_id in self.transition_int_map.items() if trans_id == pred_id) for pred_id in predicted_transition_ids]


    def is_explainable(self) -> bool:
        """Returns whether or not this guard is explainable.
        
        Returns:
            explainable (bool): Whether or not the guard is explainable

        Examples:
            ```python
            >>> from exdpn.guards import Decision_Tree_Guard
            >>> guard = Decision_Tree_Guard()
            >>> guard.is_explainable()
            True

            ```
        """
        
        return True


    def get_explainable_representation(self) -> Figure:
        """Get an explainable representation of the decision tree guard.
        
        Returns:
            explainable_representation (Figure): Matplotlib Figure of the trained decision tree classifier
        
        Raises:
            Exception: If guard has no explainable representation

        Examples:
            ```python
            >>> import os 
            >>> from exdpn.util import import_log
            >>> from exdpn.petri_net import get_petri_net
            >>> from exdpn.guard_datasets import extract_all_datasets
            >>> from exdpn import guards
            >>> from exdpn.guards import Decision_Tree_Guard
            >>> from exdpn.data_preprocessing import data_preprocessing_evaluation
            >>> #event_log = import_log('p2p_base.xes')
            >>> event_log = import_log(os.path.join(os.getcwd(), 'datasets', 'p2p_base.xes'))
            >>> pn, im, fm = get_petri_net(event_log)
            >>> dp_dataset_map = extract_all_datasets(event_log, pn, im, fm,
            ...                                       case_level_attributes =["concept:name"], 
            ...                                       event_level_attributes = ['item_category','item_id','item_amount','supplier','total_price'], 
            ...                                       activityName_key = "concept:name")
            >>> # select a certrain decision point and the corresponding data set 
            >>> dp_key = [k for k in dp_dataset_map.keys()][1]
            >>> dp_dataset = dp_dataset_map[dp_key]
            >>> X_train, X_test, y_train, y_test = data_preprocessing_evaluation(dp_dataset)
            >>> guard = Decision_Tree_Guard()
            >>> guard.train(X_train, y_train)
            >>> y_prediction = guard.predict(X_test)
            >>> guard.get_explainable_representation()
            >>> # todo: figure out how to include a plot 

            ```
        """

        if self.is_explainable() == False:
            raise Exception("Guard is not explainable and therefore has no explainable representation")

        fig, ax = plt.subplots()
        plot_tree(self.model,
                  ax=ax,
                  feature_names=self.feature_names,
                  class_names=[
                      t.label if t.label != None else f"None ({t.name})" for t in self.transition_int_map.keys()],
                  impurity=False,
                  filled=True)
        return fig


# tests implemented examples
if __name__ == "__main__":
    import doctest
    doctest.testmod()
# run python .\exdpn\guards\decision_tree_guard.py from eXdpn file 