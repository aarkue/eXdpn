"""
.. include:: ./../../docs/_templates/md/guards/guard.md

"""

from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from exdpn.data_preprocessing.data_preprocessing import apply_ohe
from exdpn.guards import Guard
from exdpn.data_preprocessing import fit_ohe

from pandas import DataFrame
from pm4py.objects.petri_net.obj import PetriNet
from typing import Dict, List, Any, Optional, Union
import numpy as np

class Decision_Tree_Guard(Guard):
    def __init__(self, hyperparameters: Dict[str, Any] = {'min_impurity_decrease': 0.0075}) -> None:
        """Initializes a decision tree based guard with the provided hyperparameters.

        Args:
            hyperparameters (Dict[str, Any]): Hyperparameters used for the classifier.

        Raises:
            TypeError: If supplied hyperparameters are invalid.

        Examples:
            
            >>> from exdpn.guards import Decision_Tree_Guard
            >>> guard = Decision_Tree_Guard()

            
            .. include:: ../../docs/_templates/md/example-end.md
        """

        super().__init__(hyperparameters)
        
        try:
            self.model = DecisionTreeClassifier(**hyperparameters)
        except TypeError:
            raise TypeError(
                "Wrong hyperparameters were supplied to the decision tree guard")

        self.transition_int_map = None
        self.feature_names = None
        self.ohe = None

    def train(self, X: DataFrame, y: DataFrame) -> None:
        """Trains the decision tree guard using the dataset and the specified hyperparameters.
        Args:
            X (DataFrame): Feature variables of the provided dataset, used to train the classifier behind the guard.
            y (DataFrame): Target variable of the provided dataset, is to be predicted using `X`.

        Examples:
            
            >>> from exdpn.util import import_log
            >>> from exdpn.petri_net import get_petri_net
            >>> from exdpn.guard_datasets import extract_all_datasets
            >>> from exdpn.guards import Decision_Tree_Guard
            >>> from exdpn.data_preprocessing import data_preprocessing_evaluation
            >>> event_log = import_log('./datasets/p2p_base.xes')
            >>> pn, im, fm = get_petri_net(event_log)
            >>> dp_dataset_map = extract_all_datasets(event_log, pn, im, fm, 
            ...                                       event_level_attributes = ['item_category','item_id','item_amount','supplier','total_price'], 
            ...                                       activityName_key = "concept:name")
            >>> # select a c decision point and the corresponding data set 
            >>> dp = list(dp_dataset_map.keys())[0]
            >>> dp_dataset = dp_dataset_map[dp]
            >>> X_train, X_test, y_train, y_test = data_preprocessing_evaluation(dp_dataset)
            >>> guard = Decision_Tree_Guard()
            >>> guard.train(X_train, y_train)

            
            .. include:: ../../docs/_templates/md/example-end.md
        """

        # one hot encoding for categorical data
        self.ohe = fit_ohe(X)
        X = apply_ohe(X, self.ohe)

        # store feature names for the explainable representation
        self.feature_names = list(X.columns)

        # make transition to integer (i.e. ID) map
        self.transition_int_map = {
            transition: index for index, transition in enumerate(list(set(y)))}
        y_transformed = [self.transition_int_map[transition]
                         for transition in y]

        if(len(np.unique(y_transformed))) == 1:
            self.single_class = True
        else:
            self.single_class = False

        self.model = self.model.fit(X, y_transformed)
    def predict(self, input_instances: DataFrame) -> List[PetriNet.Transition]:
        """Predicts the next transition based on the input instances.

        Args:
            input_instances (DataFrame): Dataset of input instances used to predict the target variable, i.e., the next transition.

        Returns:
            List[PetriNet.Transition]: The list of predicted transitions.

        Examples:
             
            >>> from exdpn.util import import_log
            >>> from exdpn.petri_net import get_petri_net
            >>> from exdpn.guard_datasets import extract_all_datasets
            >>> from exdpn.guards import Decision_Tree_Guard
            >>> from exdpn.data_preprocessing import data_preprocessing_evaluation
            >>> event_log = import_log('./datasets/p2p_base.xes')
            >>> pn, im, fm = get_petri_net(event_log)
            >>> dp_dataset_map = extract_all_datasets(event_log, pn, im, fm,
            ...                                       event_level_attributes = ['item_category','item_id','item_amount','supplier','total_price'], 
            ...                                       activityName_key = "concept:name")
            >>> # select a certain decision point and the corresponding data set 
            >>> dp = [k for k in dp_dataset_map.keys()][0]
            >>> dp_dataset = dp_dataset_map[dp]
            >>> X_train, X_test, y_train, y_test = data_preprocessing_evaluation(dp_dataset)
            >>> guard = Decision_Tree_Guard()
            >>> guard.train(X_train, y_train)
            >>> y_prediction = guard.predict(X_test)

            .. include:: ../../docs/_templates/md/example-end.md
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
            bool: Whether or not the guard is explainable.

        Examples:
            
            >>> from exdpn.guards import Decision_Tree_Guard
            >>> guard = Decision_Tree_Guard()
            >>> guard.is_explainable()
            True

            
            .. include:: ../../docs/_templates/md/example-end.md

        """

        return True

    def get_explainable_representation(self, data:Optional[DataFrame] = None) -> Figure:
        """Returns an explainable representation of the decision tree guard.
        Args:
            data (DataFrame, optional): *Not needed for Explainable Representation of Decision Trees* 

        Returns:
            Figure: Matplotlib Figure of the trained decision tree classifier.

        Raises:
            Exception: If the guard has no explainable representation.

        Examples:
            
            >>> from exdpn.util import import_log
            >>> from exdpn.petri_net import get_petri_net
            >>> from exdpn.guard_datasets import extract_all_datasets
            >>> from exdpn.guards import Decision_Tree_Guard
            >>> from exdpn.data_preprocessing import data_preprocessing_evaluation
            >>> event_log = import_log('./datasets/p2p_base.xes')
            >>> pn, im, fm = get_petri_net(event_log)
            >>> dp_dataset_map = extract_all_datasets(event_log, pn, im, fm,
            ...                                       event_level_attributes = ['item_category','item_id','item_amount','supplier','total_price'], 
            ...                                       activityName_key = "concept:name")
            >>> # select a certain decision point and the corresponding data set 
            >>> dp = list(dp_dataset_map.keys())[0]
            >>> dp_dataset = dp_dataset_map[dp]
            >>> X_train, X_test, y_train, y_test = data_preprocessing_evaluation(dp_dataset)
            >>> guard = Decision_Tree_Guard()
            >>> guard.train(X_train, y_train)
            >>> y_prediction = guard.predict(X_test)
            >>> # return figure of explainable representation
            >>> fig = guard.get_explainable_representation() # results may deviate 
            
            <img src="../../images/dt-example-representation.svg" alt="Example explainable representation of a decision tree guard" style="max-height: 350px;"/>

            .. include:: ../../docs/_templates/md/example-end.md
            
        Note: 
            For an example of the explainable representations of all machine learning techniques please check [Data Petri Net Example](https://github.com/aarkue/eXdpn/blob/main/docs/dpn_example.ipynb).

        """

        if self.is_explainable() == False:
            raise Exception(
                "Guard is not explainable and therefore has no explainable representation")

        fig, ax = plt.subplots()
        plot_tree(self.model,
                  ax=ax,
                  feature_names=self.feature_names,
                  class_names=[
                      t.label if t.label != None else f"None ({t.name})" for t in self.transition_int_map.keys()],
                  impurity=False,
                  filled=True)
        plt.suptitle("Decision Tree", fontsize=14)
        if self.single_class:
            plt.title("Warning: Only one target class present. Results may be misleading.",{'color':  'darkred'})
        plt.ylabel("Feature Attributes", fontsize=14)
        return fig

    def get_global_explanations(self, base_sample: DataFrame) -> Dict[str,Union[Figure,str]]:
        """Get a global explainable representation for the concrete machine learning classifier.
        Args:
            base_sample (DataFrame): A small (10-30) sample of the population for this decision point; Used for calculation of shap values.
        Returns:
            Dict[str,Figure]: A dictionary containing the global explainable representations. Containing the following entries:
            - "Decision Tree"
        """
        fig, ax = plt.subplots()
        plot_tree(self.model,
                  ax=ax,
                  feature_names=self.feature_names,
                  class_names=[
                      t.label if t.label != None else f"None ({t.name})" for t in self.transition_int_map.keys()],
                  impurity=False,
                  filled=True)
        plt.suptitle("Decision Tree", fontsize=14)
        if self.single_class:
            plt.title("Warning: Only one target class present. Results may be misleading.",{'color':  'darkred'})
        plt.ylabel("Feature Attributes", fontsize=14)
        return {'Decision Tree': fig}


    def get_local_explanations(self, local_data:DataFrame, base_sample: DataFrame) -> Dict[str,Figure]:
        """ Local Representations are not supported for Decision Tree. 

        Args:
            local_data (DataFrame): A dataframe containing the single decision situation.
            base_sample (DataFrame): A small (10-30) sample of the population for this decision point; Used for calculation of shap values.

        Returns:
            Dict[str,Figure]: A dictionary containing the explainable representations for the single decision situation. Containing the following entries:
            - "Decision plot (Multioutput)"
            - "Decision plot for `X`" (for all output labels X)
            - "Force plot for `X`" (for all output labels X)
                
        """
        fig = plt.figure()
        fig.text(0.5, 0.5,
            'Local explanations are not available for the decision tree.',
            fontsize = 30,
            color = "darkred",
            wrap = True,
            horizontalalignment="center"
        )
        return {'Not available': fig}


# tests implemented examples
if __name__ == "__main__":
    import doctest
    doctest.testmod()
# run python .\exdpn\guards\decision_tree_guard.py from eXdpn file
