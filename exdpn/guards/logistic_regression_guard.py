"""
.. include:: ./../../docs/_templates/md/guards/guard.md

"""

from exdpn.data_preprocessing.data_preprocessing import apply_ohe, apply_scaling, fit_scaling
from exdpn.guards import Guard
from exdpn.data_preprocessing import fit_ohe

from sklearn.linear_model import LogisticRegression
from pandas import DataFrame, Series
from pm4py.objects.petri_net.obj import PetriNet
from typing import Dict, List, Any 
import shap 
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt 
import numpy as np 



class Logistic_Regression_Guard(Guard):
    def __init__(self, hyperparameters: Dict[str, Any] = {"C": 0.5}) -> None:
        """Initializes a logistic regression based guard with the provided hyperparameters.
        
        Args:
            hyperparameters (Dict[str, Any]): Hyperparameters used for the classifier

        Raises:
            TypeError: If supplied hyperparameters are invalid

        Examples:
            ```python
            >>> from exdpn.guards import Logistic_Regression_Guard
            >>> guard = Logistic_Regression_Guard()
            
            ```
        """
        
        super().__init__(hyperparameters)
        # possible hyperparameter: C (regularization parameter)
        try:
            self.model = LogisticRegression(**hyperparameters)
        except TypeError:
            raise TypeError(
                "Wrong hyperparameters were supplied to the logistic regression guard")

        self.transition_int_map = None
        self.feature_names      = None
        self.ohe                = None
        self.scaler             = None
        self.scaler_columns     = None

    def train(self, X: DataFrame, y: DataFrame) -> None:
        """Trains the logistic regression guard using the dataframe and the specified hyperparameters.
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
            >>> from exdpn.guards import Logistic_Regression_Guard
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
            >>> guard = Logistic_Regression_Guard()
            >>> guard.train(X_train, y_train)

            ```
        """
        
        # scale numerical attributes
        self.scaler, self.scaler_columns = fit_scaling(X)
        X = apply_scaling(X, self.scaler, self.scaler_columns)
        # one hot encoding for categorical data 
        self.ohe = fit_ohe(X)
        X = apply_ohe(X, self.ohe)
        self.X_train = X 

        # store feature names for the explainable representation
        self.feature_names = list(X.columns)

        # make transition to integer (i.e. ID) map
        self.transition_int_map = {
            transition: index for index, transition in enumerate(list(set(y)))}
        y_transformed = [self.transition_int_map[transition]
                         for transition in y]

        # check if more than 1 class in training set
        if(len(np.unique(y_transformed))) == 1:
            self.single_class = True
            self.class_label = np.unique(y_transformed)
        else:
            self.single_class = False
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
            >>> from exdpn.guards import Logistic_Regression_Guard
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
            >>> guard = Logistic_Regression_Guard()
            >>> guard.train(X_train, y_train)
            >>> y_prediction = guard.predict(X_test)

            ```
        """
        
        # scale numerical attributes
        input_instances = apply_scaling(input_instances, self.scaler, self.scaler_columns)
        # one hot encoding for categorical data 
        input_instances = apply_ohe(input_instances, self.ohe)
        self.input_instances = input_instances
        
        if self.single_class:
            predicted_transition_ids = np.full(len(self.input_instances), self.class_label) 
        else:
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
            >>> from exdpn.guards import Logistic_Regression_Guard
            >>> guard = Logistic_Regression_Guard()
            >>> guard.is_explainable()
            True

            ```
        """
        
        return True


    def get_explainable_representation(self) -> Figure:
        """Get an explainable representation of the logistic regression guard, a Matplotlib plot using SHAP.
        
        Returns:
            explainable_representation (Figure): Matplotlib Figure of the trained logistic regression model
        
        Raises:
            Exception: If guard has no explainable representation

        Examples:
            ```python
            >>> import os 
            >>> from exdpn.util import import_log
            >>> from exdpn.petri_net import get_petri_net
            >>> from exdpn.guard_datasets import extract_all_datasets
            >>> from exdpn import guards
            >>> from exdpn.guards import Logistic_Regression_Guard
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
            >>> guard = Logistic_Regression_Guard()
            >>> guard.train(X_train, y_train)
            >>> y_prediction = guard.predict(X_test)
            >>> guard.get_explainable_representation()
            >>> # todo: figure out how to include a plot 

            ```
        """
        
        if self.is_explainable() == False:
            raise Exception("Guard is not explainable and therefore has no explainable representation")

        classes = [t.label if t.label != None else f"None ({t.name})" for t in self.transition_int_map.keys()]
        
        if self.single_class:
            fig = plt.figure()
            ax = fig.add_subplot()
            fig.subplots_adjust(top = 0.85)
            plt.figure(figsize = (9, 8))
            ax.text(0, 0.7, 
                    "Since only one class was represented in the training data, \nall samples are predicted as:\n" + str(classes[0]), 
                    fontsize = 14)
            ax.axis('off')
        else:
            explainer = shap.LinearExplainer(self.model, self.X_train)

            shap_values = explainer.shap_values(self.input_instances)

            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, 
                              self.input_instances, 
                              plot_type = "bar", 
                              show = False,
                              class_names = classes,
                              class_inds = range(len(classes)))
            plt.title("Feature Impact on Model Prediction", fontsize = 14)
            plt.ylabel("Feature Attributes", fontsize = 14)

        return fig 


# tests implemented examples
if __name__ == "__main__":
    import doctest
    doctest.testmod()
# run python .\exdpn\guards\logistic_regression_guard.py from eXdpn file 