from exdpn.data_preprocessing.data_preprocessing import apply_ohe, apply_scaling, fit_scaling
from exdpn.guards import Guard
from exdpn.data_preprocessing import fit_ohe

#from sklearn.svm import LinearSVC 
from sklearn.svm import SVC 
from pandas import DataFrame, Series
from pm4py.objects.petri_net.obj import PetriNet
from typing import Dict
import numpy as np 
import shap 



class SVM_Guard(Guard):
    def __init__(self, hyperparameters: Dict[str, any] = {"C": 0.5, "kernel": "linear", "probability": True}) -> None:
        """Initializes a support vector machine model based guard with the provided hyperparameters
        Args:
            hyperparameters (dict[str, any]): Hyperparameters used for the classifier"""
        super().__init__(hyperparameters)
        # possible hyperparameter: C (regularization parameter)
        try:
            self.model = SVC(**hyperparameters)
        except TypeError:
            raise TypeError(
                "Wrong hyperparameters were supplied to the decision tree guard")

        self.transition_int_map = None
        self.feature_names      = None
        self.ohe                = None
        self.ohe_columns        = None
        self.scaler             = None
        self.scaler_columns     = None

    def train(self, X: DataFrame, y: DataFrame) -> None:
        """Shall train the concrete classifier/model behind the guard using the dataframe and the specified hyperparameters.
        Args:
            X (DataFrame): Dataset used to train the classifier behind the guard (w/o the target label)
            y (DataFrame): Target label for each instance in the X dataset used to train the model"""
        # scale numerical attributes
        self.scaler, self.scaler_columns = fit_scaling(X)
        X = apply_scaling(X, self.scaler, self.scaler_columns)
        # one hot encoding for categorical data 
        self.ohe, self.ohe_columns = fit_ohe(X)
        X = apply_ohe(X, self.ohe)

        self.X_train = X

        # store feature names for the explainable representation
        self.feature_names = list(X.columns)

        # make transition to integer (i.e. ID) map
        self.transition_int_map = {
            transition: index for index, transition in enumerate(list(set(y)))}
        y_transformed = [self.transition_int_map[transition]
                         for transition in y]

        self.model = self.model.fit(X, y_transformed)

    def predict(self, input_instances: DataFrame) -> list[PetriNet.Transition]:
        """Shall use the classifier/model behind the guard to predict the next transition.
        Args:
            input_instances (DataFrame): Input instances used to predict the next transition
        Returns:
            predicted_transitions (list[PetriNet.Transition]): Predicted transitions"""
        # scale numerical attributes
        input_instances = apply_scaling(input_instances, self.scaler, self.scaler_columns)
        # one hot encoding for categorical data 
        input_instances = apply_ohe(input_instances, self.ohe)
        self.input_instances = input_instances
        
        predicted_transition_ids = self.model.predict(input_instances)
        # ty stackoverflow
        # finds the key (transition) where the value (transition integer / id) corresponds to the predicted integer / id
        # for all predicted integers
        return [next(trans for trans, trans_id in self.transition_int_map.items() if trans_id == pred_id) for pred_id in predicted_transition_ids]

    def is_explainable(self) -> bool:
        """Shall return wheter or not the internal classifier is explainable.
        Returns:
            explainable (bool): Wheter or not the guard is explainable"""
        return True

    def get_explainable_representation(self) -> DataFrame:
        """Shall return an explainable representation of the guard. Shall throw an exception if the guard is not explainable.
        Returns:
            explainable_representation (DataFrame): Explainable representation of the guard"""
        
        explainer = shap.KernelExplainer(self.model.predict_proba, self.X_train)
        shap_values = explainer.shap_values(self.input_instances)
        
        return shap.force_plot(explainer.expected_value[0], shap_values[0], self.input_instances)