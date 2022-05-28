from exdpn.data_preprocessing.data_preprocessing import apply_ohe, apply_scaling, fit_scaling
from exdpn.guards import Guard
from exdpn.data_preprocessing import fit_ohe

from sklearn.linear_model import LogisticRegression
from pandas import DataFrame, Series
from pm4py.objects.petri_net.obj import PetriNet
from typing import Dict
import numpy as np 



class Logistic_Regression_Guard(Guard):
    def __init__(self, hyperparameters: Dict[str, any] = {"C": 1}) -> None:
        """Initializes a logistic regression model based guard with the provided hyperparameters
        Args:
            hyperparameters (dict[str, any]): Hyperparameters used for the classifier"""
        super().__init__(hyperparameters)
        # possible hyperparameter: C (regularization parameter)
        try:
            self.model = LogisticRegression(**hyperparameters)
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
        
        #print("A Logistic Regression Classifier calculates the probability that a sample belongs to a certain class, if the probability is >50% the sample is assigned to that class. In a multi-class classification problem (i.e., there are more than the usual two classes), the so-called One-vs-Rest method is used. This means that the original multi-class classification problem is split into multiple binary classification problems (P(this class) vs. P(not this class)).")

        # TODO hard coded for binary classification problem 
        print("Classification Problem: P(", [k for k in self.transition_int_map.keys()][1], ") vs. P(",
              [k for k in self.transition_int_map.keys()][0], ")")

        print("Used feature variables in the Logistic Regression Classification model, sorted by their influence on the probability. Weight < 1: negative influece, Weight > 1: positive influence.")

        model_features = self.model.feature_names_in_
        model_weights = np.round(np.exp(self.model.coef_[0]), 2)
        representation_data = {"Feature Variables": model_features, "Weights": model_weights} 

        representation = DataFrame(representation_data, columns = ["Feature Variables", "Weights"])

        return representation.sort_values(by = ["Weights"], ascending = False)
