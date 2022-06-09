from typing import Dict, List

from exdpn.data_preprocessing import fit_ohe
from exdpn.data_preprocessing.data_preprocessing import apply_ohe, apply_scaling, fit_scaling
from exdpn.guards import Guard

from pandas import DataFrame
from pm4py.objects.petri_net.obj import PetriNet
from sklearn.neural_network import MLPClassifier
import numpy as np

# Explainability
import shap
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

class Neural_Network_Guard(Guard):
    def __init__(self, hyperparameters: Dict[str, any] = {'hidden_layer_sizes': (10,10)}) -> None:
        """Initializes a neural network based guard with the provided hyperparameters.
        Args:
            hyperparameters (Dict[str, any]): Hyperparameters used for the classifier
        """

        super().__init__(hyperparameters)
        try:
            self.model = MLPClassifier(**hyperparameters)
        except TypeError:
            raise TypeError(
                "Wrong hyperparameters were supplied to the neural network guard"
            )

        self.transition_int_map = None
        self.feature_names      = None
        self.ohe                = None
        self.ohe_columns        = None
        self.scaler             = None
        self.scaler_columns     = None
        self.training_data      = None


    def train(self, X: DataFrame, y: DataFrame) -> None:
        """Trains the neural network guard using the dataframe and the specified hyperparameters.
        Args:
            X (DataFrame): Feature variables of the provided dataset, used to train the classifier behind the guard 
            y (DataFrame): Target variable of the provided dataset, is to be predicted using X
        """

        # Scale numerical attributes
        self.scaler, self.scaler_columns = fit_scaling(X)
        X = apply_scaling(X, self.scaler, self.scaler_columns)

        # One-Hot Encoding for categorical data 
        self.ohe, self.ohe_columns = fit_ohe(X)
        X = apply_ohe(X, self.ohe)

        self.training_data = X

        # Store feature names for the explainable representation
        self.feature_names = list(X.columns)
        self.target_names = list(y.unique())
        # Make transition to integer (i.e. ID) map
        self.transition_int_map = {
            transition: index for index, transition in enumerate(list(set(y)))
        }

        y_transformed = [
            self.transition_int_map[transition]
            for transition in y
        ]

        self.model = self.model.fit(X, y_transformed)


    def predict(self, input_instances: DataFrame) -> List[PetriNet.Transition]:
        """Predicts the next transition based on the input instances.
        Args:
            input_instances (DataFrame): Dataset of input instances used to predict the target variable, i.e., the next transition
        Returns:
            predicted_transitions (List[PetriNet.Transition]): Predicted transitions
        """

        # Scale numerical attributes
        input_instances = apply_scaling(input_instances, self.scaler, self.scaler_columns)
        # One-Hot Encoding for categorical data 
        input_instances = apply_ohe(input_instances, self.ohe)
        
        predicted_transition_ids = self.model.predict(input_instances)

        # Retrieve the transition for which the id corresponds to the predicted id
        reverse_dict = {
            value: key for key, value in self.transition_int_map.items()
        }
        return [reverse_dict[pred_id] for pred_id in predicted_transition_ids]


    def is_explainable(self) -> bool:
        """Returns whether or not this guard is explainable.
        Returns:
            explainable (bool): Whether or not the guard is explainable
        """

        return True

    def get_explainable_representation(self) -> Figure:
        """Get an explainable representation of the neural network guard, a Matplotlib plot using SHAP.
        Returns:
            explainable_representation (Figure): Explainable representation of the guard
        """
        
        if self.is_explainable() == False:
            raise Exception("Guard is not explainable and therefore has no explainable representation")

        # X_train_summary = shap.kmeans(self.training_data, 1)
        sampled_data = self.training_data.sample(n=min(100, len(self.training_data)))
        


        explainer = shap.KernelExplainer(self.model.predict, sampled_data, output_names=self.target_names)
        
        # explainer = shap.Explainer(self.model.predict, X_train_summary, output_names=self.target_names)
        # shap_values = explainer(self.training_data.sample(n=min(100, len(self.training_data))))
        shap_values = explainer.shap_values(sampled_data)
        fig = plt.figure()
        # Force Plot
        # shap.force_plot(explainer.expected_value[0], shap_values[0], self.training_data.sample(n=min(100, len(self.training_data))))

        # Docs for this summary plot: https://shap-lrjball.readthedocs.io/en/latest/generated/shap.summary_plot.html
        shap.summary_plot(shap_values, sampled_data, plot_type="bar", show=False, class_names=self.target_names, plot_size="auto")
        # Bee-Swarm
        # shap.summary_plot(shap_values, self.training_data, show=False, plot_size="auto")
        # shap.plots.beeswarm(shap_values, order=shap_values.abs.max(0))

        # Decision Plot
        # shap.decision_plot(explainer.expected_value, shap_values,self.feature_names, show=False)

        # mean_shap_value = np.mean(shap_values, axis=0)
        # explanation = shap.Explanation(mean_shap_value,0,feature_names=self.feature_names, output_names=self.target_names)
        # shap.plots.bar(explanation)
        return fig
