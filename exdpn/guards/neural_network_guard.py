"""
.. include:: ./../../docs/_templates/md/guards/guard.md

"""

from typing import Dict, List, Any, Optional

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
    def __init__(self, hyperparameters: Dict[str, Any] = {'hidden_layer_sizes': (10, 10)}) -> None:
        """Initializes a neural network based guard with the provided hyperparameters.

        Args:
            hyperparameters (Dict[str, Any]): Hyperparameters used for the classifier.

        Raises:
            TypeError: If supplied hyperparameters are invalid

        Examples:
            
            >>> from exdpn.guards import Neural_Network_Guard
            >>> guard = Neural_Network_Guard()

            .. include:: ../../docs/_templates/md/example-end.md  

        """
        super().__init__(hyperparameters)
        try:
            self.model = MLPClassifier(**hyperparameters)
        except TypeError:
            raise TypeError(
                "Wrong hyperparameters were supplied to the neural network guard"
            )

        self.transition_int_map = None
        self.feature_names = None
        self.ohe = None
        self.scaler = None
        self.scaler_columns = None
        self.training_data = None

    def train(self, X: DataFrame, y: DataFrame) -> None:
        """Trains the neural network guard using the dataset and the specified hyperparameters.

        Args:
            X (DataFrame): Feature variables of the provided dataset, used to train the classifier behind the guard .
            y (DataFrame): Target variable of the provided dataset, is to be predicted using `X`.

        Examples:
            
            >>> import os 
            >>> from exdpn.util import import_log
            >>> from exdpn.petri_net import get_petri_net
            >>> from exdpn.guard_datasets import extract_all_datasets
            >>> from exdpn.guards import Neural_Network_Guard
            >>> from exdpn.data_preprocessing import data_preprocessing_evaluation
            >>> event_log = import_log(os.path.join(os.getcwd(), 'datasets', 'p2p_base.xes'))
            >>> pn, im, fm = get_petri_net(event_log)
            >>> dp_dataset_map = extract_all_datasets(event_log, pn, im, fm,
            ...                                       case_level_attributes =["concept:name"], 
            ...                                       event_level_attributes = ['item_category','item_id','item_amount','supplier','total_price'], 
            ...                                       activityName_key = "concept:name")
            >>> # select a certain decision point and the corresponding data set 
            >>> dp = list(dp_dataset_map.keys())[0]
            >>> dp_dataset = dp_dataset_map[dp]
            >>> X_train, X_test, y_train, y_test = data_preprocessing_evaluation(dp_dataset)
            >>> guard = Neural_Network_Guard()
            >>> guard.train(X_train, y_train)

            .. include:: ../../docs/_templates/md/example-end.md

        """
        # Scale numerical attributes
        self.scaler, self.scaler_columns = fit_scaling(X)
        X = apply_scaling(X, self.scaler, self.scaler_columns)

        # One-Hot Encoding for categorical data
        self.ohe = fit_ohe(X)
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
            
            >>> import os 
            >>> from exdpn.util import import_log
            >>> from exdpn.petri_net import get_petri_net
            >>> from exdpn.guard_datasets import extract_all_datasets
            >>> from exdpn.guards import Neural_Network_Guard
            >>> from exdpn.data_preprocessing import data_preprocessing_evaluation
            >>> event_log = import_log(os.path.join(os.getcwd(), 'datasets', 'p2p_base.xes'))
            >>> pn, im, fm = get_petri_net(event_log)
            >>> dp_dataset_map = extract_all_datasets(event_log, pn, im, fm,
            ...                                       case_level_attributes =["concept:name"], 
            ...                                       event_level_attributes = ['item_category','item_id','item_amount','supplier','total_price'], 
            ...                                       activityName_key = "concept:name")
            >>> # select a certrain decision point and the corresponding data set 
            >>> dp = list(dp_dataset_map.keys())[0]
            >>> dp_dataset = dp_dataset_map[dp]
            >>> X_train, X_test, y_train, y_test = data_preprocessing_evaluation(dp_dataset)
            >>> guard = Neural_Network_Guard()
            >>> guard.train(X_train, y_train)
            >>> y_prediction = guard.predict(X_test)

            .. include:: ../../docs/_templates/md/example-end.md

        """
        # Scale numerical attributes
        input_instances = apply_scaling(
            input_instances, self.scaler, self.scaler_columns)
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
            bool: Whether or not the guard is explainable.

        Examples:
            
            >>> from exdpn.guards import Neural_Network_Guard
            >>> guard = Neural_Network_Guard()
            >>> guard.is_explainable()
            True

            .. include:: ../../docs/_templates/md/example-end.md

        """
        return True

    def get_explainable_representation(self, data:Optional[DataFrame]) -> Figure:
        """Returns an explainable representation of the neural network guard, a Matplotlib plot using SHAP. For Neural Networks, as the calculation of the SHAP values can be very time-consuming, we recommend using sampled or aggregated (clustered) data.

        Args:
            data (DataFrame): Dataset of input instances used to construct an explainable representation.

        Returns:
            Figure: Explainable representation of the guard.

        Raises:
            Exception: If guard has no explainable representation.

        Examples:
            
            >>> import os 
            >>> from exdpn.util import import_log
            >>> from exdpn.petri_net import get_petri_net
            >>> from exdpn.guard_datasets import extract_all_datasets
            >>> from exdpn.guards import Neural_Network_Guard
            >>> from exdpn.data_preprocessing import data_preprocessing_evaluation
            >>> event_log = import_log(os.path.join(os.getcwd(), 'datasets', 'p2p_base.xes'))
            >>> pn, im, fm = get_petri_net(event_log)
            >>> dp_dataset_map = extract_all_datasets(event_log, pn, im, fm,
            ...                                       event_level_attributes = ['item_category','item_id','item_amount','supplier','total_price'], 
            ...                                       activityName_key = "concept:name")
            >>> # select a certrain decision point and the corresponding data set 
            >>> dp = list(dp_dataset_map.keys())[0]
            >>> dp_dataset = dp_dataset_map[dp]
            >>> X_train, X_test, y_train, y_test = data_preprocessing_evaluation(dp_dataset)
            >>> guard = Neural_Network_Guard()
            >>> guard.train(X_train, y_train)
            >>> y_prediction = guard.predict(X_test)
            >>> # sample from test data, as explainable representation of NN is computationally expensive
            >>> sampled_test_data = X_test.sample(n=min(100, len(X_test)));
            >>> # return figure of explainable representation
            >>> fig = guard.get_explainable_representation(sampled_test_data) # results may deviate 
            
            <img src="../../images/nn-example-representation.svg" alt="Example explainable representation of a neural network guard" style="max-height: 350px;"/>
            
            
            .. include:: ../../docs/_templates/md/example-end.md

        Note: 
            For an example of the explainable representations of all machine learning techniques please check [Data Petri Net Example](https://github.com/aarkue/eXdpn/blob/main/docs/dpn_example.ipynb).

        """
        data = apply_scaling(data, self.scaler, self.scaler_columns)
        # One-Hot Encoding for categorical data
        data = apply_ohe(data, self.ohe)

        def shap_predict(data: np.ndarray):
            data_asframe = DataFrame(data, columns=self.feature_names)
            ret = self.model.predict(data_asframe)
            return ret

        explainer = shap.KernelExplainer(
            shap_predict, data, output_names=self.target_names)

        shap_values = explainer.shap_values(
            data, nsamples=200, l1_reg=f"num_features({len(self.feature_names)})")
        fig = plt.figure()

        # Docs for this summary plot: https://shap-lrjball.readthedocs.io/en/latest/generated/shap.summary_plot.html
        shap.summary_plot(shap_values, data, plot_type="bar",
                          show=False, class_names=self.target_names, plot_size="auto")

        # Decision Plot
        # shap.decision_plot(explainer.expected_value, shap_values,self.feature_names, show=False)

        plt.suptitle("Feature Impact on Model Prediction", fontsize=14)
        if self.single_class:
            plt.title("Warning: Only one target class present. Results may be misleading.",{'color':  'darkred'})
        plt.ylabel("Feature Attributes", fontsize=14)
        return fig


# tests implemented examples
if __name__ == "__main__":
    import doctest
    doctest.testmod()
# run python .\exdpn\guards\neural_network_guard.py from eXdpn file
