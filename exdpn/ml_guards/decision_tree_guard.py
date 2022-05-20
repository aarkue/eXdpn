from sklearn.tree import DecisionTreeClassifier
from pandas import DataFrame
from pm4py.objects.petri_net.obj import PetriNet
from typing import Dict
from exdpn.guards.guard import Guard

class Decision_Tree_Guard(Guard):
    def train(self, dataframe: DataFrame, hyperparameters: Dict[str, any] = None) -> None:
        """Shall train the concrete classifier/model behind the guard using the dataframe and the specified hyperparameters.
        Args:
            dataframe (DataFrame): Dataset used to train the classifier behind the guard
            hyperparameters (dict[str, any]): Hyperparameters used for the classifier"""
        # possible hyperparameters: max_depth, min_samples_split, min_samples_leaf
        






    def predict(self, input_instance: list[any]) -> PetriNet.Transition:
        """Shall use the classifier/model behind the guard to predict the next transition.
        Args:
            input_instance (list[any]): Input instance used to predict the next transition"""
        pass

    def is_explainable(self) -> bool:
        """Shall return wheter or not the internal classifier is explainable.
        Returns:
            explainable (bool): Wheter or not the guard is explainable"""
        pass

    def get_explainable_representation(self) -> str:
        """Shall return an explainable representation of the guard. Shall throw an exception if the guard is not explainable.
        Returns:
            explainable_representation (str): Explainable representation of the guard"""
        pass