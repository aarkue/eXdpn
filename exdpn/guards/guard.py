import abc  # use abstract base classes to define interfaces

from pandas import DataFrame
from pm4py.objects.petri_net.obj import PetriNet
from typing import Dict


class Guard(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, hyperparameters: Dict[str, any]) -> None:
        """Initializes a guard with the provided hyperparameters
        Args:
            hyperparameters (dict[str, any]): Hyperparameters used for the classifier"""
        pass

    #@abc.abstractmethod
    #def evaluate(self, dataframe: DataFrame) -> tuple[float, Dict[str, any]]:
    #    """Shall evaluate a concrete guard and return it's score and optimal hyperparameters.
    #    Args:
    #        dataframe (DataFrame): Dataset used to evaluate the guard
    #    Returns:
    #        score (float): Evaluation result
    #        hyperparameters (Dict[str, any]): Parameter names mapped to their values"""
    #    pass

    @abc.abstractmethod
    def train(self, X: DataFrame, y: DataFrame) -> None:
        """Shall train the concrete classifier/model behind the guard using the dataframe and the specified hyperparameters.
        Args:
            X (np.ndarray): Dataset used to train the classifier behind the guard (w/o the target label)
            y (np.ndarray): Target label for each instance in the X dataset used to train the model"""
        pass

    @abc.abstractmethod
    def predict(self, input_instances: DataFrame) -> list[PetriNet.Transition]:
        """Shall use the classifier/model behind the guard to predict the next transition.
        Args:
            input_instance (list[any]): Input instance used to predict the next transition
        Returns:
            predicted_transitions (list[PetriNet.Transition]): Predicted transitions"""
        pass

    @abc.abstractmethod
    def is_explainable(self) -> bool:
        """Shall return wheter or not the internal classifier is explainable.
        Returns:
            explainable (bool): Wheter or not the guard is explainable"""
        pass

    @abc.abstractmethod
    def get_explainable_representation(self) -> str:
        """Shall return an explainable representation of the guard. Shall throw an exception if the guard is not explainable.
        Returns:
            explainable_representation (str): Explainable representation of the guard"""
        pass
