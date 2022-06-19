import abc
from pandas import DataFrame
from typing import Dict, List, Any 
from matplotlib.figure import Figure

from pm4py.objects.petri_net.obj import PetriNet


class Guard(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, hyperparameters: Dict[str, Any]) -> None:
        """Abstract class defining the guard interface.

        Args:
            hyperparameters (Dict[str, Any]): The hyperparameters used for the concrete machine learning classifier initialization.

        """
        pass


    @abc.abstractmethod
    def train(self, X: DataFrame, y: DataFrame) -> None:
        """Train the concrete machine learning classifier using the provided dataset.
        Args:
            X (DataFrame): The feature variables, used to train the classifier behind the guard.
            y (DataFrame): The target variable of the provided dataset.

        Note:
            It is assumed that `y` corresponds to a list of `pm4py.objects.petri_net.obj.PetriNet.Transition` objects.

        """
        pass


    @abc.abstractmethod
    def predict(self, input_instances: DataFrame) -> List[PetriNet.Transition]:
        """Uses the concrete machine learning classifier to predict a transition.

        Args:
            input_instances (DataFrame): The dataset of input instances used to predict the target variable.

        Returns:
            List[PetriNet.Transition]: The predicted transitions.

        """
        pass


    @abc.abstractmethod
    def is_explainable(self) -> bool:
        """Returns whether or not the concrete machine learning classifier is explainable.

        Returns:
            bool: Whether or not the concrete machine learning classifier is explainable.
        """
        pass


    @abc.abstractmethod
    def get_explainable_representation(self) -> Figure:
        """Return an explainable representation of the concrete machine learning classifier.

        Returns:
            Figure: The explainable representation of the concrete machine learning classifier.

        """
        pass