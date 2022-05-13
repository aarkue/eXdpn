import abc  # use abstract base classes to define interfaces

from pandas import DataFrame
from pm4py.objects.petri_net.obj import PetriNet


class Guard(metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        """Nothing to do just yet."""
        pass

    @abc.abstractmethod
    def evaluate(self, dataframe: DataFrame) -> float:
        """Shall evaluate a concrete guard based on cross validation.
        Args:
            dataframe (DataFrame): Dataset used to evaluate the guard
        Returns:
            score (float): Evaluation result"""
        pass

    @abc.abstractmethod
    def train(self, dataframe: DataFrame) -> None:
        """Shall train the concrete classifier/model behind the guard using the dataframe.
        Args:
            dataframe (DataFrame): Dataset used to train the classifier behind the guard"""
        pass

    @abc.abstractmethod
    def predict(self, input_instance: list[any]) -> PetriNet.Transition:
        """Shall use the classifier/model behind the guard to predict the next transition.
        Args:
            input_instance (list[any]): Input instance used to predict the next transition"""
        pass
