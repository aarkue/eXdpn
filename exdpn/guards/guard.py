import abc  # use abstract base classes to define interfaces
from pandas import DataFrame
from pm4py.objects.petri_net.obj import PetriNet
from typing import Dict, List, Any 
from matplotlib.figure import Figure


class Guard(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, hyperparameters: Dict[str, Any]) -> None:
        """Initializes a guard with the provided hyperparameters.
        Args:
            hyperparameters (Dict[str, Any]): Hyperparameters used for the classifier
        """
        
        pass


    @abc.abstractmethod
    def train(self, X: DataFrame, y: DataFrame) -> None:
        """Shall train the concrete classifier/model behind the guard using the dataframe and the specified hyperparameters.
        Args:
            X (DataFrame): Feature variables of the provided dataset, used to train the classifier behind the guard 
            y (DataFrame): Target variable of the provided dataset, is to be predicted using X
        """
        
        pass


    @abc.abstractmethod
    def predict(self, input_instances: DataFrame) -> List[PetriNet.Transition]:
        """Shall use the classifier/model behind the guard to predict the next transition.
        Args:
            input_instances (DataFrame): Dataset of input instances used to predict the target variable, i.e., the next transition
        Returns:
            predicted_transitions (List[PetriNet.Transition]): Predicted transitions
        """
        
        pass


    @abc.abstractmethod
    def is_explainable(self) -> bool:
        """Shall return whether or not the internal classifier is explainable.
        Returns:
            explainable (bool): Whether or not the guard is explainable
        """
        
        pass


    @abc.abstractmethod
    def get_explainable_representation(self) -> Figure:
        """Shall return an explainable representation of the guard. Shall throw an exception if the guard is not explainable.
        Returns:
            explainable_representation (Figure): Explainable representation of the guard
        """
        
        pass