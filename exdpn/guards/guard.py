import abc
from pandas import DataFrame
from typing import Dict, List, Any, Optional
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
    def get_explainable_representation(self, data: Optional[DataFrame]) -> Figure:
        """Return an explainable representation of the concrete machine learning classifier.
        Args:
            data (DataFrame, optional): Dataset of input instances used to construct an explainable representation (not needed for some techniques (Decision Trees)).
        Returns:
            Figure: The explainable representation of the concrete machine learning classifier.

        """
        pass

    @abc.abstractclassmethod
    def get_local_explanations(self, local_data:DataFrame, base_sample: DataFrame) -> Dict[str,Figure]:
        """Get explainable representations for a single decision situation. 

        Args:
            local_data (DataFrame): A dataframe containing the single decision situation.
            base_sample (DataFrame): A small (10-30) sample of the population for this decision point; Used for calculation of shap values.

        Returns:
            Dict[str,Figure]: A dictionary containing the explainable representations for the single decision situation. Containing the following entries:
            - "Decision plot (Multioutput)"
            - "Decision plot for `X`" (for all output labels X)
            - "Force plot for `X`" (for all output labels X)
                
        """        
        pass