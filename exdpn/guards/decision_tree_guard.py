from sklearn.tree import DecisionTreeClassifier, export_text
from exdpn.guards import Guard

from pandas import DataFrame
from pm4py.objects.petri_net.obj import PetriNet
from typing import Dict


class Decision_Tree_Guard(Guard):
    def __init__(self) -> None:
        super().__init__()
        self.model = None
        self.transition_int_map = None

    def train(self, X: DataFrame, y: DataFrame, hyperparameters: Dict[str, any] = {'min_samples_split': 0.1, 'min_samples_leaf': 0.1}) -> None:
        """Shall train the concrete classifier/model behind the guard using the dataframe and the specified hyperparameters.
        Args:
            X (np.ndarray): Dataset used to train the classifier behind the guard (w/o the target label)
            y (np.ndarray): Target label for each instance in the X dataset used to train the model
            hyperparameters (dict[str, any]): Hyperparameters used for the classifier"""

        # possible hyperparameters: max_depth, min_samples_split, min_samples_leaf
        try:
            model = DecisionTreeClassifier(**hyperparameters)
        except TypeError:
            raise TypeError(
                "Wrong hyperparameters were supplied to the decision tree guard")

        # make transition to integer (i.e. ID) map
        self.transition_int_map = {
            transition: index for index, transition in enumerate(list(set(y)))}
        y_transformed = [self.transition_int_map[transition]
                         for transition in y]

        self.model = model.fit(X, y_transformed)

    def predict(self, input_instance: list[any]) -> list[PetriNet.Transition]:
        """Shall use the classifier/model behind the guard to predict the next transition.
        Args:
            input_instance (list[any]): Input instances used to predict the next transition
        Returns:
            predicted_transitions (list[PetriNet.Transition]): Predicted transitions"""
        predicted_transition_ids = self.model.predict([input_instance])
        # ty stackoverflow
        return [next(trans for trans, trans_id in self.transition_int_map.items() if trans_id == pred_id) for pred_id in predicted_transition_ids]

    def is_explainable(self) -> bool:
        """Shall return wheter or not the internal classifier is explainable.
        Returns:
            explainable (bool): Wheter or not the guard is explainable"""
        return True

    def get_explainable_representation(self) -> str:
        """Shall return an explainable representation of the guard. Shall throw an exception if the guard is not explainable.
        Returns:
            explainable_representation (str): Explainable representation of the guard"""
        # TODO: add support for non generic feature names
        return export_text(self.model)
