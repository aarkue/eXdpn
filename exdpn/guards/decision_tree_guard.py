from sklearn.tree import DecisionTreeClassifier, plot_tree
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from exdpn.data_preprocessing.data_preprocessing import apply_ohe
from exdpn.guards import Guard
from exdpn.data_preprocessing import fit_ohe

from pandas import DataFrame
from pm4py.objects.petri_net.obj import PetriNet
from typing import Dict


class Decision_Tree_Guard(Guard):
    def __init__(self, hyperparameters: Dict[str, any] = {'min_samples_split': 0.1, 'min_samples_leaf': 0.1, 'ccp_alpha': 0.2}) -> None:
        """Initializes a decision tree based guard with the provided hyperparameters
        Args:
            hyperparameters (dict[str, any]): Hyperparameters used for the classifier"""
        super().__init__(hyperparameters)
        # possible hyperparameters: max_depth, min_samples_split, min_samples_leaf
        try:
            self.model = DecisionTreeClassifier(**hyperparameters)
        except TypeError:
            raise TypeError(
                "Wrong hyperparameters were supplied to the decision tree guard")

        self.transition_int_map = None
        self.feature_names = None
        self.ohe = None
        self.ohe_columns = None

    def train(self, X: DataFrame, y: DataFrame) -> None:
        """Shall train the concrete classifier/model behind the guard using the dataframe and the specified hyperparameters.
        Args:
            X (DataFrame): Dataset used to train the classifier behind the guard (w/o the target label)
            y (DataFrame): Target label for each instance in the X dataset used to train the model"""
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
        # one hot encoding for categorical data
        input_instances = apply_ohe(input_instances, self.ohe)

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

    def get_explainable_representation(self) -> Figure:
        """Shall return an explainable representation of the guard. Shall throw an exception if the guard is not explainable.
        Returns:
            explainable_representation (Figure): Matplotlib Figure of the trained decision tree classifier"""
        fig, ax = plt.subplots()
        plot_tree(self.model,
                  ax=ax,
                  feature_names=self.feature_names,
                  class_names=[
                      t.label if t.label != None else f"None ({t.name})" for t in self.transition_int_map.keys()],
                  impurity=False,
                  filled=True)
        return fig
