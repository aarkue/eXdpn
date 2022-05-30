from sklearn.tree import DecisionTreeClassifier, export_text
from data_preprocessing.data_preprocessing import apply_ohe, apply_scaling, fit_scaling
from guards import Guard
from data_preprocessing import fit_ohe

from pandas import DataFrame, Series
from pm4py.objects.petri_net.obj import PetriNet
from typing import Dict, List
from re import sub


class Decision_Tree_Guard(Guard):
    def __init__(self, hyperparameters: Dict[str, any] = {'min_samples_split': 0.1, 'min_samples_leaf': 0.1}) -> None:
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
        self.feature_names      = None
        self.ohe                = None
        self.ohe_columns        = None
        self.scaler             = None
        self.scaler_columns     = None

    def train(self, X: DataFrame, y: DataFrame) -> None:
        """Shall train the concrete classifier/model behind the guard using the dataframe and the specified hyperparameters.
        Args:
            X (DataFrame): Dataset used to train the classifier behind the guard (w/o the target label)
            y (DataFrame): Target label for each instance in the X dataset used to train the model"""
        # scale numerical attributes
        self.scaler, self.scaler_columns = fit_scaling(X)
        X = apply_scaling(X, self.scaler, self.scaler_columns)
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

    def predict(self, input_instances: DataFrame) -> List[PetriNet.Transition]:
        """Shall use the classifier/model behind the guard to predict the next transition.
        Args:
            input_instances (DataFrame): Input instances used to predict the next transition
        Returns:
            predicted_transitions (list[PetriNet.Transition]): Predicted transitions"""
        # scale numerical attributes
        input_instances = apply_scaling(input_instances, self.scaler, self.scaler_columns)
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

    def get_explainable_representation(self) -> str:
        """Shall return an explainable representation of the guard. Shall throw an exception if the guard is not explainable.
        Returns:
            explainable_representation (str): Explainable representation of the guard"""
        representation = export_text(
            self.model, feature_names=self.feature_names)
        for transition, transition_int in self.transition_int_map.items():
            representation = representation.replace(
                f"class: {transition_int}", f"class: {transition.name} / {transition.label}")

        # inverse scaler
        dummy = DataFrame([[0 for _ in self.feature_names]], columns=self.feature_names)
        # this is ugly, we know
        for scale_column in self.scaler_columns:
            pattern = fr'{scale_column} <= (.(\d*.\d*)?)'
            def inverse_transform_single(match):
                dummy[scale_column] = float(match.group(1))
                trans = DataFrame(self.scaler.inverse_transform(dummy), columns=self.feature_names)
                return f'{scale_column} <= {trans[scale_column][0]}'
            representation = sub(pattern, inverse_transform_single, representation)

            pattern = fr'{scale_column} >  (.(\d*.\d*)?)'
            def inverse_transform_single(match):
                dummy[scale_column] = float(match.group(1))
                trans = DataFrame(self.scaler.inverse_transform(dummy), columns=self.feature_names)
                return f'{scale_column} >  {trans[scale_column][0]}'
            representation = sub(pattern, inverse_transform_single, representation)

        # 'inverse' OHE
        for ohe_column in self.ohe_columns:
            pattern = fr'{ohe_column}_(.*?) <= 0.50'
            replacement = fr'{ohe_column} != \1'
            representation = sub(pattern, replacement, representation)

            pattern = fr'{ohe_column}_(.*?) >  0.50'
            replacement = fr'{ohe_column} = \1'
            representation = sub(pattern, replacement, representation)

        return representation
