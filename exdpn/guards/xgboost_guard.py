"""
.. include:: ./../../docs/_templates/md/guards/guard.md

"""

from xgboost import XGBClassifier
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from exdpn.data_preprocessing.data_preprocessing import apply_ohe
from exdpn.guards import Guard
from exdpn.data_preprocessing import fit_ohe

from pandas import DataFrame
from pm4py.objects.petri_net.obj import PetriNet
from typing import Dict, List, Any, Optional
import numpy as np


class XGBoost_Guard(Guard):
    def __init__(self, hyperparameters: Dict[str, Any] = {}) -> None:
        """Initializes a XGBoost based guard with the provided hyperparameters.

        Args:
            hyperparameters (Dict[str, Any]): Hyperparameters used for the XGBoost classifier.

        Raises:
            TypeError: If supplied hyperparameters are invalid.

        Examples:

            >>> from exdpn.guards import XGBoost_Guard
            >>> guard = XGBoost_Guard()


            .. include:: ../../docs/_templates/md/example-end.md
        """

        super().__init__(hyperparameters)

        # TODO decide on OHE vs (no OHE and approx tree method)
        # overwrite tree_method and enable_categorical so that
        # categorical data may be used. Since this is an
        # experimental feature of xgboost, we need to change
        # the default tree_method to approx/hist/gpu_hist.
        # \Note that the `enable_categorical` and `tree_method` will be forcefully set to \`True` and `approx` respectively, so that categorical data may be used for prediction.
        #hyperparameters['enable_categorical'] = True
        #hyperparameters['tree_method'] = 'approx'

        try:
            self.model = XGBClassifier(**hyperparameters)
        except TypeError:
            raise TypeError(
                "Wrong hyperparameters were supplied to the XGBoost guard")

        self.transition_int_map = None
        self.feature_names = None
        self.ohe = None

    def train(self, X: DataFrame, y: DataFrame) -> None:
        """Trains the XGBoost guard using the dataset.
        Args:
            X (DataFrame): Feature variables of the provided dataset, used to train the XGBoost classifier.
            y (DataFrame): Target variable of the provided dataset, is to be predicted using `X`.

        Examples:

            >>> from exdpn.util import import_log
            >>> from exdpn.petri_net import get_petri_net
            >>> from exdpn.guard_datasets import extract_all_datasets
            >>> from exdpn.guards import XGBoost_Guard
            >>> from exdpn.data_preprocessing import data_preprocessing_evaluation
            >>> event_log = import_log('./datasets/p2p_base.xes')
            >>> pn, im, fm = get_petri_net(event_log)
            >>> dp_dataset_map = extract_all_datasets(event_log, pn, im, fm, 
            ...                                       event_level_attributes = ['item_category','item_id','item_amount','supplier','total_price'], 
            ...                                       activityName_key = "concept:name")
            >>> # select a certain decision point and the corresponding data set 
            >>> dp = list(dp_dataset_map.keys())[0]
            >>> dp_dataset = dp_dataset_map[dp]
            >>> X_train, X_test, y_train, y_test = data_preprocessing_evaluation(dp_dataset)
            >>> guard = XGBoost_Guard()
            >>> guard.train(X_train, y_train)


            .. include:: ../../docs/_templates/md/example-end.md
        """

        # one hot encoding for categorical data
        self.ohe = fit_ohe(X)
        X = apply_ohe(X, self.ohe)

        # store feature names for the explainable representation
        self.feature_names = list(X.columns)

        # make transition to integer (i.e. ID) map
        self.transition_int_map = {
            transition: index for index, transition in enumerate(list(set(y)))}
        y_transformed = [self.transition_int_map[transition]
                         for transition in y]

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

            >>> from exdpn.util import import_log
            >>> from exdpn.petri_net import get_petri_net
            >>> from exdpn.guard_datasets import extract_all_datasets
            >>> from exdpn.guards import XGBoost_Guard
            >>> from exdpn.data_preprocessing import data_preprocessing_evaluation
            >>> event_log = import_log('./datasets/p2p_base.xes')
            >>> pn, im, fm = get_petri_net(event_log)
            >>> dp_dataset_map = extract_all_datasets(event_log, pn, im, fm,
            ...                                       event_level_attributes = ['item_category','item_id','item_amount','supplier','total_price'], 
            ...                                       activityName_key = "concept:name")
            >>> # select a certain decision point and the corresponding data set 
            >>> dp = list(dp_dataset_map.keys())[0]
            >>> dp_dataset = dp_dataset_map[dp]
            >>> X_train, X_test, y_train, y_test = data_preprocessing_evaluation(dp_dataset)
            >>> guard = XGBoost_Guard()
            >>> guard.train(X_train, y_train)
            >>> y_prediction = guard.predict(X_test)

            .. include:: ../../docs/_templates/md/example-end.md
        """

        # one hot encoding for categorical data
        input_instances = apply_ohe(input_instances, self.ohe)

        predicted_transition_ids = self.model.predict(input_instances)
        # ty stackoverflow
        # finds the key (transition) where the value (transition integer / id) corresponds to the predicted integer / id
        # for all predicted integers
        return [next(trans for trans, trans_id in self.transition_int_map.items() if trans_id == pred_id) for pred_id in predicted_transition_ids]

    def is_explainable(self) -> bool:
        """Returns whether or not this guard is explainable.

        Returns:
            bool: Whether or not the guard is explainable.

        Examples:

            >>> from exdpn.guards import XGBoost_Guard
            >>> guard = XGBoost_Guard()
            >>> guard.is_explainable()
            True


            .. include:: ../../docs/_templates/md/example-end.md
        """

        return True

    def get_explainable_representation(self, data: Optional[DataFrame] = None) -> Figure:
        """Returns an explainable representation of the XGBoost guard.
        Args:
            data (DataFrame, optional): Dataset of input instances used to construct an explainable representation. 

        Returns:
            Figure: Matplotlib Figure of the trained decision tree classifier.

        Raises:
            Exception: If the guard has no explainable representation.

        Examples:
            # TODO
            >>> from exdpn.util import import_log
            >>> from exdpn.petri_net import get_petri_net
            >>> from exdpn.guard_datasets import extract_all_datasets
            >>> from exdpn.guards import Decision_Tree_Guard
            >>> from exdpn.data_preprocessing import data_preprocessing_evaluation
            >>> event_log = import_log('./datasets/p2p_base.xes')
            >>> pn, im, fm = get_petri_net(event_log)
            >>> dp_dataset_map = extract_all_datasets(event_log, pn, im, fm,
            ...                                       event_level_attributes = ['item_category','item_id','item_amount','supplier','total_price'], 
            ...                                       activityName_key = "concept:name")
            >>> # select a certain decision point and the corresponding data set 
            >>> dp = list(dp_dataset_map.keys())[0]
            >>> dp_dataset = dp_dataset_map[dp]
            >>> X_train, X_test, y_train, y_test = data_preprocessing_evaluation(dp_dataset)
            >>> guard = Decision_Tree_Guard()
            >>> guard.train(X_train, y_train)
            >>> y_prediction = guard.predict(X_test)
            >>> # return figure of explainable representation
            >>> fig = guard.get_explainable_representation() # results may deviate 

            <img src="../../images/dt-example-representation.svg" alt="Example explainable representation of a decision tree guard" style="max-height: 350px;"/>

            .. include:: ../../docs/_templates/md/example-end.md

        Note: 
            For an example of the explainable representations of all machine learning techniques please check [Data Petri Net Example](https://github.com/aarkue/eXdpn/blob/main/docs/dpn_example.ipynb).
        """

        if self.is_explainable() == False:
            raise Exception(
                "Guard is not explainable and therefore has no explainable representation")
        # TODO
        return None


# tests implemented examples
if __name__ == "__main__":
    import doctest
    doctest.testmod()
# run python .\exdpn\guards\xgboost_guard.py from eXdpn file
