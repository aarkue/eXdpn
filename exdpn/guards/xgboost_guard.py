"""
.. include:: ./../../docs/_templates/md/guards/guard.md

"""

import io
from xgboost import XGBClassifier
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from exdpn.data_preprocessing.data_preprocessing import apply_ohe
from exdpn.guards import Guard
from exdpn.data_preprocessing import fit_ohe

from pandas import DataFrame
from pm4py.objects.petri_net.obj import PetriNet
from typing import Dict, List, Any, Optional, Union
import numpy as np
import shap


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
            >>> # return figure of explainable representation
            >>> fig = guard.get_explainable_representation(X_test) # results may deviate 

            <img src="../../images/svm-example-representation.svg" alt="Example explainable representation of a XGBoost guard" style="max-height: 350px;"/>

            .. include:: ../../docs/_templates/md/example-end.md

        Note: 
            For an example of the explainable representations of all machine learning techniques please check [Data Petri Net Example](https://github.com/aarkue/eXdpn/blob/main/docs/dpn_example.ipynb).
        """

        if self.is_explainable() == False:
            raise Exception(
                "Guard is not explainable and therefore has no explainable representation")

        classes = [t.label if t.label !=
                   None else f"None ({t.name})" for t in self.transition_int_map.keys()]
        # one hot encoding for categorical data
        data = apply_ohe(data, self.ohe)

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(data)

        # Docs for this summary plot: https://shap-lrjball.readthedocs.io/en/latest/generated/shap.summary_plot.html
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values,
                          data,
                          plot_type="bar",
                          show=False,
                          class_names=classes,
                          class_inds=range(len(classes)))
        plt.title("Feature Impact on Model Prediction", fontsize=14)
        plt.ylabel("Feature Attributes", fontsize=14)

        return fig
    
    def get_global_explanations(self, base_sample: DataFrame) -> Dict[str,Union[Figure,str]]:
        """Get a global explainable representation for the concrete machine learning classifier.
        Args:
            base_sample (DataFrame): A small (10-30) sample of the population for this decision point; Used for calculation of shap values.
        Returns:
            Dict[str,Figure]: A dictionary containing the global explainable representations. Containing the following entries:
            - "Bar plot (Summary)"
            - "Beeswarm plot for `X`" (for all output labels X)
            - "Force plot for `X`" (for all output labels X)
        """

        # one hot encoding for categorical data
        processed_base_sample = apply_ohe(base_sample, self.ohe)
        unscaled_base_sample = processed_base_sample
    
        def shap_predict(data: np.ndarray):
                    data_asframe = DataFrame(data, columns=self.feature_names)
                    ret = self.model.predict_proba(data_asframe)
                    return ret

        explainer = shap.KernelExplainer(shap_predict, processed_base_sample)

        # shap_values = explainer.shap_values(processed_base_sample)
        shap_values = explainer.shap_values(processed_base_sample, nsamples=300, l1_reg=f"num_features({len(self.feature_names)})")
        target_names = [t.label if t.label !=
                    None else f"None ({t.name})" for t in self.transition_int_map.keys()]

        ret = dict()
        fig = plt.figure()
        print(target_names)
        shap.summary_plot(shap_values, unscaled_base_sample, plot_type='bar', class_names=target_names, use_log_scale=False,max_display=10, show=False)
        ret['Bar plot (Summary)'] = fig;

        for key in range(len(target_names)):
            print(target_names[key])
            fig = plt.figure()
            shap.plots.beeswarm(shap.Explanation(values=shap_values[key], 
                                                            base_values=explainer.expected_value[key], data=unscaled_base_sample,  
                                                    feature_names=self.feature_names), show=False)
            ret[f"Beeswarm plot for {target_names[key]}"] = fig

            force_plot = shap.force_plot(explainer.expected_value[[key]],shap_values[key],features=unscaled_base_sample, out_names=target_names[key], link='logit',show=False)
            html_data = io.StringIO()
            shap.save_html(html_data,force_plot,full_html=False)
            html_data.seek(0)  # rewind the data
            ret[f"Force plot for {target_names[key]}"] = html_data.getvalue()
        return ret

    def get_local_explanations(self, local_data: DataFrame, base_sample: DataFrame) -> Dict[str, Figure]:
        assert local_data.shape[0] == 1
        # Pre-process local_data
        # One-Hot Encoding for categorical data
        processed_local_data = apply_ohe(local_data, self.ohe)

        # Pre-process base_sample
        # One-Hot Encoding for categorical data
        processed_base_sample = apply_ohe(base_sample, self.ohe)

        target_names = [t.label if t.label !=
                        None else f"None ({t.name})" for t in self.transition_int_map.keys()]

        def shap_predict(data: np.ndarray):
            data_asframe = DataFrame(data, columns=self.feature_names)
            ret = self.model.predict_proba(data_asframe)
            return ret

        predictions = shap_predict(processed_local_data)

        # TreeExplainer throws errors for me
        explainer = shap.KernelExplainer(
            shap_predict, processed_base_sample, output_names=target_names)
        single_shap = explainer.shap_values(
            processed_local_data, nsamples=200, l1_reg=f"num_features({len(self.feature_names)})")

        ret = dict()
        fig = plt.figure()
        shap.multioutput_decision_plot(list(explainer.expected_value), single_shap,
                                       features=processed_local_data, row_index=0, feature_names=self.feature_names,
                                       highlight=[np.argmax(predictions[0])], link='logit', legend_labels=target_names,
                                       legend_location="lower right", feature_display_range=slice(-1, -11, -1), show=False)
        ret['Decision plot (Multioutput)'] = fig

        winner_index = np.argmax(predictions[0])
        for key in range(len(single_shap)):
            fig = plt.figure()
            shap.decision_plot(list(explainer.expected_value)[key], single_shap[key], features=processed_local_data, link='logit',
                               legend_labels=[target_names[key]], feature_display_range=slice(-1, -11, -1), show=False, highlight=0 if (winner_index == key) else None)
            ret[f"Decision plot for {target_names[key]}"] = fig

            # fig = plt.figure()
            fig = shap.force_plot(explainer.expected_value[key],
                                  single_shap[key],
                                  processed_local_data, out_names=target_names[key], matplotlib=True,
                                  #link='logit',
                                  contribution_threshold=0.1, show=False)
            fig = plt.gcf()
            ret[f"Force plot for {target_names[key]}"] = fig
        return ret


# tests implemented examples
if __name__ == "__main__":
    import doctest
    doctest.testmod()
# run python .\exdpn\guards\xgboost_guard.py from eXdpn file
