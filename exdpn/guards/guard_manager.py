from pandas import DataFrame
from exdpn.data_preprocessing.data_preprocessing import basic_data_preprocessing

from typing import Dict, List, Tuple, Any 

from exdpn.guards import ML_Technique # imports all guard classes
from exdpn.guards import Guard
from exdpn.data_preprocessing import data_preprocessing_evaluation 
from exdpn.guards.model_builder import model_builder
from sklearn.metrics import f1_score


# idea: call Guard_Manager for each decision point to get the best possible guard model for either the default\
# machine learning techniques (all implemented) or the selected machine learning techniques

class Guard_Manager():
    def __init__(self, 
                 dataframe: DataFrame, 
                 ml_list: List[ML_Technique] = [
                    ML_Technique.DT,
                    ML_Technique.LR,
                    ML_Technique.SVM,
                    ML_Technique.NN],
                 hyperparameter: Dict[ML_Technique, Dict[str, Any]] = {ML_Technique.NN: {'hidden_layer_sizes': (10, 10)},
                                                                        ML_Technique.DT: {'min_samples_split': 0.1, 
                                                                                          'min_samples_leaf': 0.1, 
                                                                                          'ccp_alpha': 0.2},
                                                                        ML_Technique.LR: {"C": 0.5},
                                                                        ML_Technique.SVM: {"C": 0.5}}
    ) -> None:
        """Initializes all information needed for the calculation of the best guard for each decision point and /
        returns a dictionary with the list of all guards for each machine learning technique.
        Args:
            ml_list (List[ML_Technique]): List of all machine learning techniques that should be evaluated
            dataframe (DataFrame): Dataset used to evaluate the guard        
        """
        
        # TODO: refactor data_preprocessing so that it does not do more than one thing
        # or does all the things

        # TODO: think about persistence of the encoders so that new unseen instances can still be encoded

        X_train, X_test, y_train, y_test = data_preprocessing_evaluation(dataframe)
        
        self.dataframe = dataframe
        self.X_train = X_train
        self.X_test  = X_test
        self.y_train = y_train
        self.y_test  = y_test

        # create list of all needed machine learning techniques to evaluate the guards
        self.guards_list = {technique: model_builder(technique, hyperparameter[technique]) for technique in ml_list}
        
        self.guards_results = None


    def train_test(self) -> Dict[str, Any]:
        """ Calculates for a given decision point all selected guards and returns the precision of the machine learning model, \
        using the specified machine learning techniques.
        Returns:
            guards_results (Dict[str, Any]): Returns a mapping of all selected machine learning techniques \
            to the achieved F1-score and two trained guard models: the "training" guard (position 0) and final guard (position 1)
        """
        
        self.guards_results = {}
        # evaluate all selected ml techniques for all guards of the given decision point
        for guard_name, guard_models in self.guards_list.items():
            guard_models.train(self.X_train, self.y_train)
            y_prediction = guard_models.predict(self.X_test)
             
            # convert Transition objects to integers so that sklearn's F1 score doesn't freak out
            # this is ugly, we know
            transition_int_map = {transition: index for index, transition in enumerate(list(set(y_prediction + self.y_test.tolist())))}
            y_prediction_transformed = [transition_int_map[transition] for transition in y_prediction]
            y_test_transformed = [transition_int_map[transition] for transition in self.y_test.tolist()]

            self.guards_results[guard_name] = f1_score(y_test_transformed, y_prediction_transformed, average="weighted")
            
            # retrain model on all available data
            df_X, df_y = basic_data_preprocessing(self.dataframe)
            guard_models_temp = guard_models
            guard_models_temp.train(df_X, df_y)
        
        return self.guards_results


    def get_best(self) -> Tuple[str, List[Guard]]:
        """ Returns "best" guard for a decision point.
        Returns:
            best_guard (Tuple[str, List[Guard, Guard]]): Returns "best" guard for a decision point with respect to the \
            chosen metric (F1 score), the returned tuple contains the machine learning technique and a list with the \
            corresponding "training" guard (position 0) and final guard (position 1)
        """
        
        assert self.guards_results != None, "Guards must be evaluated first"
        best_guard_name = max(self.guards_results, key=self.guards_results.get)
        
        return best_guard_name, self.guards_list[best_guard_name]
