from pandas import DataFrame
from pm4py.objects.petri_net.obj import PetriNet
from typing import Dict

from exdpn.guards import ML_Technique # imports all guard classes
from exdpn.guards import Guard
from exdpn.data_preprocessing import data_preprocessing

from sklearn.metrics import f1_score


# idea: call Guard_Manager for each decision point to get the best possible guard model for either the default\
# machine learning techniques (all implemented) or the selected machine learning techniques

class Guard_Manager():
    def __init__(self, dataframe: DataFrame, ml_list: list[ML_Technique] = [ML_Technique.NN,
                                                                            ML_Technique.DT,
                                                                            ML_Technique.LG,
                                                                            ML_Technique.SVM]) -> Dict[str, Guard]:
        """Initializes all information needed for the calculation of the best guard for each decision point and /
        returns a dictionary with the list of all guards for each machine learning technique
        Args: 
            ml_list (list[ML_technique]): List of all machine learning techniques that should be evaluated, default is all \
                implemented techniques
            dataframe (DataFrame): Dataset used to evaluate the guard    
        Returns: 
            guards_list (Dict[str, Guard]): Returns a dictionary with all used machine learning techniques \
                mapped to the guards for the selected machine learning techniques       
        """
        self.dataframe = dataframe

        X_train, X_test, y_train, y_test = data_preprocessing(dataframe)
        self.X_train = X_train
        self.X_test  = X_test
        self.y_train = y_train
        self.y_test  = y_test

        # create list of all needed machine learning techniques to evaluate the guards
        self.ml_list = ml_list
        self.guards_list = {technique: technique.value() for technique in self.ml_list}

    def evaluate_guards(self) -> Dict[str, any]:
        """ Calculates for a given decision point all selected guards and returns the precision of the machine learning model, \
        using the specified machine learning techniques
        Returns:
            guards_results (Dict[str, any]): Returns a mapping of all selected machine learning techniques \
                to the achieved F1-score and the trained model
            """
        self.guards_results = {}
        # evaluate all selected ml techniques for all guards of the given decision point
        for guard_name, guard in self.guards_list.items():
            # currently proposed pipeline:
            # 1) guards_list[guard_name].train( train portion of data )
            guard.train(self.X_train, self.y_train)
            # 2) guards_list[guard_name].predict( test portion of data )
            print(type(self.X_test),type(self.y_test))
            y_prediction = guard.predict(self.X_test)
            # 3) calculate F1 score using desired and predicted transitions
            self.guards_results[guard_name] = f1_score(self.y_test, y_prediction)
            # decide on keeping model trained w/ train portion of data
            # or "retraining" the model w/ all data available
        return self.guards_results

    def get_best(self) -> tuple[ML_Technique, Guard]:
        """ Returns "best" guard for a decision point
        Returns: 
            best_guard (tuple[ML_Technique, Guard]): Returns "best" guard for a decision point with respect to the \
                chosen metric (F1 score), the returned tuple contains the machine learning technique and corresponding guard
            """
        best_guard_name = max(self.guards_results, key=self.guards_results.get)
        return best_guard_name, self.guards_list[best_guard_name]
