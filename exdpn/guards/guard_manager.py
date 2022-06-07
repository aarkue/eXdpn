from pandas import DataFrame, concat
from exdpn.data_preprocessing.data_preprocessing import basic_data_preprocessing
from pm4py.objects.petri_net.obj import PetriNet
from typing import Dict, List, Tuple

from exdpn.guards import ML_Technique # imports all guard classes
from exdpn.guards import Guard
from exdpn.data_preprocessing import data_preprocessing_evaluation 

from sklearn.metrics import f1_score


# idea: call Guard_Manager for each decision point to get the best possible guard model for either the default\
# machine learning techniques (all implemented) or the selected machine learning techniques

class Guard_Manager():
    def __init__(self, 
                 dataframe: DataFrame, 
                 numeric_attributes: List[str], 
                 ml_list: List[ML_Technique]) -> None:
        """Initializes all information needed for the calculation of the best guard for each decision point and /
        returns a dictionary with the list of all guards for each machine learning technique
        Args: 
            ml_list (List[ML_technique]): List of all machine learning techniques that should be evaluated
            dataframe (DataFrame): Dataset used to evaluate the guard    
        Returns: 
            guards_List (Dict[str, Guard]): Returns a dictionary with all used machine learning techniques \
                mapped to the guards for the selected machine learning techniques       
        """
        # TODO: refactor data_preprocessing so that it does not do more than one thing
        # or does all the things

        # TODO: think about persistence of the encoders so that new unseen instances can still be encoded

        X_train, X_test, y_train, y_test = data_preprocessing_evaluation(dataframe, numeric_attributes)
        
        #self.dataframe = dataframe
        self.X_train = X_train
        self.X_test  = X_test
        self.y_train = y_train
        self.y_test  = y_test

        # create list of all needed machine learning techniques to evaluate the guards
        self.ml_list = ml_list
        self.guards_list = {technique: technique.value() for technique in self.ml_list}
        self.guards_results = None


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
            guard.train(self.X_train, self.y_train)
            
            y_prediction = guard.predict(self.X_test)

            # convert Transition objects to integers so that sklearn's F1 score doesn't freak out
            # this is ugly, we know
            transition_int_map = {transition: index for index, transition in enumerate(list(set(y_prediction + self.y_test.tolist())))}
            y_prediction_transformed = [transition_int_map[transition] for transition in y_prediction]
            y_test_transformed = [transition_int_map[transition] for transition in self.y_test.tolist()]

            self.guards_results[guard_name] = f1_score(y_test_transformed, y_prediction_transformed, average="weighted")
            
            # TODO retrain model on all available data, decide at which point would be best
            #df_X, df_y = basic_data_preprocessing(self.dataframe)
            #guard.train(df_X, df_y)
        
        return self.guards_results


    def get_best(self) -> Tuple[ML_Technique, Guard]:
        """ Returns "best" guard for a decision point
        Returns: 
            best_guard (Tuple[ML_Technique, Guard]): Returns "best" guard for a decision point with respect to the \
                chosen metric (F1 score), the returned tuple contains the machine learning technique and corresponding guard
            """
        assert self.guards_results != None, "Guards must be evaluated first"
        best_guard_name = max(self.guards_results, key=self.guards_results.get)
        
        return best_guard_name, self.guards_list[best_guard_name]
