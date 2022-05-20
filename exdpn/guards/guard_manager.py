from pandas import DataFrame
from pm4py.objects.petri_net.obj import PetriNet
from typing import Dict

from ml_technique import ML_Technique
from guards import Guard
# TODO: import all concrete Guard implementations


# idea: call Guard_Manager for each decision point to get the best possible guard model for either the default\
# machine learning techniques (all implemented) or the selected machine learning techniques

class Guard_Manager():
    def __init__(self, dataframe: DataFrame, ml_list: list[str] = [ML_Technique.NN, 
                                                                   ML_Technique.DT, 
                                                                   ML_Technique.LG, 
                                                                   ML_Technique.SVM]) -> Dict[str, Guard]:
        """Initializes all information needed for the calculation of the best guard for each decision point and /
        returns a dictionary with the list of all guards for each machine learning technique
        Args: 
            ml_list (list[str]): List of all machine learning techniques that should be evaluated, default is all \
                implemented techniques
            dataframe (DataFrame): Dataset used to evaluate the guard    
        Returns: 
            guards_list (Dict[str, Guard]): Returns a dictionary with all used machine learning techniques \
                mapped to the guards for the selected machine learning techniques       
        """
        self.dataframe = dataframe
        self.ml_list = ml_list 
        # create list of all needed machine learning techniques to evaluate the guards
        self.guards_list = {technique: eval(technique.name + "_Guard")() for technique in self.ml_list}


    def evaluate_guards(self) -> Dict[str, any]:
        """ Calculates for a given decision point all selected guards and returns the precision of the machine learning model, \
        using the specified machine learning techniques
        Returns:
            guards_results (Dict[str, any]): Returns a mapping of all selected machine learning techniques \
                to the achieved F1-score and the trained model
            """
        self.guards_results = {} 
        # evaluate all selected ml techniques for all guards of the given decision point
        for guard_name in self.guards_list:
            # currently proposed pipeline:
            # 1) guards_list[guard_name].train( train portion of data )
            # 2) guards_list[guard_name].predict( test portion of data )
            # 3) calculate F1 score using desired and predicted transitions

            # decide on keeping model trained w/ train portion of data
            # or "retraining" the model w/ all data available
            pass
        return self.guards_results 


    def get_best(self) -> tuple[str, any]:
        """ Returns "best" guard for a decision point
        Returns: 
            best_guard (Dict[list[any], float]): Returns "best" guard for a decision point with respect to the \
                chosen precision metric, return contains name of machine learning technique and corresponding guard model
            """
        # TODO decide what precision metric should be used
        #best_guard = [min(guards_results[place]) for place in guards_results.keys()]
        #best_guard = [max(guards_results[place]) for place in guards_results.keys()]

        #return best_guard 
        pass
