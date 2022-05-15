from pandas import DataFrame
from pm4py.objects.petri_net.obj import PetriNet
from typing import Dict

# idea: call Guard_Manager for each decision point to get the best possible guard model for either the default\
# machine learning techniques (all implemented) or the selected machine learning techniques

class Guard_Manager():
    def __init__(self, dataframe: DataFrame, ml_list: list[str] = ["NN", "DT", "LG", "SVM"]) -> Dict[str, Guard]:
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
        self.guards_list = {key: eval(key+"Guard")() for key in self.ml_list}

    # TODO output an output von Guard.evaluate anpassen
    def evaluate_guards(self) -> Dict[str, any]:
        """ Calculates for a given decision point all selected guards and returns the precision of the machine learning model, \
        using the specified machine learning techniques
        Returns:
            guards_results (Dict[str, any]): Returns a mapping of all selected machine learning techniques \
                to the achieved precision and the corresponding model [TBD]
            """
        self.guards_results = {} 
        # evaluate all selected ml techniques for all guards of the given decision point
        for guard in self.guards_list:
            self.guards_results[guard] = [self.guards_list[guard].evaluate(self.dataframe)] 
        return self.guards_results 

    # TODO an output von evaluate_guards anpassen
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
