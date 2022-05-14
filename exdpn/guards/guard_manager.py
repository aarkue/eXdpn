from pandas import DataFrame
from pm4py.objects.petri_net.obj import PetriNet
from typing import Dict


class Guard_Manager():
    def __init__(self, guards_list: list[str], dataframe: DataFrame, ml_list: list[str] = ["NN", "DT", "LG", "SVM"]):
        """Initialize all information needed for the calculation of the best guard for each decision point
        Args: 
            ml_list (list[str]): List of all machine learning techniques that should be evaluated, default is all \
                implemented techniques
            guards_list (list[str]): Lists of all calculated guards for a place that is a decision point
            dataframe (DataFrame): Dataset used to evaluate the guard            
        """
        self.guards_list = guards_list
        self.dataframe = dataframe
        self.ml_list = ml_list 


    def evaluate_guards(self, guards_list: list[str], dataframe: DataFrame, ml_list: list[str]) -> Dict[list[any], list[float]]:
        """ Calculates for a given decision place all guards and the precision of a machine learning model, \
            using the specified machine learning techniques
        Args:
            ml_list (list[str]): List of all machine learning techniques that should be evaluated
            guards_list (list): Lists of all guards for a place that is a decision point
            dataframe (DataFrame): Dataset used to evaluate the guard
        Returns:
            guards_results (Dict[list[any], list[float]]): Returns for each guard a list of the precision \
                metric results for each machine learning technique
            """
        guards_results = []
        # evaluate all selected ml techniques for all guards of the given decision point
        for guard in guards_list:
            # TODO synatx anpassen, wenn ml techniques implementiert sind
            guards_results[guard] = [eval(ml_technique+"_guard")(dataframe) for ml_technique in ml_list]
        return guards_results

    
    def get_best(self, guards_results: Dict[list[any], list[float]]) -> Dict[list[any], float]:
        """ Returns "best" guard for a decision point
        Args: 
            guards_results (Dict[list[any], list[float]]): List of the different precision metric results for each \
                machine learning technique for each guard
        Returns: 
            best_guard (Dict[list[any], float]): Returns "best" guard for a decision point with respect to the \
                chosen precision metric
            """
        # TODO decide what precision metric should be used
        #best_guard = [min(guards_results[place]) for place in guards_results.keys()]
        #best_guard = [max(guards_results[place]) for place in guards_results.keys()]

        #return best_guard 
        pass
