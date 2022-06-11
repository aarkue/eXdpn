from exdpn.guards import Decision_Tree_Guard, Logistic_Regression_Guard, SVM_Guard, Neural_Network_Guard
from exdpn.guards import Guard
from typing import Dict
    
def model_builder(model_type: str, hp: Dict[str, any]) -> Guard:
    """Internal function to build a specific guard with the provided hyperparameters.
    Args:
        model_typ (str): Specification of machine learning technique, use "NN" for neural network, \
        "SVM" for support vector machine, "DT" for decision tree and "LR" for logistic regression
        hp (Dict[str, any]): Hyperparameters for the machine learning model
    Returns:
        Guard: Machine learning guard of desired type with provided hyperparameters
    """
    if model_type == "SVM":
        return SVM_Guard(hp)
    elif model_type == "NN":
        return Neural_Network_Guard(hp)
    elif model_type == "LR":
        return Logistic_Regression_Guard(hp)
    elif model_type == "DT":
        return Decision_Tree_Guard(hp)
    else:
        raise TypeError ("Guard not implemented")