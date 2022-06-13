from exdpn.guards import Decision_Tree_Guard, Logistic_Regression_Guard, SVM_Guard, Neural_Network_Guard
from exdpn.guards import Guard
from typing import Dict, Any 
from exdpn.guards import ML_Technique

    
def model_builder(model_type: ML_Technique, hp: Dict[str, Any]) -> Guard:
    """Internal function to build a specific guard with the provided hyperparameters.
    Args:
        model_typ (ML_Technique): Specification of machine learning technique, use "NN" for neural network, \
        "SVM" for support vector machine, "DT" for decision tree and "LR" for logistic regression
        hp (Dict[str, any]): Hyperparameters for the machine learning model
    Returns:
        Guard: Machine learning guard of desired type with provided hyperparameters
    """
    if model_type == ML_Technique.SVM:
        return SVM_Guard(hp)
    elif model_type == ML_Technique.NN:
        return Neural_Network_Guard(hp)
    elif model_type == ML_Technique.LR:
        return Logistic_Regression_Guard(hp)
    elif model_type == ML_Technique.DT:
        return Decision_Tree_Guard(hp)
    else:
        raise TypeError ("Guard not implemented")
