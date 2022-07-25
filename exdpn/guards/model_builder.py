from exdpn.guards import Decision_Tree_Guard, Logistic_Regression_Guard, SVM_Guard, Neural_Network_Guard, XGBoost_Guard
from exdpn.guards import Guard
from typing import Dict, Any 
from exdpn.guards import ML_Technique

    
def model_builder(model_type: ML_Technique, hp: Dict[str, Any]) -> Guard:
    """Internal function to build a specific guard with the provided hyperparameters.
    
    Args:
        model_type (ML_Technique): Specification of machine learning technique (see `exdpn.guards.ml_technique.ML_Technique`).
        hp (Dict[str, any]): Hyperparameters for the machine learning model.
    
    Returns:
        Guard: Machine learning guard of desired type with provided hyperparameters.

    Raises:
        TypeError: If entered model type is not supported.
    
    Examples:
        
        >>> from exdpn.guards.model_builder import model_builder
        >>> from exdpn.guards import ML_Technique
        >>> decision_tree_guard = model_builder(ML_Technique.DT, {'min_samples_split': 0.1, 
        ...                                                       'min_samples_leaf': 0.1, 
        ...                                                       'ccp_alpha': 0.2})
        >>> logistic_regression_guard = model_builder(ML_Technique.LR, {"C": 0.5})

        .. include:: ../../docs/_templates/md/example-end.md

    """
    if model_type == ML_Technique.SVM:
        return SVM_Guard(hp)
    elif model_type == ML_Technique.NN:
        return Neural_Network_Guard(hp)
    elif model_type == ML_Technique.LR:
        return Logistic_Regression_Guard(hp)
    elif model_type == ML_Technique.DT:
        return Decision_Tree_Guard(hp)
    elif model_type == ML_Technique.XGB:
        return XGBoost_Guard(hp)
    else:
        raise TypeError ("Guard not implemented")


# tests implemented examples
if __name__ == "__main__":
    import doctest
    doctest.testmod()
# run python .\exdpn\guards\model_builder.py from eXdpn file 