import enum
from exdpn.guards import Decision_Tree_Guard, Logistic_Regression_Guard, SVM_Guard, Neural_Network_Guard

# Enum to provide all implemeted machine learning guard classes 
class ML_Technique(enum.Enum):
    NN  = Neural_Network_Guard
    DT  = Decision_Tree_Guard
    LR  = Logistic_Regression_Guard
    SVM = SVM_Guard 

    def __str__(self) -> str:
        if self == ML_Technique.DT:
            return "Decision Tree"
        elif self == ML_Technique.SVM:
            return "Support Vector Machine"
        elif self == ML_Technique.LR:
            return "Logistic Regression"
        elif self == ML_Technique.NN:
            return "Neural Network"
        else:
            return "Unknown"