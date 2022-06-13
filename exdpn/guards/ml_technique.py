import enum
from exdpn.guards import Decision_Tree_Guard, Logistic_Regression_Guard, SVM_Guard, Neural_Network_Guard

# class to provide all implemeted machine learning guard classes 
class ML_Technique(enum.Enum):
    NN  = Neural_Network_Guard
    DT  = Decision_Tree_Guard
    LR  = Logistic_Regression_Guard
    SVM = SVM_Guard 