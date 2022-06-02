import enum
from exdpn.guards import Decision_Tree_Guard
from exdpn.guards.logistic_regression_guard import Logistic_Regression_Guard
from exdpn.guards.svm_guard import SVM_Guard

class ML_Technique(enum.Enum):
    NN  = None
    DT  = Decision_Tree_Guard
    LR  = Logistic_Regression_Guard
    SVM = SVM_Guard 