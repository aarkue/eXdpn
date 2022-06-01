import enum
from exdpn.guards import Decision_Tree_Guard, Neural_Network_Guard

class ML_Technique(enum.Enum):
    NN  = Neural_Network_Guard
    DT  = Decision_Tree_Guard
    LG  = None
    SVM = None