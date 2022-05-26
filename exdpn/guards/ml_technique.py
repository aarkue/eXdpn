import enum
from exdpn.guards import Decision_Tree_Guard

class ML_Technique(enum.Enum):
    NN  = None
    DT  = Decision_Tree_Guard
    LG  = None
    SVM = None