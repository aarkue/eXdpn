import enum

from exdpn.guards import Decision_Tree_Guard
from exdpn.guards import Logistic_Regression_Guard
from exdpn.guards import SVM_Guard
from exdpn.guards import Neural_Network_Guard


class ML_Technique(enum.Enum):
    """This enum acts as a selector for guard classes which correspond to different machine learning classifiers.

    Examples:
        Creating a data Petri net (see `exdpn.data_petri_net.data_petri_net.Data_Petri_Net`) \
            with only decision tree-based guards can be done using the ml_list parameter a list containing only `DT`:
        
        >>> from exdpn.util import import_log
        >>> from exdpn.data_petri_net import data_petri_net
        >>> from exdpn.guards import ML_Technique
        >>> event_log = import_log('./datasets/p2p_base.xes')
        >>> dpn = data_petri_net.Data_Petri_Net(event_log = event_log,
        ...                                     event_level_attributes = ['item_category','item_id','item_amount','supplier','total_price'],
        ...                                     ml_list = [ML_Technique.DT],
        ...                                     verbose = False)

        .. include:: ../../docs/_templates/md/example-end.md

    """
    NN = Neural_Network_Guard
    """Corresponds to the guard implementation using a neural network classifier."""
    DT = Decision_Tree_Guard
    """Corresponds to the guard implementation using a decision tree classifier."""
    LR = Logistic_Regression_Guard
    """Corresponds to the guard implementation using a logistic regression classifier."""
    SVM = SVM_Guard
    """Corresponds to the guard implementation using a support vector machine classifier."""

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


# tests implemented examples
if __name__ == "__main__":
    import doctest
    doctest.testmod()
# run python .\exdpn\guards\ml_technique.py from eXdpn file
