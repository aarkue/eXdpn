"""
.. include:: ./../../docs/_templates/md/petri_net/petri_net.md

"""

import pm4py
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.log.obj import EventLog
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from typing import Tuple


def get_petri_net (log: EventLog, miner_type: str = "AM") -> Tuple[PetriNet, Marking, Marking]:
    """Mines Petri Net based on given event log and returns found Petri Net.

    Args:
        log (EventLog): The given event log to mine the Petri net with.
        miner_type (str): Specifies the type of mining algorithm. Either inductive miner ("IM") or alpha miner ("AM", default).
    
    Returns:
        * net (PetriNet): Petri Net based on input data, later used to find decision find decision points 
        * initial_marking (Marking): Initial Marking
        * final_marking (Marking): Final Marking 

    Raises:
        TypeError: If `miner_type` neither equal to "AM" nor "IM".

    Examples:
        
        >>> from exdpn.util import import_log
        >>> from exdpn import petri_net
        >>> event_log = import_log('./datasets/p2p_base.xes')
        >>> net, im, fm = petri_net.get_petri_net(event_log)

        .. include:: ../../docs/_templates/md/example-end.md

    """
    if miner_type == "AM":
        # use alpha miner 
        net, initial_marking, final_marking = pm4py.discover_petri_net_alpha(log)
    elif miner_type == "IM":    
        # mine petri net using inductive miner to fit all traces
        net, initial_marking, final_marking = inductive_miner.apply(log, variant=inductive_miner.Variants.IM)
    else:
        raise TypeError("Invalid mining type, use IM for inductive miner or AM for alpha miner ")

    return net, initial_marking, final_marking


# tests implemented examples
if __name__ == "__main__":
    import doctest
    doctest.testmod()
#run python .\exdpn\petri_net\get_petri_net.py from eXdpn file