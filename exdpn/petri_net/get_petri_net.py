"""
.. include:: ./../../docs/_templates/md/petri_net/petri_net.md

"""

import pm4py
from pm4py.objects.petri_net.obj import PetriNet
from pm4py.objects.log.obj import EventLog
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from typing import Tuple


def get_petri_net (log: EventLog, miner_type: str = "AM") -> Tuple[PetriNet, PetriNet.Place, PetriNet.Place]:
    """ Mines Petri Net based on given event log and returns found Petri Net.

    Args:
        log (EventLog): Given event log, as EventLog
        miner_type (str): Spezifies type of mining algorithm, either inductive miner ("IM") or alpha miner ("AM", default)
    
    Returns:
        net (PetriNet): Petri Net based on input data, later used to find decision find decision points 
        initial_marking (PetriNet.Place): Initial Marking
        final_marking (PetriNet.Place): Final Marking 

    Raises:
        TypeError: If an miner_type is any other than "AM" or "IM".

    Examples:
        ```python
        >>> import os 
        >>> from exdpn.util import import_log
        >>> from exdpn import petri_net
        >>> event_log = import_log(os.path.join(os.getcwd(), 'datasets', 'p2p_base.xes'))
        >>> #event_log = import_log('p2p_base.xes')
        >>> net, im, fm = petri_net.get_petri_net(event_log)

        ```
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