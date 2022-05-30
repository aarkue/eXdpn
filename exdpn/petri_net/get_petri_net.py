import pm4py
from pm4py.objects.petri_net.obj import PetriNet
from pm4py.objects.log.obj import EventLog
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from typing import Tuple

def get_petri_net (log: EventLog) -> Tuple[PetriNet, PetriNet.Place, PetriNet.Place]:
    """ Mines Petri Net based on given event log and returns found Petri Net.

    Args: 
        log: Given event log, as EventLog
    
    Returns: 
        PetriNet: Petri Net based on input data, later used to find decision find decision points 
        PetriNet.Place: Initial Marking
        PetriNet.Place: Final Marking 
    """
    
    # mine petri net using inductive miner to fit all traces
    net, initial_marking, final_marking = inductive_miner.apply(log)
    
    return net, initial_marking, final_marking
