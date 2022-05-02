import pm4py
from pm4py.objects.petri_net.obj import PetriNet
from pm4py.algo.discovery.inductive import factory as inductive_miner

# data type for log? dataframe or event log event? 
def mining (log) -> PetriNet:
    """ Mines Petri Net based on given event log and returns found Petri Net.

    Args: 
        log: Given event log, as # insert data type
    
    Returns: 
        PetriNet: Petri Net based on input data, later used to find decision find decision points 
    """
    

    # mine petri net
    net, initial_marking, final_marking = inductive_miner.apply(log)
    # do we need marking later?
    # return petri net 
