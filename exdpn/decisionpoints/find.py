from pm4py.objects.petri_net.obj import PetriNet
from typing import Set, Dict

def find_decision_points(net: PetriNet) -> Dict[PetriNet.Place, Set[PetriNet.Transition]]:
    """ Finds decision points in a Petri net.
    Args:
        net (PetriNet): The pm4py Petri net in which to find decision points
    Returns:
        Dict[PetriNet.Place, Set[PetriNet.Transition]]: A dictionary mapping places where decisions are made to the transitions which are the choices
    """
    
    # Build mapping of places to their associated target transitions
    targets : Dict[PetriNet.Place,Set[PetriNet.Transition]] = {p: set() for p in net.places}
    for arc in net.arcs:
        if arc.source in targets.keys():
            targets[arc.source].add(arc.target)

    # Keep only the places which have multiple target transitions
    decision_points = {key: target_set for key, target_set in targets.items() if len(target_set) > 1}
    
    return decision_points