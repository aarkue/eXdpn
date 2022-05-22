from exdpn.decisionpoints import find_decision_points

from pandas import DataFrame
from typing import Dict
import numpy as np

from pm4py.objects.petri_net.obj import PetriNet
from pm4py.objects.log.obj import EventLog
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay

# in the following, map or mapping will often refer to python's dictionary data structure
# -----------------------------------------------------------------------------------------------------------------------------


def get_instances_per_transition(log: EventLog, net: PetriNet, im: PetriNet.Place, fm: PetriNet.Place,
                                 case_level_attributes: list[str],
                                 event_attributes: list[str],
                                 sliding_window_size: int,
                                 act_name_attr: str) -> Dict[PetriNet.Transition, list[any]]:
    """ Returns a dictionary of transitions (activities) mapped to the attribute lists of all instances that passed that activity.
    Args:
        log (EventLog): The pm4py event log to use for dataset generation
        net (PetriNet): The pm4py Petri net in which to generate the datasets
        im (PetriNet.Place): The initial marking of net
        fm (PetriNet.Place): The final marking of net
        case_level_attributes (list[str]): Attribute list on the level of cases to be considered for each instance in the datasets
        event_attributes (list[str]): Attribute list on the level of events to be considered for each instance in the datasets
        sliding_window_size (int): Size of the sliding window recording the last sliding_window_size events
        act_name_attr (str): Event level attribute name corresponding to the name of an event
    Returns:
        transition_instance_map (Dict[PetriNet.Transition, list[any]]): Mapping of transitions (t) to instance lists \
            corresponding to trace attributes which passed transition t. List of event attributes has to be list[any] since one \
            can not assume the type of attributes in an event log

    """
    transition_instance_map = dict()
    replay = token_replay.apply(log, net, im, fm)
    # Token based replay is enough. All traces will be fitting, since we replay on the same log used for mining the petri net 
    # (mining is based on inductive miner, see get_petri_net function)

    # instance scheme:
    # [case_level_attr_1, .., case_level_attr_i, event_attr_1, .., event_attr_j, window_last_1, .., window_last_(sliding_window_size)]
    for index_trace, trace_replay in enumerate(replay):
        # initialize count for passed activities
        index_activity = 0
        window = [np.nan for _ in range(sliding_window_size)] 

        for transition in trace_replay["activated_transitions"]:
            # add transitions to dict if they are not already in it
            if not transition in transition_instance_map:
                transition_instance_map[transition] = []

            # get all case level attributes of the current instance
            instance = [log[index_trace].attributes[attr] for attr in case_level_attributes]
            # create trace of all case level attributes, event attributes, and the last events per instance
            if index_activity > 0:
                transition_instance_map[transition].append(instance
                                                      + [log[index_trace][index_activity-1][attr]
                                                         if attr in log[index_trace][index_activity-1] else np.nan
                                                         for attr in event_attributes]
                                                      + [window[-(i+1)] for i in range(sliding_window_size)])
            else:
                # special case: current place is source, i.e., no previous events were recorded. in this case no event attributes can be added
                transition_instance_map[transition].append(instance
                                                      + [np.nan for _ in event_attributes]
                                                      + [window[-(i+1)] for i in range(sliding_window_size)])

            # update count for passed activities and add activity to sliding window
            if transition.label != None:
                index_activity += 1
                window.append(log[index_trace]
                              [index_activity-1][act_name_attr])
    
    # output: transition -> [instance]
    return transition_instance_map

# -----------------------------------------------------------------------------------------------------------------------------


def get_instances_per_place_per_transition(log: EventLog, net: PetriNet, im: PetriNet.Place, fm: PetriNet.Place,
                                           case_level_attributes: list[str],
                                           event_attributes: list[str],
                                           sliding_window_size: int,
                                           act_name_attr: str) -> tuple[Dict[PetriNet.Place, Dict[PetriNet.Transition, list[any]]], list[str]]:
    """ Returns a dictionary with all places, that are decision points, as keys and as values all transitions that follow these places corresponding transitions. \
        The transitions are mapped on the particular attribute lists of all instances that passed that activity.
    Args:
        log (EventLog): The pm4py event log to use for dataset generation
        net (PetriNet): The pm4py Petri net in which to generate the datasets
        im (PetriNet.Place): The initial marking of net
        fm (PetriNet.Place): The final marking of net
        case_level_attributes (list[str]): Attribute list on the level of cases to be considered for each instance in the datasets
        event_attributes (list[str]): Attribute list on the level of events to be considered for each instance in the datasets
        sliding_window_size (int): Size of the sliding window recording the last sliding_window_size events
        act_name_attr (str): Event level attribute name corresponding to the name of an event
    Returns:
        place_transition_instance_map (Dict[PetriNet.Place, Dict[PetriNet.Transition, list[any]]]): The mapping of places (p) to mappings of transitions (t) to instance lists \
            corresponding to trace attributes which visited p during replay and proceeded by taking transition t
        attribute_list (list[str]): The attribute names corresponding to the tuple entries of the instances
    """
    decision_points = find_decision_points(net)

    # this is the core data consisting of instance tuples (created using token based replay) per transition in the Petri net
    transition_instance_map = get_instances_per_transition(
        log, net, im, fm, case_level_attributes, event_attributes, sliding_window_size, act_name_attr)

    # output: place -> transition -> [instance]
    # use decision points as keys and map the corresponding transitions and their instance tuples 
    place_transition_instance_map = {decision_point: {transition: transition_instance_map[transition] if transition in transition_instance_map else set()
                                                 for transition in transitions}
                                for decision_point, transitions in decision_points.items()}

    # create string description of attribute names of each entry in an instance tuple
    # used as column names in the guard datasets
    case_level_attributes_annot = [f"case::{cla}" for cla in case_level_attributes]
    event_attributes_annot      = [f"event::{ea}" for ea in event_attributes]
    attribute_list = case_level_attributes_annot + event_attributes_annot + \
        [f"sw::prev-{i+1}" for i in range(sliding_window_size)]

    return place_transition_instance_map, attribute_list

# -----------------------------------------------------------------------------------------------------------------------------


def get_guard_dataset(place: PetriNet.Place, 
                      place_transition_instance_map: Dict[PetriNet.Place, Dict[PetriNet.Transition, list[any]]], 
                      attribute_list: list[str]) -> DataFrame:
    """ Returns a guard dataset for a specific place. This contains all contain all combinations of trace attributes and outgoing transition seen in the data for the given guard.
    Args:
        place (PetriNet.Place): The pm4py Petri net place to use for dataset generation
        place_transition_instance_map (Dict[PetriNet.Place,Dict[PetriNet.Transition, list[any]]]): The mapping of places (p) to mappings of transitions (t) to instance lists \
            corresponding to trace attributes which visited p during replay and proceeded by taking transition t 
        attribute_list (list[str]): The attribute names corresponding to the instance tuple entries
    Returns:
        pd.DataFrame: A dataset corresponding to trace attributes with their outgoing transition (for traces which visited place during replay)
    """
    # get mapping of place to transitions and the names of the occuring transitions
    transition_instance = place_transition_instance_map[place]
    transitions_taken = list(transition_instance.keys())
    
    # set up data frame with attribute names as columns and a target column describing the transition
    df = DataFrame(columns=attribute_list + ["target"])

    # add trace attributes and outgoing transition to data frame
    for transition in transitions_taken:
        for instance in transition_instance[transition]:
            df.loc[len(df.index)] = instance + [transition]

    return df

# -----------------------------------------------------------------------------------------------------------------------------


def get_all_guard_datasets(log: EventLog, net: PetriNet, im: PetriNet.Place, fm: PetriNet.Place,
                           case_level_attributes: list[str] = ["concept:name"],
                           event_attributes: list[str] = [],
                           sliding_window_size: int = 3,
                           act_name_attr: str = "concept:name") -> Dict[PetriNet.Place, DataFrame]:
    """ Returns a mapping of all guards (desicion points) to their corresponding guard dataset. Thes guard dataset contains all combinations of trace attributes and outgoing \
        transition seen in the data for the particular guard, i.e., the attribute values of all instances that passed the corresponding decision point as well as the outgoing transition.
    Args:
        log (EventLog): The pm4py event log to use for dataset generation
        net (PetriNet): The pm4py Petri net in which to generate the datasets
        im (PetriNet.Place): The initial marking of net
        fm (PetriNet.Place): The final marking of net
        case_level_attributes (list[str]): Attribute list on the level of cases to be considered for each instance in the datasets
        event_attributes (list[str]): Attribute list on the level of events to be considered for each instance in the datasets
        sliding_window_size (int): Size of the sliding window recording the last sliding_window_size events
        act_name_attr (str): Event level attribute name corresponding to the name of an event
    Returns:
        Dict[PetriNet.Place, pd.DataFrame]: A dictionary mapping places where decisions are made to the dataset \
            corresponding to trace attributes with their outgoing transition (for traces which visited this place during replay)
    """
    
    # get mapping of all places to their transitions and the corresponding attribute names
    place_transition_instance_map, attribute_list = get_instances_per_place_per_transition(
        log, net, im, fm, case_level_attributes, event_attributes, sliding_window_size, act_name_attr)

    # get trace attributes for all places 
    place_df_map = {place: get_guard_dataset(place, place_transition_instance_map, attribute_list)
                    for place in place_transition_instance_map.keys()}
    
    return place_df_map
