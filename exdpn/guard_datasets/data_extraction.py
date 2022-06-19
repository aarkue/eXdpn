"""
.. include:: ./../../docs/_templates/md/guard_datasets/guard_datasets.md

"""

from exdpn.decisionpoints import find_decision_points

from pandas import DataFrame
from typing import Dict, List, Tuple, Union, Any
import numpy as np

from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.log.obj import EventLog
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.util import xes_constants as xes


def extract_all_datasets(
    log: EventLog,
    net: PetriNet,
    initial_marking: Marking = None,
    final_marking: Marking = None,
    case_level_attributes: List[str] = [],
    event_level_attributes: List[str] = [],
    tail_length: int = 3,
    activityName_key: str = xes.DEFAULT_NAME_KEY,
    places: List[PetriNet.Place] = None,
    padding: Any = "#"
) -> Dict[PetriNet.Place, DataFrame]:
    """Extracts a dataset for each decision point using token-based replay. For each instance of this decision found in the log, the following data is extracted:
    1. The specified case-level attributes of the case
    2. The specified event-level attributes of the last event of the case before this decision is made
    3. The acitivities executed in the events contained in the `tail_length` events before the decision
    4. The transition which is chosen (the *target* class)

    Args:
        log (EventLog): The event log to extract the data from.
        net (PetriNet, optional): The Petri net on which the token-based replay will be performed (and on which the decision points are identified, \
        not optional if places are not provided).
        initial_marking (Marking, optional): The initial marking of the Petri net.
        final_marking (Marking, optional): The final marking of the Petri net.
        case_level_attributes (List[str], optional): The list of attributes to be extracted on a case-level. Defaults to empty list.
        event_level_attributes (List[str], optional): The list of attributes to be extracted on an event-level. Defaults to empty list.
        tail_length (int, optional): The number of preceding events to record. Defaults to 3.
        activityName_key (str, optional): The key of the activity name in the event log. Defaults to `pm4py.util.xes_constants.DEFAULT_NAME_KEY` ("concept:name").
        places (List[Place], optional): The list of places to extract datasets for. If not present, all decision points are regarded.
        padding (Any, optional): The padding to be used when the tail goes over beginning of the case. Defaults to "#".

    Returns:
        Dict[Place, DataFrame]: The dictionary mapping places in the Petri net to their corresponding dataset.

    Examples:
        ```python
        >>> import os 
        >>> from exdpn.util import import_log
        >>> from exdpn.util import extend_event_log_with_preceding_event_delay
        >>> from exdpn.petri_net import get_petri_net
        >>> from exdpn.guard_datasets import extract_all_datasets
        >>> #event_log = import_log('p2p_base.xes')
        >>> event_log = import_log(os.path.join(os.getcwd(), 'datasets', 'p2p_base.xes'))
        >>> extend_event_log_with_preceding_event_delay(event_log, 'delay')
        >>> pn, im, fm = get_petri_net(event_log)
        >>> dp_dataset_map = extract_all_datasets(event_log, pn, im, fm, 
        ...                                       event_level_attributes = ['delay'])
        
        ```

        ```python
        >>> import os 
        >>> from exdpn.util import import_log
        >>> from exdpn.petri_net import get_petri_net
        >>> from exdpn.guard_datasets import extract_all_datasets
        >>> #event_log = import_log('p2p_base.xes')
        >>> event_log = import_log(os.path.join(os.getcwd(), 'datasets', 'p2p_base.xes'))
        >>> pn, im, fm = get_petri_net(event_log)
        >>> dp_dataset_map = extract_all_datasets(event_log, pn, im, fm,
        ...                                       case_level_attributes =["concept:name"], 
        ...                                       event_level_attributes = ['item_category','item_id','item_amount','supplier','total_price'], 
        ...                                       activityName_key = "concept:name")
        
        ```
    
    """ 

    # Get list of places and mapping which transitions they correspond to
    if places is None:
        target_transitions = find_decision_points(net)
        places = list(target_transitions.keys())
    else:
        target_transitions = {
            place: set(arc.target for arc in net.arcs if arc.source == place) for place in places
        }

    # Compute Token-Based Replay
    replay = _compute_replay(log, net, initial_marking, final_marking, activityName_key, False)
    ## Extract a dataset for each place ##
    datasets = dict()
    for place in places:
        datasets[place] = extract_dataset_for_place(place, target_transitions, log, replay, case_level_attributes, event_level_attributes, tail_length, activityName_key, padding)
    return datasets


def _compute_replay(log:EventLog, net:PetriNet, initial_marking:Marking, final_marking:Marking, activityName_key:str = xes.DEFAULT_NAME_KEY, show_progress_bar:bool = False) -> Dict[str, Any]:
    """Wrapper for PM4Py's token-based replay function.

    Args:
        log (EventLog): The event log to use for Replay.
        net (PetriNet): The Petri net to replay on.
        initial_marking (Marking): The initial Marking of the Petri net.
        final_marking (Marking): The final Marking of the Petri net.
        activityName_key (str, optional): The key of the activity name in the event log. Defaults to `pm4py.util.xes_constants.DEFAULT_NAME_KEY` ("concept:name").
        show_progress_bar (bool, optional): Whether or not to show a progress bar. Defaults to False.

    Returns:
        The token-based replay results.
    
    """    
    variant = token_replay.Variants.TOKEN_REPLAY
    replay_params = {
        variant.value.Parameters.SHOW_PROGRESS_BAR: show_progress_bar,
        variant.value.Parameters.ACTIVITY_KEY: activityName_key,
    }
    return token_replay.apply(log, net, initial_marking, final_marking, variant=variant, parameters=replay_params)

def extract_dataset_for_place(
    place: PetriNet.Place,
    target_transitions:Dict[PetriNet.Place, PetriNet.Transition],
    log: EventLog, 
    replay:Union[List[Dict[str,Any]], Tuple[PetriNet, Marking, Marking]],
    case_level_attributes: List[str] = [],
    event_level_attributes: List[str] = [],
    tail_length: int = 3,
    activityName_key:str = xes.DEFAULT_NAME_KEY,
    padding:Any="#"
) -> DataFrame:
    """Extracts the dataset for a single place using token-based replay. For each instance of this decision found in the log, the following data is extracted:
    1. The specified case-level attributes of the case
    2. The specified event-level attributes of the last event of the case before this decision is made
    3. The acitivities executed in the events contained in the ```tail_length``` events before the decision
    4. The transition which is chosen (the *target* class)


    Args:
        place (PetriNet.Place): The place for which to extract the data.
        target_transitions (Dict[PetriNet.Place, PetriNet.Transition]): The transitions which have an input arc from this place.
        log (EventLog): The Event Log from which to extract the data.
        replay (List[Dict[str, Any]] | Tuple[PetriNet, Marking, Marking]): Either the token-based replay computed by PM4Py, or the net which to use to compute the replay.
        case_level_attributes (List[str], optional): The list of attributes to be extracted on a case-level. Defaults to empty list.
        event_level_attributes (List[str], optional): The list of attributes to be extracted on an event-level. Defaults to empty list.
        tail_length (int, optional): The number of preceding events to record. Defaults to 3.
        activityName_key (str, optional): The key of the activity name in the event log. Defaults to `pm4py.util.xes_constants.DEFAULT_NAME_KEY` ("concept:name").
        padding (Any, optional): The padding to be used when the tail goes over beginning of the case. Defaults to "#".
    
    Returns:
        DataFrame: The guard-dataset extracted for the decision point at `place`.

    Raises:
        Exception: If the default case ID key defined by the XES standard ("concept:name") is not among the case-level attributes.
    
    """    

    # Compute replay if necessary
    if type(replay) is tuple:
        net, im, fm = replay
        replay = _compute_replay(log, net, im, fm, activityName_key, False)


    # Extract the data for the place
    instances = []
    indices = []
    for idx, trace_replay in enumerate(replay):
        # Track how often this decision is made in the trace, for unique Dataframe index
        decision_repetition = 0
        if not trace_replay["trace_is_fit"]:
            # Skip non-fitting traces
            continue
        # Track index of current event because invisible transitions can be present
        event_index = 0
        for transition in trace_replay["activated_transitions"]:

            if transition in target_transitions[place]:
                # Extract Case-Level Attributes
                case = log[idx]
                case_attr_values = [case.attributes.get(attr, np.NaN) for attr in case_level_attributes]

                if event_index <= 0:
                    # There is no "previous event", so we cannot collect this info
                    event_attr_values = [np.NaN] * len(event_level_attributes)
                else:
                    # Get the values of the event level attribute
                    last_event = case[event_index-1]
                    event_attr_values = [last_event.get(attr, np.NaN) for attr in event_level_attributes]
                

                # Finally, extract recent activities
                tail_activities = []
                for i in range(1,tail_length+1):
                    if event_index-i >= 0:
                        tail_activities.append(case[event_index-i].get(activityName_key, ""))
                    else:
                        tail_activities.append(padding)

                # This instance record  now descibes the decision situation
                instance = case_attr_values + event_attr_values + tail_activities + [transition]
                instances.append(instance)
                # Give this index a unique index
                if xes.DEFAULT_TRACEID_KEY not in case.attributes:
                    raise Exception(f"A case in the Event Log Object has no caseid (No case attribute {xes.DEFAULT_TRACEID_KEY})")
                else:
                    indices.append((case.attributes[xes.DEFAULT_TRACEID_KEY],decision_repetition))
                decision_repetition += 1

                # Dont't count silent transitions
            if transition.label is not None:
                event_index += 1
    from pandas import MultiIndex
    return DataFrame(
        instances,
        columns=["case::" + attr for attr in case_level_attributes] + ["event::"+ attr for attr in event_level_attributes] + [f"tail::prev{i}" for i in range(1,tail_length+1)] +  ["target"],
        index=MultiIndex.from_tuples(indices, names=[xes.DEFAULT_TRACEID_KEY,"decision_repetiton"])
    )

# tests implemented examples
if __name__ == "__main__":
    import doctest
    doctest.testmod()
# run python .\exdpn\guard_datasets\data_extraction.py from eXdpn file 