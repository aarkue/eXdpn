from exdpn.decisionpoints import find_decision_points

from pandas import DataFrame
from typing import Dict, List, Any
import numpy as np

from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.log.obj import EventLog
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.util import xes_constants as xes


def extract_all_datasets(
    log: EventLog,
    net: PetriNet,
    initial_marking: Marking,
    final_marking: Marking,
    case_level_attributes: List[str] = None,
    event_level_attributes: List[str] = None,
    tail_length: int = 3,
    activityName_key:str = xes.DEFAULT_NAME_KEY,
    places: List[PetriNet.Place] = None,
    padding: Any = "#"
) -> Dict[PetriNet.Place, DataFrame]:
    """Extracts a dataset for each Decision Point using Token-Based Replay. For each instance of this decision found in the log, the following data is extracted:
    1. The specified Case-Level attributes of the case
    2. The specified Event-Level attributes of the last event of the case before this decision is made
    3. The acitivities executed in the events contained in the ```tail_length``` events before the decision
    4. The transition which is chosen (the *target* class)

    Args:
        log (EventLog): The event Log to extract the data from
        net (PetriNet): The petri net on which the Replay will be performed (and on which the decision points are identified)
        initial_marking (Marking): Initial Marking of the Petri Net
        final_marking (Marking): Final Marking of the Petri Net
        case_level_attributes (List[str], optional): List of attributes to be extracted on a Case level. If none specified, all are used.
        event_level_attributes (List[str], optional): List of attributes to be extracted on an Event level. If none specified, all are used.
        tail_length (int, optional): Number of events lookback to extract executed activity. Defaults to 3.
        activityName_key (str, optional): Key of the activity name in the event log. Defaults to pm4py.util.xes_constants.DEFAULT_NAME_KEY ("concept:name").
        places (List[Place], optional): List of places to extract the Datasets for. If not present, all decision points are regarded.
        padding (Any, optional): Padding to be used when tail goes over beginning of case. Defaults to "#".

    Returns:
        Dict[Place, DataFrame]: Dictionary mapping each Place to its corresponding dataset.
    """ 

    # Get list of places and mapping which transitions they correspond to
    if places is None:
        target_transitions = find_decision_points(net)
        places = list(target_transitions.keys())
    else:
        target_transitions = {
            place: set(arc.target for arc in net.arcs if arc.source == place) for place in places
        }


    # Get attributes if necessary (Were not specified)
    if case_level_attributes is None or event_level_attributes is None:
        case_attrs = set()
        event_attrs = set()
        for case in log:
            for event in case:
                case_attrs.update(case.attributes.keys())
                event_attrs.update(event.keys())
        if case_level_attributes is None:
            case_level_attributes = case_attrs
        if event_level_attributes is None:
            event_level_attributes = event_attrs

    # Compute Token-Based Replay using PM4PY
    # Parameters for the token-based replay; Supress progress bar and use supplied activity name key
    variant = token_replay.Variants.TOKEN_REPLAY
    replay_params = {
        variant.value.Parameters.SHOW_PROGRESS_BAR: False,
        variant.value.Parameters.ACTIVITY_KEY: activityName_key,
    }
    replay = token_replay.apply(log, net, initial_marking, final_marking, variant=variant, parameters=replay_params)


    ## Extract a dataset for each place ##
    datasets = dict()
    # TODO: Iterate over replay only once, handle all places simultaneously?? However it seems very fast the way it is already.
    for place in places:
        instances = []
        for idx, trace_replay in enumerate(replay):
            for transition_index, transition in enumerate(trace_replay["activated_transitions"]):
                if transition in target_transitions[place]:

                    # Extract Case-Level Attributes
                    case = log[idx]
                    case_attr_values = [case.attributes.get(attr, np.NaN) for attr in case_level_attributes]

                    if transition_index < 0:
                        # There is no "previous event", so we cannot collect this info
                        event_attr_values = [np.NaN] * len(event_level_attributes)
                    else:
                        # Get the values of the event level attribute
                        last_event = case[transition_index-1]
                        event_attr_values = [last_event.get(attr, np.NaN) for attr in event_level_attributes]
                    

                    # Finally, extract recent activities
                    tail_activities = []
                    for i in range(1,tail_length+1):
                        if transition_index-i >= 0:
                            tail_activities.append(case[transition_index-i].get(activityName_key, ""))
                        else:
                            tail_activities.append(padding)

                    # This instance record  now descibes the decision situation
                    instance = case_attr_values + event_attr_values + tail_activities + [transition]
                    instances.append(instance)
        df = DataFrame(instances, columns=["case::" + attr for attr in case_level_attributes] + ["event::"+ attr for attr in event_level_attributes] + [f"tail::prev{i}" for i in range(1,tail_length+1)] +  ["target"])
        datasets[place] = df
    return datasets
        
