from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.obj import EventLog
import pm4py.util.xes_constants as xes


def import_log(path: str, verbose: bool = False) -> EventLog:
    """Imports an event log from a given path

    Args:
        path (str): The path to the event log
        verbose (bool, optional): If verbose, a progress bar is shown in the console. Defaults to False.

    Returns:
        EventLog: The event log object from importing the log
    """
    variant = xes_importer.Variants.ITERPARSE
    parameters = {variant.value.Parameters.SHOW_PROGRESS_BAR: verbose}
    
    return xes_importer.apply(path, variant=variant, parameters=parameters)

def extend_event_log_with_total_elapsed_time(log: EventLog, total_elapsed_time_attribute_name: str = "eXdpn::total_elapsed_time", timestamp_attribute_name: str = xes.DEFAULT_TIMESTAMP_KEY) -> None:
    """Extends each event in an event log with an attribute corresponding to the total time elapsed (in seconds) since the start \
        of the corresponding case.
    
    Args:
        log (EventLog): The event log to be extended
        total_elapsed_time_attribute_name (str, optional): Event level attribute name to be used. Default is "eXdpn::total_elapsed_time"
        timestamp_attribute_name (str, optional): Timestamp attribute name present in the event log. Default is xes.DEFAULT_TIMESTAMP_KEY ("time:timestamp")
    """
    assert timestamp_attribute_name in list(log[0][0].keys()), \
        f"Error: attribute '{timestamp_attribute_name}' needs to be present in the event log"

    for case in log:
        start_timestamp = case[0][timestamp_attribute_name]
        for event in case:            
            event[total_elapsed_time_attribute_name] = (event[timestamp_attribute_name] - start_timestamp).total_seconds()


def extend_event_log_with_preceding_event_delay(log: EventLog, preceding_event_delay_attribute_name: str = "eXdpn::preceding_event_delay", timestamp_attribute_name: str = xes.DEFAULT_TIMESTAMP_KEY) -> None:
    """Extends each event in an event log with an attribute corresponding to the delay (in seconds) between the current event and the preceding event \
        of the corresponding case. Initial events of each case have a delay of 0 seconds.
    
    Args:
        log (EventLog): The event log to be extended
        preceding_event_delay_attribute_name (str, optional): Event level attribute name to be used. Default is "eXdpn::preceding_event_delay"
        timestamp_attribute_name (str, optional): Timestamp attribute name present in the event log. Default is xes.DEFAULT_TIMESTAMP_KEY ("time:timestamp")
    """
    assert timestamp_attribute_name in list(log[0][0].keys()), \
        f"Error: attribute '{timestamp_attribute_name}' needs to be present in the event log"

    for case in log:
        preceding_timestamp = case[0][timestamp_attribute_name]
        for event in case:            
            event[preceding_event_delay_attribute_name] = (event[timestamp_attribute_name] - preceding_timestamp).total_seconds()
            preceding_timestamp = event[timestamp_attribute_name]

