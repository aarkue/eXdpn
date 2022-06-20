"""
.. include:: ./../../docs/_templates/md/util/util.md

"""

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.obj import EventLog
import pm4py.util.xes_constants as xes
from pm4py.statistics.attributes.log.get import get_all_event_attributes_from_log


def import_log(path: str, verbose: bool = False) -> EventLog:
    """Imports an XES event log from a given path.

    Args:
        path (str): The path to the XES event log file.
        verbose (bool, optional): If verbose, a progress bar is shown in the console. Defaults to False.

    Returns:
        EventLog: The event log object.

    Note:
        Please make sure that the event log follows the XES standard.

    Examples:
        ```python
        >>> import os
        >>> from exdpn.util import import_log
        >>> event_log = import_log(os.path.join(os.getcwd(), 'datasets', 'p2p_base.xes'))

        ```
        
    """
    variant = xes_importer.Variants.ITERPARSE
    parameters = {variant.value.Parameters.SHOW_PROGRESS_BAR: verbose}
    
    return xes_importer.apply(path, variant=variant, parameters=parameters)

def extend_event_log_with_total_elapsed_time(log: EventLog, total_elapsed_time_attribute_name: str = "eXdpn::total_elapsed_time", timestamp_attribute_name: str = xes.DEFAULT_TIMESTAMP_KEY) -> None:
    """Extends each event in an event log with an attribute corresponding to the total time elapsed (in seconds) since the start \
        of the corresponding case.
    
    Args:
        log (EventLog): The event log to be extended.
        total_elapsed_time_attribute_name (str, optional): The event level attribute name to be used. Default is "eXdpn::total_elapsed_time".
        timestamp_attribute_name (str, optional): The timestamp attribute name present in the event log. Default is `pm4py.util.xes_constants.DEFAULT_NAME_KEY` ("time:timestamp").
    
    Raises:
        KeyError: If the attribute with name `timestamp_attribute_name` is not present in the event log.

    Examples:
        ```python
        >>> import os 
        >>> from exdpn.util import import_log
        >>> from exdpn.util import extend_event_log_with_total_elapsed_time
        >>> from exdpn.data_petri_net import Data_Petri_Net
        >>> event_log = import_log(os.path.join(os.getcwd(), 'datasets', 'p2p_base.xes'))
        >>> extend_event_log_with_total_elapsed_time(event_log, 'elapsed_time')
        >>> dpn = Data_Petri_Net(event_log = event_log,
        ...                      event_level_attributes = ['elapsed_time'],
        ...                      verbose = False)
        
        ```

    """
    if timestamp_attribute_name not in get_all_event_attributes_from_log(log):
        raise KeyError(f"Attribute with name '{timestamp_attribute_name}' is not present in the event log.")

    for case in log:
        start_timestamp = case[0][timestamp_attribute_name]
        for event in case:            
            event[total_elapsed_time_attribute_name] = (event[timestamp_attribute_name] - start_timestamp).total_seconds()


def extend_event_log_with_preceding_event_delay(log: EventLog, preceding_event_delay_attribute_name: str = "eXdpn::preceding_event_delay", timestamp_attribute_name: str = xes.DEFAULT_TIMESTAMP_KEY) -> None:
    """Extends each event in an event log with an attribute corresponding to the delay (in seconds) between the current event and the preceding event \
        of the corresponding case. Initial events of each case have a delay of 0 seconds.
    
    Args:
        log (EventLog): The event log to be extended.
        preceding_event_delay_attribute_name (str, optional): The event level attribute name to be used. Default is "eXdpn::preceding_event_delay".
        timestamp_attribute_name (str, optional): The timestamp attribute name present in the event log. Default is `pm4py.util.xes_constants.DEFAULT_NAME_KEY` ("time:timestamp").
    
    Raises:
        KeyError: If the attribute with name `timestamp_attribute_name` is not present in the event log.

    Examples:
        ```python
        >>> import os
        >>> from exdpn.util import import_log
        >>> from exdpn.util import extend_event_log_with_preceding_event_delay
        >>> from exdpn.data_petri_net import Data_Petri_Net
        >>> event_log = import_log(os.path.join(os.getcwd(), 'datasets', 'p2p_base.xes'))        
        >>> extend_event_log_with_preceding_event_delay(event_log, 'delay')
        >>> dpn = Data_Petri_Net(event_log = event_log,
        ...                      event_level_attributes = ['delay'],
        ...                      verbose = False)

        ```

    """
    if timestamp_attribute_name not in get_all_event_attributes_from_log(log):
        raise KeyError(f"Attribute with name '{timestamp_attribute_name}' is not present in the event log.")

    for case in log:
        preceding_timestamp = case[0][timestamp_attribute_name]
        for event in case:            
            event[preceding_event_delay_attribute_name] = (event[timestamp_attribute_name] - preceding_timestamp).total_seconds()
            preceding_timestamp = event[timestamp_attribute_name]


# tests implemented examples
if __name__ == "__main__":
    import doctest
    doctest.testmod()
# run python .\exdpn\util\util.py from eXdpn file 