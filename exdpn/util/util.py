from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.obj import EventLog


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
