from pm4py.objects.log.importer.xes import importer as xes_importer

def import_xes(filepath : str):
    return xes_importer.apply(filepath)