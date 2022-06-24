<!--This file is a copy of ./docs/_templates/md/exdpn.md, adding code block denotation around code examples. Please keep this file up to date-->
exdpn (e**x**plainable **d**ata **P**etri **n**ets) is a tool to mine and evaluate explainable data Petri nets using different classification techniques.

---

## Getting Started ##
Installing exdpn is possible via pip:
```bash
python -m pip install exdpn 
```

Now you can mine your first explainable data Petri net given an event log in XES format:
```python
>>> from exdpn.util import import_log
>>> from exdpn.data_petri_net import Data_Petri_Net
>>> event_log = import_log('<path_to_event_log.xes>')
>>> dpn = Data_Petri_Net(event_log, event_level_attributes=['event_level_attribute'])
```


This will mine a data Petri net for your event log, considering only "event_level_attribute" as a possible attribute for classification. 
The `exdpn.data_petri_net.data_petri_net.Data_Petri_Net` class already takes care of the workflow to create a data Petri net. In cases where fine-grained 
control of the data Petri net creation is needed or only certain functionallity of this package is needed, one can simply call all the needed functions and methods directly. 

Let's say we are only interested in extracting the guard dataset at one specific decision point in the Petri net.
We start off by importing the event log from memory and creating a standard Petri net:

```python
>>> from exdpn.util import import_log
>>> from exdpn.petri_net import get_petri_net
>>> event_log = import_log('<path_to_event_log.xes>')
>>> pn, im, fm = get_petri_net(event_log)
```

We then extract all the decision points and specify our place of interest using the `exdpn.decisionpoints` module:

```python
>>> from exdpn.decisionpoints import find_decision_points
>>> dp_dict = find_decision_points(pn)
>>> decision_point = list(dp_dict.keys())[0]
```

To extract a guard dataset for the specific place `decision_point`, we call the following data extraction function from `exdpn.guard_datasets`:

```python
>>> from exdpn.guard_datasets import extract_all_datasets
>>> dataset = extract_all_datasets(event_log, net, im, fm, event_level_attributes=['event_level_attribute'], places=[decision_point])
```


Further examples can be seen in the API documentation. The sometimes referenced XES file `p2p_base.xes` can be found on Github.

---

## Source Code and UI-application ##
The source code of this package is available on Github ([aarkue/eXdpn](https://github.com/aarkue/eXdpn)).
Furthermore, the Github also includes a graphical user interface in the form of a Python-webserver and a Docker container to easily start the web-UI locally. 

---

## Qualitative Analysis of eXdpn ##
To provide some insights to the eXdpn application, the tool was tested and analyzed using four different syntetic p2p event logs. This allowed us to test whether the different machine learning techniques are able to model the decision-making behavior in the event logs. The analysis can be found on Github ([aarkue/eXdpn](https://github.com/aarkue/eXdpn)).

---