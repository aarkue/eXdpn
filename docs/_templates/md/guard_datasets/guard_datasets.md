This module provides the function to extract a dataset for each decision point in a Petri net given an event log and attributes to record.

---

## Guard-Datasets ##
Each row/instance in a guard-dataset corresponds to a trace visiting the corresponding decision point (see module `exdpn.decisionpoints`) during token-based replay. This could potentially happen several times. For each such visit, the algorithm records the specified case-level attributes, event-level attributes (for the previous event in the case), and a tail of preceding events. These records make up the columns of the guard-dataset. In order to eventually do predictions with the data, the following event (i.e., transition) is recorded in the "target" column.

---