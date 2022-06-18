This module provides the core functionality of the package and contains everything needed to set up the data petri net.

--- 
# Data Petri Net #
A data petri net is a petri net with additional information added to the contained decision points. A decision points describes a place followed by more than one transition. To get some insight in which transition a token chooses guards are used to model the behavior. This guards can be understood as classification models based on the data contained in the event log and they predict which transition is most likely to follow given a certain trace and the corresponding data.

---