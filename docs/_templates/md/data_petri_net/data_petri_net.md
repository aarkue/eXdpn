This module provides the core functionality of the package and contains everything needed to set up the data Petri net.

--- 

# Data Petri Nets #
A data Petri net is a Petri net with additional information added to the contained decision points (see `exdpn.decisionpoints`). Guards are used to model the behavior to get some insight into which transition a token chooses (see `exdpn.guards`). These guards can be understood as classification models based on the data contained in the event log. They predict which transition is most likely to follow given a certain trace and the corresponding event data.

---