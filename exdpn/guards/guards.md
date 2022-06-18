This module contains the functionality to create different machine learning guards to model the behavior at decision points as well as an guard manager to automatically handle the guard training and selected for each decision point.

---
# Guard Manager #
The guard manager is called for each decision point to get the best possible guard model for either the default machine learning techniques (all implemented) or the selected machine learning techniques.

---
# Guards #
A guard is in this case a machine learning model which aims to model the behavior on a decision point. The goal is to receive a classification model which predicts the transition following a certain decision point using the data contained in the given event log. To get a better understanding on how the model works and provide a better understanding all machine learning techniques return an explainable representation of the fitted model. The implemented machine learning techniques are: 
`exdpn.guards.decision_tree_guard`, `exdpn.guards.logistic_regression_guard`, `exdpn.guards.svm_guard`, and `exdpn.guards.neural_network_guard`.

---

