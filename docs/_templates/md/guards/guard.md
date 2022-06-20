This module contains the functionality to create different machine learning guards to model the behavior at decision points as well as an guard manager to automatically handle the guard training and selection for each decision point.

---
# Guard #
A guard is in this case a machine learning model which aims to model the behavior on a decision point. The goal is to receive a classification model which predicts the transition following a certain decision point using the data contained in the given event log. To get a better understanding on how the model works and provide a better understanding all machine learning techniques return an explainable representation of the fitted model. 

---