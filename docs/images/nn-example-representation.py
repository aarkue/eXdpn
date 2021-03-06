import os 
from exdpn.util import import_log
from exdpn.petri_net import get_petri_net
from exdpn.guard_datasets import extract_all_datasets
from exdpn.guards import Neural_Network_Guard
from exdpn.data_preprocessing import data_preprocessing_evaluation
event_log = import_log(os.path.join(os.getcwd(), 'datasets', 'p2p_base.xes'))
pn, im, fm = get_petri_net(event_log)
dp_dataset_map = extract_all_datasets(event_log, pn, im, fm,
                                      event_level_attributes = ['item_category','item_id','item_amount','supplier','total_price'], 
                                      activityName_key = "concept:name")
# select a certain decision point and the corresponding data set 
dp = list(dp_dataset_map.keys())[1]
dp_dataset = dp_dataset_map[dp]
X_train, X_test, y_train, y_test = data_preprocessing_evaluation(dp_dataset)
guard = Neural_Network_Guard()
guard.train(X_train, y_train)
y_prediction = guard.predict(X_test)
# Sample from test data, as explainable representation of NN is computationally expensive
sampled_test_data = X_test.sample(n=min(100, len(X_test)))
guard.get_explainable_representation(sampled_test_data)

import matplotlib.pyplot as plt
plt.savefig(os.path.join(os.getcwd(), "docs", "images", "nn-example-representation.svg"), bbox_inches = "tight") 