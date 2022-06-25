import os 
from exdpn.util import import_log
from exdpn.petri_net import get_petri_net
from exdpn.guard_datasets import extract_all_datasets
from exdpn import guards
event_log = import_log(os.path.join(os.getcwd(), 'datasets', 'p2p_base.xes'))
pn, im, fm = get_petri_net(event_log)
dp_dataset_map = extract_all_datasets(event_log, pn, im, fm,
                                      case_level_attributes =["concept:name"], 
                                      event_level_attributes = ['item_category','item_id','item_amount','supplier','total_price'], 
                                      activityName_key = "concept:name")
# select a certrain decision point and the corresponding data set 
dp = list(dp_dataset_map.keys())[0]
dp_dataset = dp_dataset_map[dp]
# create a guard manager for that decision point
guard_manager = guards.Guard_Manager(dataframe = dp_dataset)
guard_manager_results = guard_manager.train_test()
guard_manager.get_comparison_plot()

import matplotlib.pyplot as plt
plt.savefig(os.path.join(os.getcwd(), "docs", "images", "comparision-plot.svg"), bbox_inches = "tight") 