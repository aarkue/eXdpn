import io
from typing import Any, Dict, Tuple
from flask import Flask, render_template, request, redirect
import os
from datetime import datetime as dt
from pandas import DataFrame

app = Flask("eXdpn")
from werkzeug.utils import secure_filename
from werkzeug.security import safe_join

import pm4py
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.objects.log.obj import EventLog
from pm4py.objects.petri_net.obj import PetriNet, Marking

import uuid
from exdpn.util import import_log, extend_event_log_with_preceding_event_delay, extend_event_log_with_total_elapsed_time
from exdpn.decisionpoints import find_decision_points
from exdpn.guards import ML_Technique
from exdpn.data_petri_net import Data_Petri_Net
import matplotlib.pyplot as plt
from exdpn.petri_net import get_petri_net


import matplotlib
matplotlib.use("agg")

def get_upload_path(name):
    return safe_join("./uploads/", name)


def get_file_info(name):
    return {
        "name": name.split("@")[0],
        "size": os.path.getsize(get_upload_path(name)),
        "uploaded_at": dt.fromtimestamp(
            os.path.getmtime(get_upload_path(name))
        ).strftime("%Y-%m-%d %H:%M:%S"),
    }


# Initialize uploaded logs from uploads folder
if not os.path.exists("./uploads"):
    os.mkdir("./uploads")
uploaded_logs = {
    file_name: get_file_info(file_name) for file_name in os.listdir("./uploads/")
}

loaded_event_logs: Dict[
    str, Tuple[Dict[str, Any], EventLog]
] = dict()

discovered_models: Dict[str,Tuple[PetriNet, Marking, Marking]] = dict()
data_petri_nets: Dict[str, Data_Petri_Net] = dict()
explainable_representations: Dict[str,Dict[Tuple[int,ML_Technique],str]] = dict() # Logid -> (placeid,ML_Technique) -> explanation

ATTR_IGNORE_LIST = ["concept:name", "time:timestamp"]


# Index page
@app.route("/")
def index():
    return render_template("index.html", uploaded_logs=uploaded_logs)


# Form for uploading an event log
@app.route("/upload-log", methods=["POST"])
def upload_log():
    if request.method == "POST":
        # Get the form input from request
        log = request.files["log"]
        safe_name = secure_filename(log.filename)
        safe_name = safe_name.replace("@", "_") + "@" + str(uuid.uuid4())

        save_path = get_upload_path(safe_name)
        log.save(save_path)
        uploaded_logs[safe_name] = get_file_info(safe_name)
        return redirect("/")


# Details page for a single log
@app.route("/log/<logid>", methods=["GET", "DELETE"])
def log_page(logid: str):
    if request.method == "DELETE":
        if logid in uploaded_logs:
            # Remove all traces of the event log
            uploaded_logs.pop(logid)
            if logid in loaded_event_logs:
                loaded_event_logs.pop(logid)
            if logid in discovered_models:
                discovered_models.pop(logid)

            safe_path = get_upload_path(logid)
            if safe_path is not None:
                os.remove(safe_path)
            else:
                return "Invalid log name provided.", 403
            return "", 200
        else:
            return "Could not find log.", 404
    else:
        if logid in uploaded_logs:
            log = uploaded_logs[logid]
            return render_template("log.html", log=log, log_id=logid)
        else:
            return redirect("/")


@app.route("/log/<logid>/load", methods=["GET"])
def load_log(logid: str):
    if logid in uploaded_logs:
        if logid in loaded_event_logs:
            log_info = loaded_event_logs[logid][0]
        else:
            path = get_upload_path(logid)
            # Check for validity
            if path is not None and os.path.exists(path):
                xes: EventLog = import_log(path, verbose=False)

                events = [evt for case in xes for evt in case]
                event_attributes = {attr for evt in events for attr in evt if attr not in ATTR_IGNORE_LIST}
                case_attributes = {attr for case in xes for attr in case.attributes if attr not in ATTR_IGNORE_LIST}
                activities = {evt["concept:name"] for evt in events}
                log_info = {
                    "xes_stats": {
                        "num_cases": len(xes),
                        "num_events": len(events),
                        "num_activities": len(activities)
                    },
                    "attributes": {
                        "event_attributes": list(event_attributes),
                        "case_attributes": list(case_attributes)
                    }
                }

                loaded_event_logs[logid] = (log_info, xes)
            else:
                return {"message": "Invalid path"}, 400
        return log_info, 200
    else:
        return {"message": "Log not found"}, 400


@app.route("/log/<logid>/discover/<algo_name>", methods=["GET"])
def discover_model(logid: str, algo_name:str):
    if logid not in loaded_event_logs:
        return {"message": "Log not loaded. Please make sure the event log exists."}, 400
    else:
        log = loaded_event_logs[logid][1]
        if algo_name == "inductive_miner":
            net, im, fm = get_petri_net(log, "IM")
        elif algo_name == "alpha_miner":
            net, im, fm = pm4py.discover_petri_net_alpha(log)
        else:
            return {"message": "Invalid algorithm name"}, 400
        discovered_models[logid] = (net,im,fm)
        decision_points = find_decision_points(net)
        place_ids = [str(id(p)) for p in decision_points]
        gviz = pn_visualizer.apply(net, im, fm)
        dot = str(gviz)
        return {"dot": dot, "decision_points": place_ids}, 200

@app.route("/log/<logid>/mine-decisions", methods=["POST"])
def mine_decisions(logid: str):
    if logid not in loaded_event_logs or logid not in discovered_models:
        return {"message": "Log or model not loaded"}, 400
    else:
        body = request.get_json()

        event_log = loaded_event_logs[logid][1]
        event_level_attributes = body['event_attributes']
        case_level_attributes = body['case_attributes']

        synth_attrs = body["synthetic_attributes"]
        if "time_since_last" in synth_attrs:
            extend_event_log_with_preceding_event_delay(event_log,"eXdpn::time_since_last_event")
            event_level_attributes.append("eXdpn::time_since_last_event")
        if "total_elapsed_time" in synth_attrs:
            extend_event_log_with_total_elapsed_time(event_log, "eXdpn::elapsed_case_time")
            event_level_attributes.append("eXdpn::elapsed_case_time")

        ml_techniques = body["ml_techniques"]
        ml_techniques = [technique for technique in ML_Technique if str(technique) in ml_techniques]
        dpn = Data_Petri_Net(
            event_log = event_log,
            petri_net = discovered_models[logid][0],
            initial_marking = discovered_models[logid][1],
            final_marking = discovered_models[logid][2],
            case_level_attributes = case_level_attributes,
            event_level_attributes = event_level_attributes,
            guard_threshold = 0,
            ml_list=ml_techniques,
            hyperparameters = {ML_Technique.NN: {'hidden_layer_sizes': (30, 30)},
                                                                        ML_Technique.DT: {'min_impurity_decrease': 0.001},
                                                                        ML_Technique.LR: {"C": 0.5},
                                                                        ML_Technique.SVM: {"C": 0.5},
                                                                        ML_Technique.XGB: {},
                                                                        ML_Technique.RF: {'n_estimators': 100,
                                                                                          'min_impurity_decrease': 0.001}},
        )
        return_info = dict()
        for p, best_guard in dpn.get_best().items():
            guard_result_svg = ""
            fig = dpn.guard_manager_per_place[p].get_comparison_plot()

            guard_result_svg = get_svg_and_close_figure(fig)
            # guard_result_svg = imgdata.getvalue()
            if best_guard.is_explainable():
                # Find Explainable Representation
                # sampled_test_data = dpn.guard_manager_per_place[place].X_test.sample(
                #         n=min(100, len(dpn.guard_manager_per_place[place].X_test)));
                sampled_data = dpn.guard_manager_per_place[p].dataframe.sample(30)
                sampled_data.drop(['target'],axis=1,inplace=True)
                explainable_representation:plt.Figure = best_guard.get_global_explanations(sampled_data)
                svg_representations = {
                    plot_type: {'data': (get_svg_and_close_figure(explainable_representation) if type(explainable_representation) != str else explainable_representation), 'type': 'svg' if type(explainable_representation) != str else 'html'}
                    for plot_type, explainable_representation in explainable_representation.items()
                }
                # sampled_test_data = dpn.guard_manager_per_place[p].X_test.sample(
                #      n=min(100, len(dpn.guard_manager_per_place[p].X_test)));
            else:
                svg_representations = {}
            cache_representation(logid, id(p), dpn.ml_technique_per_place[p], svg_representations)
            failed_techniques = [str(key) for key in ml_techniques if key not in dpn.guard_manager_per_place[p].guards_list.keys()]
            return_info[id(p)] = {
                'performance': dpn.performance_per_place[p],
                'name': str(dpn.ml_technique_per_place[p]),
                'svg_representations': svg_representations,
                'guard_result_svg': guard_result_svg,
                'techniques': [str(key) for key in dpn.guard_manager_per_place[p].guards_list.keys()],
                'warning_text': '' if len(failed_techniques) == 0 else 'Some techniques failed: ' +  ', '.join([str(key) for key in ml_techniques if key not in dpn.guard_manager_per_place[p].guards_list.keys()]),
                'instances': dpn.guard_manager_per_place[p].dataframe.index.to_list(),
            }
        
        data_petri_nets[logid] = dpn
        return { 
            'mean_guard_conformance': dpn.get_mean_guard_conformance(event_log),
            'place_info': return_info
        }, 200;

@app.route("/log/<logid>/place/<int:placeid>/explainable-representation/<ml_technique>", methods=["GET"])
def get_explainable_representation(logid: str, placeid:int, ml_technique: str):
    dpn = data_petri_nets.get(logid, None)
    if dpn is None:
        return {"message": "No models trained yet."}, 400
    # Find the corresponding place object
    place = None
    for p in dpn.petri_net.places:
        if id(p) == placeid:
            place = p
            break
    if place is None:
        return {"message": "Place not found."}, 400

    # Get the Enum representation of the selected Technique
    technique_enum_value = get_ml_technique_from_str(ml_technique)
    if technique_enum_value is None:
        return {"message": "Invalid ML technique"}, 400

    # See if the representation exists:
    if logid in explainable_representations and (placeid,technique_enum_value) in explainable_representations[logid]:
        return {
            'svg_representations': explainable_representations[logid][(placeid,technique_enum_value)]
        }, 200
        # return explainable_representations[logid][(placeid,technique_enum_value)], 200	

    guards = dpn.guard_manager_per_place[place].guards_list


    selected_guard = guards[technique_enum_value]
    if selected_guard.is_explainable():
        # Find Explainable Representation
        # sampled_test_data = dpn.guard_manager_per_place[place].X_test.sample(
        #         n=min(100, len(dpn.guard_manager_per_place[place].X_test)));

        sampled_data = dpn.guard_manager_per_place[place].dataframe.sample(30)
        sampled_data.drop(['target'],axis=1,inplace=True)
        explainable_representation:plt.Figure = selected_guard.get_global_explanations(sampled_data)
        svg_representations = {
            plot_type: {'data': (get_svg_and_close_figure(explainable_representation) if type(explainable_representation) != str else explainable_representation), 'type': 'svg' if type(explainable_representation) != str else 'html'}
            for plot_type, explainable_representation in explainable_representation.items()
        }
    else:
        svg_representations = {}

    cache_representation(logid, placeid, technique_enum_value, svg_representations)

    return {
        'svg_representations': svg_representations
    }, 200
    # return svg_representation

@app.route("/log/<logid>/place/<int:placeid>/explainable-representation/<ml_technique>/local/<case_id>/<int:decision_repetition>", methods=["GET"])
def get_local_explanations(logid: str, placeid:int, ml_technique: str, case_id: str, decision_repetition: int):
    dpn = data_petri_nets.get(logid, None)
    if dpn is None:
        return {"message": "No models trained yet."}, 400
    # Find the corresponding place object
    place = None
    for p in dpn.petri_net.places:
        if id(p) == placeid:
            place = p
            break
    if place is None:
        return {"message": "Place not found."}, 400

    # Get the Enum representation of the selected Technique
    technique_enum_value = get_ml_technique_from_str(ml_technique)
    if technique_enum_value is None:
        return {"message": "Invalid ML technique"}, 400

    guards = dpn.guard_manager_per_place[place].guards_list


    selected_guard = guards[technique_enum_value]
    if selected_guard.is_explainable():
        # Find Explainable Representation
        sampled_test_data = dpn.guard_manager_per_place[place].X_test.sample(
                n=min(100, len(dpn.guard_manager_per_place[place].X_test)));
        local_data : DataFrame = dpn.guard_manager_per_place[place].dataframe.loc[[(case_id,decision_repetition)]]
        local_data.drop(['target'],axis=1, inplace=True)
        explainable_representations: Dict[str,plt.Figure] = selected_guard.get_local_explanations(local_data,sampled_test_data)

        svg_representations = {}

        svg_representations = {
            plot_type: get_svg_and_close_figure(explainable_representation)
            for plot_type, explainable_representation in explainable_representations.items()
        }
    else:
        svg_representations = {}

    return {
        'svg_representations': svg_representations
    }, 200

def get_svg_and_close_figure(figure: plt.Figure):
    imgdata = io.StringIO()
    figure.savefig(imgdata, format='svg', bbox_inches="tight")
    imgdata.seek(0)  # rewind the data
    plt.close(figure)
    return imgdata.getvalue()

def get_ml_technique_from_str(ml_technique:str):
    matching_techniques = [technique for technique in ML_Technique if str(technique) == ml_technique]
    if(len(matching_techniques) > 0):
        return matching_techniques[0]
    else:
        return None
    

def cache_representation(logid:str, placeid:int, technique_enum_value: ML_Technique, svg_representation:str):
    log_representations = explainable_representations.get(logid, dict())
    log_representations[(placeid, technique_enum_value)] = svg_representation
    explainable_representations[logid] = log_representations

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')