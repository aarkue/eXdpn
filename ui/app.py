import io
from typing import Any, Dict, Tuple
from flask import Flask, render_template, request, redirect
import os
from datetime import datetime as dt

app = Flask("eXdpn")
from werkzeug.utils import secure_filename
from werkzeug.security import safe_join

import pm4py
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.objects.log.obj import EventLog
from pm4py.objects.petri_net.obj import PetriNet, Marking

import uuid
from exdpn.util import import_log
from exdpn.decisionpoints import find_decision_points
from exdpn.guards import ML_Technique
from exdpn.data_petri_net import Data_Petri_Net
import matplotlib.pyplot as plt
from exdpn.petri_net import get_petri_net

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
            loaded_event_logs.pop(logid)
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
        return {"message": "Log not loaded"}, 400
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
    if logid not in loaded_event_logs and logid in discovered_models:
        return {"message": "Log or model not loaded"}, 400
    else:
        body = request.get_json()

        dpn = Data_Petri_Net(
            event_log = loaded_event_logs[logid][1],
            petri_net = discovered_models[logid][0],
            initial_marking = discovered_models[logid][1],
            final_marking = discovered_models[logid][2],
            case_level_attributes = body['case_attributes'],
            event_level_attributes = body['event_attributes'],
            guard_threshold = 0
        )
        return_info = dict()
        def convert_ML_enum_to_name(ml_technique):
            if ml_technique == ML_Technique.DT:
                return "Decision Tree"
            elif ml_technique == ML_Technique.SVM:
                return "Support Vector Machine"
            elif ml_technique == ML_Technique.LR:
                return "Logistic Regression"
            elif ml_technique == ML_Technique.NN:
                return "Neural Network"
            else:
                return "Unknown"
        x = False;
        for p in dpn.get_best():
            best_guard = dpn.get_guard_at_place(p)
            if not x:
                print(best_guard.feature_names)
                x = True
            if best_guard.is_explainable():
                # Find Explainable Representation
                explainable_representation:plt.Figure = best_guard.get_explainable_representation()
                imgdata = io.StringIO()
                explainable_representation.savefig(imgdata, format='svg', bbox_inches="tight")
                imgdata.seek(0)  # rewind the data
                svg_representation = imgdata.getvalue()
            else:
                svg_representation = ""
            return_info[id(p)] = {
                'performance': dpn.performance_per_place[p],
                'name': convert_ML_enum_to_name(dpn.ml_technique_per_place[p]),
                'svg_representation': svg_representation
            }
        
        return return_info, 200;
