from typing import Any, Dict, Tuple
from flask import Flask, render_template, request, redirect
import os
from datetime import datetime as dt

app = Flask("eXdpn")
from werkzeug.utils import secure_filename
from werkzeug.security import safe_join

import pm4py
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.log.obj import EventLog
from pm4py.objects.petri_net.obj import PetriNet, Marking

import uuid
from exdpn.util import import_log
from exdpn.decisionpoints import find_decision_points
from exdpn.guard_datasets import get_all_guard_datasets
from exdpn.guards import Guard_Manager, ML_Technique

# from exdpn.petri_net import get_petri_net

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
            # net, im, fm = get_petri_net(log)
            net, im, fm = inductive_miner.apply(log, variant=inductive_miner.Variants.IM)
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
        datasets = get_all_guard_datasets(loaded_event_logs[logid][1],discovered_models[logid][0],discovered_models[logid][1],discovered_models[logid][2],body['case_attributes'],body['event_attributes'])
        managers = dict()
        explainers = dict()
        evaluation_results = dict()
        for place,dataframe in datasets.items():
            guard_manager = Guard_Manager(dataframe, [ML_Technique.DT])
            evaluation = guard_manager.evaluate_guards()
            technique_name, trained_technique = guard_manager.get_best()
            evaluation_results[id(place)] = evaluation[technique_name]
            if trained_technique.is_explainable():
                explainable_representation = trained_technique.get_explainable_representation()
            else:
                explainable_representation = None
            explainers[id(place)] = explainable_representation
            print(f"Best technique for {place.name}: {technique_name}")
            managers[place] = guard_manager
        return {'evaluation_results': evaluation_results}, 200
