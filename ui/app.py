from typing import Any, Dict, Tuple
from flask import Flask, render_template, request, redirect
import os
from datetime import datetime as dt

app = Flask("eXdpn")
from werkzeug.utils import secure_filename
from werkzeug.security import safe_join
import pm4py
import uuid
from exdpn import load_event_log


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
uploaded_logs = {
    file_name: get_file_info(file_name) for file_name in os.listdir("./uploads/")
}

loaded_event_logs: Dict[
    str, Tuple[Dict[str, Any], pm4py.objects.log.obj.EventLog]
] = dict()


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
            uploaded_logs.pop(logid)
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
            xes_stats = loaded_event_logs[logid][0]
        else:
            path = get_upload_path(logid)
            # Check for validity
            if path is not None and os.path.exists(path):
                xes: pm4py.objects.log.obj.EventLog = load_event_log.import_xes(path)

                events = [evt for case in xes for evt in case]
                activities = {evt["concept:name"] for evt in events}
                xes_stats = {
                    "num_cases": len(xes),
                    "num_events": len(events),
                    "num_activities": len(activities),
                }

                loaded_event_logs[logid] = (xes_stats, xes)
            else:
                return {"message": "Invalid path"}, 400
        return xes_stats, 200
    else:
        return {"message": "Log not found"}, 400
