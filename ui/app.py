from flask import Flask, render_template, request, redirect
import os
from datetime import datetime as dt
app = Flask('eXdpn')
from werkzeug.utils import secure_filename
from werkzeug.security import safe_join
import uuid



def get_upload_path(name):
    return safe_join('./uploads/', name)


def get_file_info(name):
    return {
        'name': name.split("@")[0],
        'size': os.path.getsize(get_upload_path(name)), 
        'uploaded_at': dt.fromtimestamp(os.path.getmtime(get_upload_path(name))).strftime("%Y-%m-%d %H:%M:%S")
    }


# Initialize uploaded logs from uploads folder
uploaded_logs = { 
    file_name: get_file_info(file_name)
    for file_name in os.listdir('./uploads/')
}


# Index page
@app.route("/")
def index():
    return render_template("index.html",uploaded_logs=uploaded_logs)


# Form for uploading an event log
@app.route("/load-log", methods=["GET","POST"])
def load_log():
    if request.method == "POST":
        # Get the form input from request
        log = request.files['log']
        safe_name = secure_filename(log.filename);
        safe_name = safe_name.replace('@','_') + "@" + str(uuid.uuid4())

        save_path = get_upload_path(safe_name)
        log.save(save_path)
        uploaded_logs[safe_name] = get_file_info(safe_name)
        return redirect('/')
    else:
        return render_template("load-log.html")

# Details page for a single log
@app.route("/log/<logid>", methods=["GET","DELETE"])
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
        log = uploaded_logs[logid]
        return render_template("log.html",log=log)