from flask import Flask, render_template, request, redirect
import os
from datetime import datetime as dt
app = Flask('eXdpn')
from werkzeug.utils import secure_filename
import uuid


def get_upload_path(name):
    return os.path.join('./uploads/', name)


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
@app.route("/log/<logid>")
def log_page(logid: str):
    log = uploaded_logs[logid]
    return render_template("log.html",log=log)