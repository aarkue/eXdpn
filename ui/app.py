from flask import Flask, render_template, request
from werkzeug import FileStorage

app = Flask('eXdpn')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/load-log", methods=["GET","POST"])
def load_log():
    if request.method == "POST":
        # Get the form input from request
        log: FileStorage = request.files['log']
        return "<h1>Loading file...</h1>"
    else:
        return render_template("load-log.html")