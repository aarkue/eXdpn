from flask import Flask, render_template
app = Flask('eXdpn')

@app.route("/")
def index():
    return render_template("index.html")

