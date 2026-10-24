from flask import Flask, render_template, jsonify
from utils.data_utils import build_payload

app = Flask(__name__)
PAYLOAD = build_payload()          # computed once at startup (old app retrained the model on every request)

@app.route("/")
def index():
    return render_template("index.html", data=PAYLOAD)

@app.route("/api/data")
def api_data():
    return jsonify(PAYLOAD)

if __name__ == "__main__":
    app.run(debug=True, port= 5160)
