from crypt import methods
from flask import Flask, render_template, request, jsonify
import flask
import joblib
import numpy as np

app = Flask(__name__)

app.route("/")
def index():
    return render_template("index.html")

app.route("/hasil", methods=["POST"])
def output():
    text = request.form["text"]
    model = load('------')
    return render_template('output.html', text)