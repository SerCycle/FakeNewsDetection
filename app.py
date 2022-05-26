
from flask import Flask, render_template, request, jsonify
import flask
import joblib
import numpy as np
from newsapi import NewsApiClient
import requests
import json

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/home")
def home():
    return render_template("index.html")

@app.route("/hasil", methods=["POST"])
def output():
    text = request.form["text"]
    # model = joblib.load('------')
    return render_template('output.html', text=text)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/news", methods=["GET"])
def news():
    url = ('https://newsapi.org/v2/top-headlines?country=id&apiKey=e7dceaffc9274a57a8116a9cf19386a8')
    response = requests.get(url)
    newsapi = json.loads(response.content)
    return render_template("news.html", newsapi=newsapi['articles'])

if __name__ == '__main__':
    app.run(port=5000, debug=True)