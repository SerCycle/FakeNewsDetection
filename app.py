from flask import Flask, render_template, request, jsonify
import numpy as np
from newsapi import NewsApiClient
import requests
import json
import random
import numpy as np
import pandas as pd
import seaborn as sns
import nltk
import re
import transformers
from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification
import tensorflow as tf
from tensorflow import keras

PRE_TRAINED_MODEL = 'indolem/indobert-base-uncased'
bert_tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL)
model = TFBertForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL, num_labels = 2, from_pt=True)
model.load_weights("model/nyoba_plis_dpt_89.h5")

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/home")
def home():
    return render_template("index.html")

@app.route("/hasil", methods=["POST"])
def hasil():
    text = request.form["isiberita"]
    input_text_tokenized =  bert_tokenizer.encode(text, truncation = True, padding = 'max_length', return_tensors = 'tf')
    bert_predict = model(input_text_tokenized)
    bert_output = tf.nn.softmax(bert_predict[0], axis = -1)
    kategori = ['Valid', 'Hoax']
    label = tf.argmax(bert_output, axis = 1)
    label = label.numpy()
    hasil = kategori[label[0]]

    return render_template('output.html', text=hasil)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/news")
def news():
    url = ('https://newsapi.org/v2/top-headlines?country=id&apiKey=e7dceaffc9274a57a8116a9cf19386a8')
    response = requests.get(url)
    newsapi = json.loads(response.content)
    return render_template("news.html", newsapi=newsapi['articles'])

if __name__ == '__main__':
    app.run(port=5000, debug=True)