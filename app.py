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
from sklearn import preprocessing
import transformers
from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification
import tensorflow as tf
from tensorflow import keras
import sqlite3
import os

def text_preprocessing(text):
    text = text.lower()                               
    text = re.sub(r'https?://\S+|www\.\S+', '', text) 
    text = re.sub(r'[-+]?[0-9]+', '', text)           
    text = re.sub(r'[^\w\s]','', text)
    text = re.sub(r'ï½','', text)
    text = re.sub(r'ï¿½','', text)
    text = re.sub(r'\n',' ', text)                                         
    text = text.strip()                               
    return text

PRE_TRAINED_MODEL = 'indolem/indobert-base-uncased'
bert_tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL)
model = TFBertForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL, num_labels = 2, from_pt=True)
model.load_weights("model/nyoba_plis_dpt_89.h5")

def add_history(teks, hasil):
    a = teks
    b = hasil
    conn = sqlite3.connect("sotaken.db")
    history = conn.cursor()
    history.execute("insert into history (teks, hasil) values('{a}','{b}');".format(a = a, b = b))
    conn.commit()

def history_data():
    conn = sqlite3.connect("sotaken.db")
    history = conn.cursor()
    history.execute("SELECT * FROM history ORDER BY id DESC LIMIT 3;")
    rows = history.fetchall()
    return rows

app = Flask(__name__)

@app.route("/")
def index():
    history = history_data()
    return render_template("index.html", history=history)

@app.route("/hasil", methods=["POST"])
def hasil():

    rawtext = request.form["isiberita"]
    text = text_preprocessing(rawtext)
    input_text_tokenized =  bert_tokenizer.encode(text, truncation = True, padding = 'max_length', return_tensors = 'tf')
    bert_predict = model(input_text_tokenized)
    bert_output = tf.nn.softmax(bert_predict[0], axis = -1)
    kategori = ['Valid', 'Hoax']
    label = tf.argmax(bert_output, axis = 1)
    label = label.numpy()
    hasil = kategori[label[0]]

    add_history(rawtext, hasil)
    history = history_data()

    return render_template('output.html', text=rawtext, hasil=hasil, history=history)

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