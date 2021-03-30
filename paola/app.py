import numpy as np
from flask import Flask, request, jsonify, render_template, redirect
from flask_cors import CORS
import os
from sklearn.externals import joblib
import pickle
import newspaper
from newspaper import Article
import urllib
import nltk
# Download punkt sentence tokenizer
nltk.download('punkt')

#Load flask and model
app = Flask(__name__)
CORS
model = pickle.load(open('fake_news_model.pickle','rb'))

####################################
# Flask Routes
####################################
# HOMEPAGE
@app.route("/")
def homepage():
    return render_template("index.html")

# Route for model
@app.route("/detect", methods=['GET', 'POST'])
def predict():


    news = article.summary

    # Passing the news article and returning if it is 'Real' or 'Fake'
    prediction = model.predict([news])
    return render_template('index.html', article_text = 'This article is "{}"'.format(pred[0]))



if __name__ == '__main__':
    app.run(debug = True, port=5000)