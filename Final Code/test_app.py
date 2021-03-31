import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, send_from_directory
import pickle
from flask_cors import CORS
import os
import newspaper
from newspaper import Article
import urllib
import nltk
# Download punkt sentence tokenizer
nltk.download('punkt')

app = Flask(__name__, template_folder='templates', static_url_path='')
CORS(app)
model = pickle.load(open('fake_news_model.pickle','rb'))

####################################
# Flask Routes
####################################
# HOMEPAGE
@app.route("/")
@app.route("/index4.html")
def home():
    return render_template("/index4.html")


# Route for model
@app.route("/detect", methods=['GET', 'POST'])
def predict():
    url = request.get_data(as_text = True)
    url = urllib.parse.unquote(url)
    article = Article(str(url))
    article.download()
    article.parse()
    article.nlp()

    news = article.summary

    # Passing the news article and returning if it is 'Real' or 'Fake'
    prediction = model.predict([news])
    return render_template('index4.html', article_text = 'This article is "{}"'.format(prediction[0]))


if __name__ == '__main__':
    app.run(debug = True, port=5000)