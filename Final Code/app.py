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
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
import pandas as pd


app = Flask(__name__, template_folder='templates', static_url_path='')
CORS(app)
model = pickle.load(open('fake_news_model.pickle','rb'))

xy_train = pd.read_csv("./xy_train.csv")
x_train_vec = tfidf_vectorizer.fit_transform(xy_train['text'])
model.fit(x_train_vec, xy_train['label'])


####################################
# Flask Routes
####################################
# HOMEPAGE
@app.route("/")
@app.route("/index4.html")
def home():
    return render_template("/index4.html")


# Route for model
@app.route("/results", methods=['GET', 'POST'])
def predict():
    
    # url = request.get_data(as_text = True)[5:]
    url = request.form["URL"]
    url = urllib.parse.unquote(url)
    article = Article(str(url))
    article.download()
    article.parse()
    article.nlp()
    
    news = article.summary
    prep_news = tfidf_vectorizer.transform(pd.Series(news))

    # Passing the news article and returning if it is 'Real' or 'Fake'
    prediction = model.predict(prep_news)
    print("=========================")
    print(prediction)
    print("=========================")
    return render_template('index4.html', article_text = 'This article is "{}"'.format(prediction[0]))

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(port = port, debug = True, use_reloader = False)