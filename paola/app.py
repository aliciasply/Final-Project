import numpy as np
from flask import Flask, request, jsonify, render_template, redirect
import pickle

app = Flask(__name__)
model = pickle.load(open('fake_news_model.pickle','rb'))

####################################
# Flask Routes
####################################
# HOMEPAGE
@app.route("/")
def homepage():
    return render_template("index.html")

# Route for model
@app.route("/api", methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict([[np.array(data['exp'])]])
    output = prediction[0]
    return jsonify(output)


if __name__ == '__main__':
    app.run(debug = True, port=5000)