import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, send_from_directory
import pickle

app = Flask(__name__, template_folder='templates', static_url_path='')
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
@app.route("/api", methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict([[np.array(data['exp'])]])
    output = prediction[0]
    return jsonify(output)


if __name__ == '__main__':
    app.run(debug = True, port=5000)