import numpy as np
from flask import Flask, request, jsonify
import pickle

####################################
# Flask Routes
####################################
app = Flask(__name__)
model = pickle.load(open('fake_news_model.pickle','rb'))

@app.route('/api', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json(force=True)

    # Make prediction using loaded model
    prediction = model.predict([[np.array(data['exp'])]])

    # Take the first value of prediction
    output = prediction[0]

    return jsonify(output)


if __name__ == '__main__':
    app.run(debug=True, port=5000)