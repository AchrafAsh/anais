import json

from app.db import add_prediction, rate_prediction, get_training_data
from flask import Flask, request, jsonify
from flask_cors import CORS
from models.knn import KNN
from src.constants import constants
from src.data import get_train_test_split, regexp_processing

model = KNN(classes=constants['classes'], k=3)
# train, _ = get_train_test_split(split=1)
train = get_training_data()
model.fit(train)


app = Flask(__name__)
CORS(app)


@app.route('/', methods=['GET'])
def welcome():
    return 'Welcome to ANAIS - ENSTA web app'


@app.route('/predict', methods=['POST'])
def predict():
    destination = request.data.decode('utf-8')

    if destination.upper() in model.destinations.keys():
        return jsonify(id=-1, label=model.destinations[destination.upper()])

    pred = model(destination)
    pred_id = add_prediction(destination=destination, label=pred[0])

    return jsonify(id=pred_id, label=pred[0])


@ app.route('/rate', methods=['POST'])
def rate():
    data = json.loads(request.data)
    rate_prediction(data['id'], data['rate'])
    return 'OK'
