from flask import Flask, request
from flask_cors import CORS
from models.knn import KNN
from src.constants import constants
from src.data import get_train_test_split, regexp_processing

model = KNN(classes=constants['classes'], k=3)
train, _ = get_train_test_split(
    '../10_ports.csv', split=1
)
model.fit(train)


app = Flask(__name__)
CORS(app)


@app.route('/', methods=['GET'])
def welcome():
    return 'Welcome to ANAIS - ENSTA web app'


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        destination = request.data.decode('utf-8')

        if destination.upper() in model.destinations.keys():
            return model.destinations[destination.upper()]

        pred = model(destination)
        return pred[0]


if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)
