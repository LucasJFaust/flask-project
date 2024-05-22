import pickle
import pandas as pd
from flask import Flask, request  # Corrigido de 'requests' para 'request'

# O primeiro passo ao se fazer uma aplicação Flask, é instanciar a aplicação
app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return "Welcome to the Medical Insurance Prediction API"

@app.route('/predict', methods=['POST'])
def index():
    data_json = request.get_json()['data']
    df = pd.DataFrame(data_json)

    with open('models/model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    output = model.predict(df).tolist()
    return str(output)

# Sempre no final de uma aplicação Flask temos que chamar o run
if __name__ == '__main__':
    app.run(debug=True)
