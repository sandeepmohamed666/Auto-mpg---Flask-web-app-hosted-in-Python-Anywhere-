from flask import Flask, render_template, request
# import numpy as np
# import pickle
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model

model = joblib.load('model.pkl')
import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")

# model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        cylinders = float(request.form['cylinders'])
        displacement = float(request.form['displacement'])
        horsepower = float(request.form['horsepower'])
        weight = float(request.form['weight'])
        acceleration = float(request.form['acceleration'])
        model_year = float(request.form['model_year'])
        origin = request.form['origin']

        # Create dataframe
        input_data = pd.DataFrame({
            'cylinders': [cylinders],
            'displacement': [displacement],
            'horsepower': [horsepower],
            'weight': [weight],
            'acceleration': [acceleration],
            'model year': [model_year],
            'origin': [origin]
        })

        # Prediction
        prediction = model.predict(input_data)[0]

        return render_template(
            'index.html',
            prediction_text=f'Predicted MPG: {prediction:.2f}'
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f'Error: {str(e)}'
        )

if __name__ == '__main__':
    app.run(debug=True)

# app = Flask(__name__)

# # Load model & scaler
# model = pickle.load(open("model.pkl", "rb"))
# scaler = pickle.load(open("scaler.pkl", "rb"))

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():

#     # Collect inputs (example features — adjust based on dataset columns)
#     cylinders = float(request.form['cylinders'])
#     displacement = float(request.form['displacement'])
#     horsepower = float(request.form['horsepower'])
#     weight = float(request.form['weight'])
#     acceleration = float(request.form['acceleration'])
#     model_year = float(request.form['model_year'])
#     origin = float(request.form['origin'])

#     # Create feature array
#     input_data = np.array([[cylinders, displacement, horsepower,
#                             weight, acceleration, model_year, origin]])

#     # Scale input
#     input_scaled = scaler.transform(input_data)

#     # Predict
#     prediction = model.predict(input_scaled)[0]

#     return render_template("result.html", prediction=prediction)

# if __name__ == "__main__":
#     app.run(debug=True)














# from flask import Flask, render_template

# app = Flask(__name__)

# @app.route("/")
# def home():
#     return render_template("index.html")