# app.py
from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

app = Flask(__name__)

# Load models and encoders
model = tf.keras.models.load_model('model.h5')

with open('gender_encoding.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

with open('geography_encoder.pkl', 'rb') as file:
    ohe = pickle.load(file)

with open('standard_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html',
                         geography_options=ohe.categories_[0].tolist(),
                         gender_options=label_encoder.classes_.tolist())

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = request.form

    # Prepare input data with explicit column names
    input_data = pd.DataFrame({
        'CreditScore': [float(data['credit_score'])],
        'Gender': [label_encoder.transform([data['gender']])[0]],
        'Age': [float(data['age'])],
        'Tenure': [float(data['tenure'])],
        'Balance': [float(data['balance'])],
        'NumOfProducts': [float(data['num_of_products'])],
        'HasCrCard': [float(data['has_cr_card'])],
        'IsActiveMember': [float(data['is_active_member'])],
        'EstimatedSalary': [float(data['estimated_salary'])],
        'Geography': [data['geography']]  # Add Geography explicitly
    })

    # Apply OneHotEncoder for Geography
    geo_encoded = ohe.transform(input_data[['Geography']])
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe.get_feature_names_out(['Geography']))

    # Concatenate the encoded geography with other features
    input_data = pd.concat([input_data.drop(columns=['Geography']).reset_index(drop=True), geo_encoded_df], axis=1)

    # Scale data
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)
    prediction_proba = float(prediction[0][0])

    return jsonify({
        'probability': prediction_proba,
        'will_churn': prediction_proba > 0.5
    })

if __name__ == '__main__':
    app.run(debug = False)
