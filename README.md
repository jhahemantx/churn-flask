# Customer Churn Prediction App

## Overview
This project is a **Customer Churn Prediction** web application built using **Flask** and a **TensorFlow** deep learning model. The app predicts whether a customer is likely to churn based on various input features.

## Features
- Uses a **deep learning model** (`model.h5`) for churn prediction.
- Accepts customer details such as **credit score, age, tenure, balance, etc.**
- Encodes categorical features (`Gender` and `Geography`) using **LabelEncoder** and **OneHotEncoder**.
- Scales numerical inputs using **StandardScaler**.
- Provides a **probability score** for churn prediction.
- Web interface for easy input and real-time predictions.

## Project Structure
```
|-- customer_churn_app
    |-- templates/
        |-- index.html         # Web UI template
    |-- model.h5           # Trained TensorFlow model
    |-- gender_encoding.pkl  # Label encoder for gender
    |-- geography_encoder.pkl  # One-hot encoder for geography
    |-- standard_scaler.pkl   # Standard scaler for numerical data
    |-- app.py                 # Main Flask application
    |-- README.md              # Project documentation
    |-- requirements.txt       # Dependencies
```

## Installation
To set up the environment and install dependencies, run:
```bash
pip install flask numpy pandas tensorflow scikit-learn
```

## Usage
### Running the Web Application
To launch the Flask app, execute:
```bash
python app.py
```
Then, open `http://127.0.0.1:5000/` in your browser.

### Predicting Customer Churn
1. Enter customer details in the web form.
2. Click the **Predict** button.
3. The model will predict whether the customer is likely to churn.

## Model Details
- **Deep Learning Model:** A TensorFlow-based model trained to predict customer churn.
- **Preprocessing Steps:**
  - **Label Encoding:** Converts `Gender` values to numerical format.
  - **One-Hot Encoding:** Converts `Geography` into multiple binary columns.
  - **Feature Scaling:** Uses `StandardScaler` for numerical normalization.

## Future Enhancements
- Improve model performance using **LSTMs or XGBoost**.
- Deploy the app using **Docker and AWS**.
- Add a **dashboard for analytics and insights**.
- Integrate a **customer retention strategy** based on predictions.

