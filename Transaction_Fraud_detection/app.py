# Copyright (c) 2025 Buddham Rajbhandari (RA2311026011107)
# Email: buddhamrajbhandari30@gmail.com
# This project is licensed under the MIT License – see the LICENSE file for details.
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

app = Flask(__name__)

model_dir = 'models'
model_path = os.path.join(model_dir, 'fraud_model.pkl')
columns_path = os.path.join(model_dir, 'model_columns.pkl')
preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
label_encoders_path = os.path.join(model_dir, 'label_encoders.pkl')

with open(model_path, 'rb') as file:
    model = pickle.load(file)

with open(columns_path, 'rb') as file:
    model_columns = pickle.load(file)

with open(label_encoders_path, 'rb') as file:
    label_encoders = pickle.load(file)

with open(preprocessor_path, 'rb') as file:
    preprocessor = pickle.load(file)
    try:
        # Extract feature names from the preprocessor
        numeric_features = preprocessor.transformers_[0][2]
        categorical_features = preprocessor.transformers_[1][2]
    except:
        # Fallback if extraction fails
        numeric_features = [col for col in model_columns if not col.startswith('Transaction')]
        categorical_features = [col for col in model_columns if col not in numeric_features]

# Available options for dropdown fields
location_options = ["New York", "Los Angeles", "Chicago", "Houston", "Miami", 
                   "Seattle", "San Francisco", "Dallas", "Boston", "Denver"]

payment_methods = ["Credit Card", "Debit Card", "PayPal", "Apple Pay", 
                  "Google Pay", "Bank Transfer", "Cryptocurrency", "Gift Card"]

device_options = ["Desktop", "Mobile", "Tablet", "Smart TV", "IoT Device"]

@app.route('/')
def home():
    return render_template('index.html', 
                          locations=location_options,
                          payment_methods=payment_methods,
                          devices=device_options)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.form.to_dict()
        
        # Create a DataFrame with a single row
        input_data = pd.DataFrame([data])
        
        # Convert numeric fields
        numeric_fields = ['Transaction Amount', 'Previous Transactions Count', 'Account Age Days']
        for field in numeric_fields:
            if field in input_data:
                input_data[field] = pd.to_numeric(input_data[field], errors='coerce')
        
        # Process transaction date
        transaction_date = datetime.now()
        if 'transaction_date' in data and data['transaction_date']:
            try:
                transaction_date = datetime.strptime(data['transaction_date'], '%Y-%m-%d')
            except:
                pass
                
        # Add date features
        input_data['Transaction Day'] = transaction_date.day
        input_data['Transaction Month'] = transaction_date.month
        input_data['Transaction Year'] = transaction_date.year
        input_data['Transaction Weekday'] = transaction_date.weekday()
        
        # Make sure all required columns are present
        for col in model_columns:
            if col not in input_data.columns:
                input_data[col] = np.nan
        
        # Reorder columns to match model expectations
        input_data = input_data[model_columns]
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        # Determine risk level
        risk_level = "Low"
        if probability > 0.7:
            risk_level = "High"
        elif probability > 0.4:
            risk_level = "Medium"
            
        # Create response
        result = {
            'prediction': int(prediction),
            'probability': float(probability),
            'risk_level': risk_level,
            'is_fraud': bool(prediction),
            'confidence': f"{probability * 100:.2f}%"
        }
        
        return jsonify(result)

@app.route('/sample_data', methods=['GET'])
def sample_data():
    sample_type = request.args.get('type', 'legitimate')
    
    if sample_type == 'fraudulent':
        # Sample fraudulent transaction
        return jsonify({
            'Transaction Amount': '2999.99',
            'Customer Location': 'Miami',
            'Payment Method': 'Gift Card',
            'Device Used': 'Mobile',
            'Previous Transactions Count': '1',
            'Account Age Days': '3',
            'Shipping Address': 'Different from billing address',
            'transaction_date': datetime.now().strftime('%Y-%m-%d')
        })
    else:
        # Sample legitimate transaction
        return jsonify({
            'Transaction Amount': '149.99',
            'Customer Location': 'New York',
            'Payment Method': 'Credit Card',
            'Device Used': 'Desktop',
            'Previous Transactions Count': '12',
            'Account Age Days': '187',
            'Shipping Address': 'Same as billing address',
            'transaction_date': datetime.now().strftime('%Y-%m-%d')
        })

if __name__ == '__main__':
    app.run(debug=True)
