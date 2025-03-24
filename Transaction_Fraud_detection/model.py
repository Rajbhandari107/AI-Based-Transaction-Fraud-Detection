import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import os

start_time = time.time()

print("Starting model training process...")
print("Step 1/9: Loading data...")


data = pd.read_csv('fraud_data.csv')
print(f"Data loaded successfully. Shape: {data.shape}")
print(f"Time elapsed: {time.time() - start_time:.2f} seconds")

print("\nStep 2/9: Preprocessing data...")

# Convert date to datetime and extract features
if 'Transaction Date' in data.columns:
    data['Transaction Date'] = pd.to_datetime(data['Transaction Date'], errors='coerce')
    data['Transaction Day'] = data['Transaction Date'].dt.day
    data['Transaction Month'] = data['Transaction Date'].dt.month
    data['Transaction Year'] = data['Transaction Date'].dt.year
    data['Transaction Weekday'] = data['Transaction Date'].dt.weekday
else:
    # If Transaction Date doesn't exist, create default values
    print("Warning: 'Transaction Date' column not found. Creating default date features.")
    data['Transaction Day'] = 1
    data['Transaction Month'] = 1
    data['Transaction Year'] = 2025
    data['Transaction Weekday'] = 0

# Create directory for model artifacts
if not os.path.exists('models'):
    os.makedirs('models')
    print("Created 'models' directory")

# Handle categorical variables - identify them dynamically
categorical_features = data.select_dtypes(include=['object', 'category']).columns.tolist()
if 'Transaction Date' in categorical_features:
    categorical_features.remove('Transaction Date')
if 'Is Fraudulent' in categorical_features:
    categorical_features.remove('Is Fraudulent')
if 'Transaction ID' in categorical_features:
    categorical_features.remove('Transaction ID')
if 'Customer ID' in categorical_features:
    categorical_features.remove('Customer ID')
if 'IP Address' in categorical_features:
    categorical_features.remove('IP Address')

print(f"Identified {len(categorical_features)} categorical features: {categorical_features}")

# Identify numeric features dynamically
numeric_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
# Remove ID columns and target variable from numeric features
if 'Transaction ID' in numeric_features:
    numeric_features.remove('Transaction ID')
if 'Customer ID' in numeric_features:
    numeric_features.remove('Customer ID')
if 'Is Fraudulent' in numeric_features:
    numeric_features.remove('Is Fraudulent')

print(f"Identified {len(numeric_features)} numeric features: {numeric_features}")

# Features to drop (non-predictive or unique identifiers)
drop_features = []
if 'Transaction ID' in data.columns:
    drop_features.append('Transaction ID')
if 'Customer ID' in data.columns:
    drop_features.append('Customer ID')
if 'Transaction Date' in data.columns:
    drop_features.append('Transaction Date')
if 'IP Address' in data.columns:
    drop_features.append('IP Address')
if 'Is Fraudulent' in data.columns:
    drop_features.append('Is Fraudulent')

print(f"Features to drop: {drop_features}")
print(f"Time elapsed: {time.time() - start_time:.2f} seconds")

print("\nStep 3/9: Preparing feature and target variables...")
# Check if Is Fraudulent exists
if 'Is Fraudulent' in data.columns:
    # Convert to binary if it's not already
    if data['Is Fraudulent'].dtype == 'object':
        # Handle potential string values like 'Yes'/'No' or 'True'/'False'
        if data['Is Fraudulent'].nunique() <= 2:
            # Map string values to binary
            unique_values = data['Is Fraudulent'].unique()
            if len(unique_values) == 2:
                # Determine which value should be considered "fraud"
                if 'Yes' in unique_values or 'True' in unique_values or '1' in unique_values or 1 in unique_values:
                    # Create mapping dictionary
                    fraud_map = {val: 1 if val in ['Yes', 'True', '1', 1] else 0 for val in unique_values}
                    data['Is Fraudulent'] = data['Is Fraudulent'].map(fraud_map)
                else:
                    # Default mapping (first value = 0, second value = 1)
                    data['Is Fraudulent'] = data['Is Fraudulent'].map({unique_values[0]: 0, unique_values[1]: 1})
    
    # Separate features and target
    X = data.drop(drop_features, axis=1)
    y = data['Is Fraudulent']
else:
    print("Warning: 'Is Fraudulent' column not found. Creating dummy target variable.")
    X = data.drop(drop_features, axis=1)
    # Create a dummy target variable (all legitimate) to allow the model to train
    y = pd.Series(0, index=data.index)

print(f"Features shape: {X.shape}, Target shape: {y.shape}")
print(f"Target distribution:\n{y.value_counts(normalize=True) * 100}")
print(f"Time elapsed: {time.time() - start_time:.2f} seconds")

print("\nStep 4/9: Creating and saving preprocessing components...")

# Create and save label encoders for categorical variables
label_encoders = {}
for column in categorical_features:
    le = LabelEncoder()
    le.fit(data[column].fillna('missing').astype(str))
    label_encoders[column] = le

# Save label encoders
with open('models/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
print(f"Label encoders saved to models/label_encoders.pkl")

# Create and save standard scaler for numeric features
scaler = StandardScaler()
scaler.fit(data[numeric_features].fillna(data[numeric_features].median()))

# Save scaler
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print(f"Standard scaler saved to models/scaler.pkl")

print("\nStep 5/9: Building model pipeline...")

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])


numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    sparse_threshold=0
)

# Create a RandomForest model
print("Using RandomForestClassifier for fraud detection...")
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, max_depth=15, 
                                          min_samples_split=10, random_state=42, 
                                          n_jobs=-1, class_weight='balanced'))
])

print("Model pipeline created successfully.")
print(f"Time elapsed: {time.time() - start_time:.2f} seconds")

print("\nStep 6/9: Splitting data and training model...")
# Split data with stratification to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training on {X_train.shape[0]} samples, testing on {X_test.shape[0]} samples")
print("Starting model training (this may take a few minutes)...")
train_start_time = time.time()

# Train the model
model.fit(X_train, y_train)

training_time = time.time() - train_start_time
print(f"Model training completed in {training_time:.2f} seconds")

# Evaluate the model
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
y_test_prob = model.predict_proba(X_test)[:, 1]

train_score = accuracy_score(y_train, y_train_pred)
test_score = accuracy_score(y_test, y_test_pred)
print(f"Training accuracy: {train_score:.4f}")
print(f"Testing accuracy: {test_score:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))
print(f"Total time elapsed: {time.time() - start_time:.2f} seconds")

print("\nStep 7/9: Generating confusion matrix and model evaluation visualizations...")
# Create static directory if it doesn't exist
if not os.path.exists('static'):
    os.makedirs('static')
    print("Created 'static' directory")

# Generate confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Legitimate', 'Fraudulent'],
            yticklabels=['Legitimate', 'Fraudulent'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('static/confusion_matrix.png')
print("Confusion matrix visualization saved.")

# Try to get feature names
try:
    # Get feature names from pipeline
    all_feature_names = (
        numeric_features + 
        model.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_features).tolist()
    )
    
    # Get feature importances
    importances = model.named_steps['classifier'].feature_importances_
    
    # Sort feature importances
    indices = np.argsort(importances)[::-1]
    num_features_to_show = min(15, len(all_feature_names))
    sorted_feature_names = [all_feature_names[i] for i in indices[:num_features_to_show]]
    sorted_importances = importances[indices[:num_features_to_show]]
    
    # Create a feature importance plot
    plt.figure(figsize=(12, 8))
    plt.title("Top Feature Importances for Fraud Detection")
    bars = plt.bar(range(num_features_to_show), sorted_importances, align='center', color='#3498db')
    plt.xticks(range(num_features_to_show), sorted_feature_names, rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                f'{height:.4f}', ha='center', va='bottom', rotation=0)
                
    plt.tight_layout()
    plt.savefig('static/feature_importance.png')
    print("Feature importance visualization saved.")
except Exception as e:
    print(f"Warning: Could not generate feature importance plot. Error: {e}")

# Create ROC curve
plt.figure(figsize=(8, 6))
from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(y_test, y_test_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('static/roc_curve.png')
print("ROC curve visualization saved.")

print("\nStep 8/9: Creating data analysis visualizations...")
# Create data analysis visualizations
try:
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Fraud distribution
    plt.subplot(2, 2, 1)
    fraud_counts = data['Is Fraudulent'].value_counts()
    plt.pie(fraud_counts, labels=['Legitimate', 'Fraudulent'], autopct='%1.1f%%', 
            colors=['#4CAF50', '#F44336'], explode=(0, 0.1))
    plt.title('Distribution of Fraudulent Transactions')
    
    # Plot 2: Transaction Amount vs Is Fraudulent
    if 'Transaction Amount' in data.columns:
        plt.subplot(2, 2, 2)
        sns.boxplot(x='Is Fraudulent', y='Transaction Amount', data=data, palette=['#4CAF50', '#F44336'])
        plt.title('Transaction Amount vs Fraud Status')
        plt.ylabel('Transaction Amount')
        plt.xlabel('Is Fraudulent (0=Legitimate, 1=Fraudulent)')
    
    # Plot 3: Payment Method Distribution
    if 'Payment Method' in data.columns:
        plt.subplot(2, 2, 3)
        payment_fraud = pd.crosstab(data['Payment Method'], data['Is Fraudulent'])
        payment_fraud_pct = payment_fraud.div(payment_fraud.sum(axis=1), axis=0) * 100
        if 1 in payment_fraud_pct.columns:
            payment_fraud_pct[1].sort_values(ascending=False).head(5).plot(kind='bar', color='#FF5722')
            plt.title('Top 5 Payment Methods by Fraud Rate')
            plt.ylabel('Fraud Rate (%)')
            plt.xlabel('Payment Method')
    
    # Plot 4: Device Used vs Fraud
    if 'Device Used' in data.columns:
        plt.subplot(2, 2, 4)
        device_fraud = pd.crosstab(data['Device Used'], data['Is Fraudulent'])
        device_fraud_pct = device_fraud.div(device_fraud.sum(axis=1), axis=0) * 100
        if 1 in device_fraud_pct.columns:
            device_fraud_pct[1].sort_values(ascending=False).plot(kind='bar', color='#2196F3')
            plt.title('Device Used vs Fraud Rate')
            plt.ylabel('Fraud Rate (%)')
            plt.xlabel('Device Used')
    
    plt.tight_layout()
    plt.savefig('static/data_analysis.png')
    print("Data analysis visualizations saved.")
except Exception as e:
    print(f"Warning: Could not generate data analysis plots. Error: {e}")
print(f"Time elapsed: {time.time() - start_time:.2f} seconds")

print("\nStep 9/9: Saving model and feature information...")
# Save the model
with open('models/fraud_model.pkl', 'wb') as f:
    pickle.dump(model, f)
    print("Model saved as models/fraud_model.pkl")

# Save column names for the web app
with open('models/model_columns.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)
    print("Column names saved as models/model_columns.pkl")

# Save preprocessor separately
with open('models/preprocessor.pkl', 'wb') as f:
    pickle.dump(model.named_steps['preprocessor'], f)
    print("Preprocessor saved as models/preprocessor.pkl")

total_time = time.time() - start_time
print(f"\nTotal process completed in {total_time:.2f} seconds")
print("Model training process completed successfully!")