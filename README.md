AI-Based Transaction Fraud Detection

Overview

AI-Based Transaction Fraud Detection is a system that identifies fraudulent transactions using a Random Forest model. It processes transaction details and classifies them as fraudulent or legitimate in real time. The project is designed for accuracy and ease of use.

Features

Real-Time Fraud Detection – Instantly classifies transactions.

Machine Learning Model – Uses Random Forest for analysis.

Minimalist Project Structure – Simple and efficient design.

Graphical Representation – Provides insights into transaction patterns.

Tech Stack

Backend: Python, Flask

Machine Learning: Scikit-Learn (Random Forest)

Dataset: PaySim (fraud_data.csv)

Project Structure

AI-Based-Transaction-Fraud-Detection/  
│── model.py       # Trains and saves the fraud detection model  
│── app.py         # Backend for processing transactions  
│── templates/     # Contains the frontend files  
│── static/        # Stores static resources  
│── fraud_data.csv # Dataset for training the model  
│── requirements.txt # Dependencies  
│── README.md      # Project documentation  

How to Run

Clone the repository:

git clone https://github.com/yourusername/AI-Based-Transaction-Fraud-Detection.git
cd AI-Based-Transaction-Fraud-Detection

Install dependencies:

pip install -r requirements.txt

Train the model (if needed):

python model.py

Run the application:

python app.py

Open in Browser:Visit http://127.0.0.1:5000/ to use the fraud detection system.

Future Improvements

Enhance model accuracy with advanced techniques.

Implement real-time transaction streaming.

Improve user experience with a refined interface.
