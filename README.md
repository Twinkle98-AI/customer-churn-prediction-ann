# customer-churn-prediction-ann
ANN-based customer churn prediction web app using TensorFlow and Streamlit.
📊 Customer Churn Prediction using Artificial Neural Network (ANN)

An end-to-end machine learning and deep learning project that predicts bank customer churn using an Artificial Neural Network (ANN) built with TensorFlow (Keras) and deployed as an interactive Streamlit web application.

🔍 Problem Statement (ATS Optimized)

Customer churn prediction is a critical business intelligence and predictive analytics problem in the banking and financial services industry.
This project applies supervised learning, binary classification, and deep learning techniques to identify customers who are likely to exit the bank.

🧠 Machine Learning & Deep Learning Approach

Model Type: Artificial Neural Network (ANN)

Learning Type: Supervised Learning

Problem Type: Binary Classification

Target Variable: Exited (0 = Retained, 1 = Churned)

Framework: TensorFlow (Keras API)

⚙️ Tech Stack & Tools (ATS Keywords)

Programming Language: Python 3.10

Deep Learning: TensorFlow, Keras

Machine Learning: Scikit-learn

Data Analysis: Pandas, NumPy

Data Preprocessing: Label Encoding, One-Hot Encoding, Feature Scaling

Model Evaluation: Accuracy Score, Confusion Matrix

Web App Framework: Streamlit

Version Control: Git, GitHub

📁 Dataset Information

Dataset Name: Churn_Modelling.csv

Domain: Banking / Finance Analytics

Features:

Credit Score

Geography

Gender

Age

Tenure

Account Balance

Number of Products

Credit Card Ownership

Active Member Status

Estimated Salary

🔄 Data Preprocessing Pipeline

Handling categorical variables using:

LabelEncoder (Gender)

OneHotEncoder (Geography)

Feature scaling using StandardScaler

Train-test split (80% training, 20% testing)

Data transformation using ColumnTransformer

🏗️ ANN Model Architecture
Input Layer
↓
Dense Layer (6 neurons, ReLU)
↓
Dense Layer (6 neurons, ReLU)
↓
Dense Layer (5 neurons, ReLU)
↓
Dense Layer (4 neurons, ReLU)
↓
Output Layer (1 neuron, Sigmoid)


Optimizer: Adam

Loss Function: Binary Crossentropy

Evaluation Metric: Accuracy

📊 Model Performance & Evaluation

Accuracy: ~85% (may vary per run)

Evaluation Metrics Used:

Accuracy Score

Confusion Matrix

Demonstrates strong performance on unseen test data creating a reliable predictive model.

🖥️ Streamlit Web Application Features

Interactive user interface for real-time predictions

Sidebar-based feature input

Probability-based churn prediction

Model accuracy display

Confusion matrix visualization

Deployment-ready structure

▶️ How to Run the Project
1️⃣ Clone Repository
git clone https://github.com/your-username/customer-churn-prediction-ann.git
cd customer-churn-prediction-ann

2️⃣ Create Virtual Environment
py -3.10 -m venv tf_env
tf_env\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run Streamlit App
streamlit run app.py

📌 Project Highlights (ATS Boost Section)

End-to-end machine learning pipeline

Deep learning model using TensorFlow ANN

Strong focus on data preprocessing and feature engineering

Model evaluation using classification metrics

Real-world banking analytics use case

Deployment-ready Streamlit web application

Clean, modular, and scalable codebase

🔮 Future Enhancements

Model persistence using .keras format

Hyperparameter tuning

ROC-AUC and Precision-Recall analysis

Cloud deployment (Streamlit Cloud / AWS)

Integration with real-time customer data

👤 Author

Haimabati Haripriya Sahu
Aspiring Data Scientist | Machine Learning & Deep Learning Enthusiast

⭐ Keywords for Recruiters (Hidden ATS Advantage)

Machine Learning, Deep Learning, TensorFlow, Keras, Artificial Neural Network,
Customer Churn Prediction, Binary Classification, Feature Engineering,
Scikit-learn, Streamlit, Data Science, Predictive Analytics,
Banking Analytics, Python
