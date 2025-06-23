# loadDigits_Ann
🧾 Overview:
This project classifies handwritten digits (0–9) using an Artificial Neural Network (ANN). It was developed as part of an MLOps course project and demonstrates the practical use of neural networks for basic image recognition tasks. The trained model is deployed via Streamlit for real-time predictions.

📌 Features:
Multi-class classification (digits 0–9)

Trained on a structured dataset (similar to MNIST)

User uploads an image and receives instant prediction

Lightweight web app using Streamlit

Includes experiment tracking with MLflow

🛠 Tech Stack:
Python

TensorFlow / Keras (ANN)

NumPy, Pandas, Matplotlib

MLflow

Streamlit

Pickle for model serialization

📁 Project Structure:
├── digit_dataset/                  # Input digit images
├── digit_ann_model.h5             # Trained ANN model
├── streamlit_digit_app.py         # Web app script
├── mlflow_logs/                   # Experiment logs
├── requirements.txt               # Dependencies
└── README.md                      # Documentation
🔬 Model Summary:
Input Layer → Dense Hidden Layers → Output Layer (10 classes)

Optimizer: Adam

Loss: Categorical Crossentropy

Evaluation metrics: Accuracy, Confusion Matrix, Precision/Recall

🚀 How to Run:
pip install -r requirements.txt
streamlit run streamlit_digit_app.py
🎯 Goal:
To demonstrate practical application of Artificial Neural Networks for pattern recognition, with a real-time, user-friendly interface.

🙋‍♀️ Creator:
Developed by Minahil Mehdi
Certified in AI, Data Science & Blockchain (NAVTTC)
