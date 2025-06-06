# Credit Card Fraud Detection

This project demonstrates a simple machine learning model to detect credit card fraud using a reduced and cleaned dataset. A Random Forest classifier is used to identify fraudulent transactions with performance evaluation and visualization.

## 📁 Files

- `fraud detection.py` - The main Python script for data loading, preprocessing, training, and evaluation.
- `creditcard_reduced.csv` - Reduced dataset used for training and testing the model.
- `README.md` - Project overview and setup guide.

## 🔍 Dataset

The dataset is a reduced version of a credit card transaction dataset. It contains anonymized features and a target variable `Class` where:

- `0` → Not Fraud
- `1` → Fraud

## ⚙️ Features

- Handles missing values in the dataset.
- Splits data into training and testing sets while preserving class distribution.
- Trains a Random Forest model for classification.
- Evaluates model performance with:
  - Confusion Matrix
  - Classification Report
  - Accuracy Score
- Visualizes:
  - Confusion Matrix using Seaborn heatmap
  - Feature Importances from the Random Forest model

## 🧠 Model

- **Algorithm**: Random Forest Classifier
- **Library**: `scikit-learn`
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

## 📊 Sample Output

- Accuracy: ~99%
- Feature importance graph
- Heatmap of confusion matrix

## 🛠️ Requirements

Install the required libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
