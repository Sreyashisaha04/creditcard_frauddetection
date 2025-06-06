# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Display settings
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# Load the dataset
try:
    data = pd.read_csv('creditcard_reduced.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: The file 'creditcard.csv' was not found.")
    exit()

# Explore the dataset
print("\nFirst 5 rows of the dataset:")
print(data.head())

print("\nDataset Information:")
print(data.info())

print("\nClass distribution before dropping NaNs:")
print(data['Class'].value_counts(dropna=False))

# ✅ Drop rows with missing values in 'Class' column
data = data.dropna(subset=['Class'])

# Separate features and target variable
X = data.drop('Class', axis=1)  # Features
y = data['Class']               # Target variable

# Split into training and testing sets (Stratify to maintain class distribution)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Initialize the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy Score: {:.4f}".format(accuracy))

# Visualize the Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Fraud', 'Fraud'],
            yticklabels=['Not Fraud', 'Fraud'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# Feature Importance
importances = model.feature_importances_
feature_names = X.columns
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot Feature Importances
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
plt.title('Feature Importances from Random Forest')
plt.tight_layout()
plt.show()

print("\nTop 10 Important Features:\n")
print(feature_importance.head(10))
