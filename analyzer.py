import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# --- 1. DATA LOADING & CLEANING ---
# We'll create a dictionary first, then save to CSV as per your project requirements
data = {
    'Study_Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Exam_Score': [35, 42, 48, 55, 60, 68, 75, 82, 89, 95]
}
df = pd.DataFrame(data)
df.to_csv('student_data.csv', index=False)

print("Step 1: Dataset loaded and cleaned.")
print(df.head()) # Shows the first few rows

# --- 2. SIMPLE STATISTICS ---
print("\nStep 2: Simple Statistics")
print(df.describe())

# --- 3. VISUALIZATION ---
plt.scatter(df.Study_Hours, df.Exam_Score, color='blue')
plt.title('Study Hours vs Exam Score')
plt.xlabel('Hours Studied')
plt.ylabel('Score')
plt.show()

# --- 4. MACHINE LEARNING MODEL (Linear Regression) ---
X = df[['Study_Hours']] # Features
y = df['Exam_Score']    # Target

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction & Evaluation
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)

print(f"\nStep 3: ML Model Training Complete.")
print(f"Mean Squared Error: {mse:.2f}")

# Example Prediction
new_hour = [[12]]
predicted_score = model.predict(new_hour)
print(f"Prediction for 12 hours of study: {predicted_score[0]:.2f}%")