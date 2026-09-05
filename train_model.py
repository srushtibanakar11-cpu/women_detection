import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Filtered CSV Data
df = pd.read_csv('womens_safety_filtered_data.csv')

# Clean coordinates and handle missing values
df = df.dropna(subset=['Latitude', 'Longitude']).copy()

# 2. Synthetic Safety Metric Aggregation for ML Target
# Calculate localized danger score based on spatial crime clusters & incident counts
df['crime_weight'] = 1
if 'Victim_Count' in df.columns:
    df['crime_weight'] = df['Victim_Count'].fillna(1)

# Assign High Risk (1) vs Moderate/Safe (0) threshold based on incident density
threshold = df['crime_weight'].median()
df['risk_level'] = (df['crime_weight'] >= threshold).astype(int)

# 3. Select Features for ML Model
feature_cols = ['Latitude', 'Longitude']
if 'FIR_MONTH' in df.columns:
    df['FIR_MONTH'] = df['FIR_MONTH'].fillna(1)
    feature_cols.append('FIR_MONTH')
if 'Distance_from_PS' in df.columns:
    df['Distance_from_PS'] = df['Distance_from_PS'].fillna(df['Distance_from_PS'].mean())
    feature_cols.append('Distance_from_PS')

X = df[feature_cols]
y = df['risk_level']

# 4. Train Random Forest Classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# 5. Model Evaluation
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"--- Model Training Complete ---")
print(f"Accuracy: {acc * 100:.2f}%")

# Save model and feature list
with open('safety_model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'features': feature_cols}, f)
print("Saved model as 'safety_model.pkl' successfully!")