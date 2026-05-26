import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

print("--- Failsafe ML Pipeline Started ---")

# 1. REAL DATASET LOADING
try:
    
    df = pd.read_csv('student-mat.csv')
    print("1. Real dataset 'student-mat.csv' loaded successfully.")
except FileNotFoundError:
    print("Error: 'student-mat.csv' file not found!")
    exit()


df.columns = df.columns.str.replace('"', '').str.strip()

# 2. TARGET VARIABLE CREATION
if 'G3' in df.columns:
    df['At_Risk'] = (df['G3'] < 10).astype(int)
    df = df.drop(columns=['G3'])
else:
    print(f"Error: 'G3' Column not find: {list(df.columns)}")
    exit()

# 3. AUTOMATIC CATEGORICAL ENCODING
categorical_cols = df.select_dtypes(include=['object']).columns
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    if df[col].dtype == 'object':
        df[col] = df[col].astype(str).str.replace('"', '').str.strip()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Baseline Row Template for API
baseline_row = df.drop(columns=['At_Risk']).median(numeric_only=True).to_dict()
for col in df.drop(columns=['At_Risk']).columns:
    if col not in baseline_row:
        baseline_row[col] = int(df[col].mode()[0])

print("3. Categorical columns encoded successfully.")

# 4. TRAIN-TEST SPLIT
X = df.drop(columns=['At_Risk'])
y = df['At_Risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. MODEL TRAINING
model = xgb.XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42, eval_metric='logloss')
model.fit(X_train, y_train)
print("5. XGBoost Model trained successfully.")

# 6. SAVE ARTIFACTS
joblib.dump(model, 'failsafe_xgb_model.pkl')
joblib.dump(encoders, 'label_encoders_dict.pkl')
joblib.dump(baseline_row, 'baseline_row.pkl')
print("8. All artifacts saved successfully.")
print("--- Pipeline Completed Successfully ---")