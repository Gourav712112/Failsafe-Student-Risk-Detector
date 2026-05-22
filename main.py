from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import shap

app = FastAPI(title="Failsafe Early Warning System API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# सेव की गई चीज़ों को लोड करना
try:
    model = joblib.load('failsafe_xgb_model.pkl')
    encoders = joblib.load('label_encoders_dict.pkl')
    baseline_row = joblib.load('baseline_row.pkl')
    explainer = shap.TreeExplainer(model)
    print("All Real-Dataset Model files loaded successfully!")
except Exception as e:
    print(f"Error loading model files: {e}")

class StudentData(BaseModel):
    sex: str          # 'M' या 'F'
    studytime: int    # 1 से 4
    failures: int     # 0 से 3
    absences: int     # 0 से 93
    G1: int           # 0 से 20
    G2: int           # 0 से 20

@app.get("/")
def home():
    return {"message": "Welcome to Failsafe API. Go to /docs"}

@app.post("/predict")
def predict_student_risk(student: StudentData):
    try:
        # 1. बेसलाइन रो की कॉपी बनाएं (ताकि सभी 32 फीचर्स मौजूद रहें)
        full_features = baseline_row.copy()
        
        # 2. यूजर द्वारा फ्रंटएंड से भेजे गए मुख्य फीचर्स को अपडेट करें
        full_features['sex'] = student.sex
        full_features['studytime'] = student.studytime
        full_features['failures'] = student.failures
        full_features['absences'] = student.absences
        full_features['G1'] = student.G1
        full_features['G2'] = student.G2

        input_data = pd.DataFrame([full_features])

        # 3. सभी कैटेगोरिकल फीचर्स को उनके संबंधित एनकोडर से कन्वर्ट करें
        for col, le in encoders.items():
            if col in input_data.columns:
                # अगर नया इनपुट स्ट्रिंग है, तो उसे एनकोड करें
                if isinstance(input_data[col].iloc[0], str):
                    input_data[col] = le.transform(input_data[col])

        # यह सुनिश्चित करने के लिए कि कॉलम्स का ऑर्डर बिल्कुल ट्रेनिंग जैसा ही हो
        feature_order = model.get_booster().feature_names
        input_data = input_data[feature_order]

        # 4. ML Prediction
        risk_prob = float(model.predict_proba(input_data)[0][1])
        risk_percentage = round(risk_prob * 100, 2)
        
        is_at_risk = 1 if (risk_percentage >= 50.0 or student.failures > 0) else 0

        # 5. SHAP Values - केवल उन्हीं 6 मुख्य फीचर्स का इम्पैक्ट दिखाएंगे जो यूजर को समझ आएं
        shap_values = explainer(input_data)
        user_visible_features = ['sex', 'studytime', 'failures', 'absences', 'G1', 'G2']
        
        feature_importance = {}
        for f in user_visible_features:
            f_idx = feature_order.index(f)
            feature_importance[f] = float(shap_values.values[0][f_idx])

        # 6. AUTOMATED INTERVENTION LOGIC
        interventions = []
        if is_at_risk == 1:
            interventions.append("High Risk Alert: Immediate faculty intervention required.")
            if student.failures > 0:
                interventions.append("Arrange mandatory remedial classes for clearing backlog concepts.")
            if student.studytime < 2:
                interventions.append("Schedule a mentorship session to improve weekly self-study habits.")
            if student.absences > 10:
                interventions.append("Issue an attendance warning and refer the student to a student counselor.")
            if student.G2 < 10:
                interventions.append("Provide targeted question banks and weekly mock tests based on G2 weak areas.")
            if len(interventions) == 1:
                interventions.append("Formulate a customized regular peer-tutoring and monitoring plan.")
        else:
            interventions.append("Student performance is currently stable. Continue monitoring routine academic progress.")

        return {
            "status": "success",
            "at_risk_prediction": is_at_risk, 
            "risk_percentage": risk_percentage,
            "automated_intervention_plan": interventions,
            "explainable_ai_shap": feature_importance
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))