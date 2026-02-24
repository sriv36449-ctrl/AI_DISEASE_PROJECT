import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="AI Healthcare System",
    page_icon="??",
    layout="wide"
)

st.title("?? AI-Based Disease Prediction System")
st.markdown("### Predict common diseases based on patient symptoms using AI")
st.markdown("---")

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data
def load_data():
    return pd.read_csv("disease_data.csv")

data = load_data()

# Encode disease labels
le = LabelEncoder()
data["Disease"] = le.fit_transform(data["Disease"])

X = data.drop("Disease", axis=1)
y = data["Disease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

# -------------------------
# SIDEBAR INFO
# -------------------------
st.sidebar.header("?? Project Info")
st.sidebar.write("Algorithm: Random Forest Classifier")
st.sidebar.write(f"Model Accuracy: {accuracy*100:.2f}%")
st.sidebar.write("Purpose: Educational only")

# -------------------------
# INPUT SECTION
# -------------------------
st.subheader("?? Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    name = st.text_input("Patient Name")
    age = st.number_input("Age", 1, 100)
    gender_option = st.selectbox("Gender", ["Female", "Male"])
    gender = 1 if gender_option == "Male" else 0
    fever = 1 if st.selectbox("Fever", ["No", "Yes"]) == "Yes" else 0
    cough = 1 if st.selectbox("Cough", ["No", "Yes"]) == "Yes" else 0

with col2:
    headache = 1 if st.selectbox("Headache", ["No", "Yes"]) == "Yes" else 0
    soreThroat = 1 if st.selectbox("Sore Throat", ["No", "Yes"]) == "Yes" else 0
    bodyPain = 1 if st.selectbox("Body Pain", ["No", "Yes"]) == "Yes" else 0
    vomiting = 1 if st.selectbox("Vomiting", ["No", "Yes"]) == "Yes" else 0
    rash = 1 if st.selectbox("Rash", ["No", "Yes"]) == "Yes" else 0
    breathing = 1 if st.selectbox("Breathing Problem", ["No", "Yes"]) == "Yes" else 0

st.markdown("---")

# -------------------------
# PREDICTION
# -------------------------
if st.button("?? Generate Medical Report"):

    if name.strip() == "":
        st.warning("?? Please enter the patient name")
    else:
        # Prepare input
        input_data = [[age, gender, fever, cough, headache,
                       soreThroat, bodyPain, vomiting, rash, breathing]]

        # Predict disease
        prediction = model.predict(input_data)
        disease = le.inverse_transform(prediction)[0]

        # Prediction confidence
        confidence = model.predict_proba(input_data).max() * 100

        # Prescription dictionary
        prescription_dict = {
            "COVID-19": ["Isolation", "Paracetamol", "Steam inhalation", "Doctor consultation"],
            "Dengue": ["Hydration", "Paracetamol (No aspirin)", "Rest"],
            "Malaria": ["Antimalarial drugs", "Hydration", "Doctor supervision"],
            "Flu": ["Rest", "Fluids", "Paracetamol"],
            "Cold": ["Steam inhalation", "Warm fluids", "Rest"],
            "Typhoid": ["Antibiotics (Doctor prescribed)", "Hydration", "Soft diet"]
        }
        prescription = prescription_dict.get(disease, ["Consult Doctor"])

        # -------------------------
        # DISPLAY REPORT
        # -------------------------
        st.success("? Prediction Complete")

        st.subheader("?? Medical Report")
        st.write(f"**Date:** {datetime.now().strftime('%d-%m-%Y %H:%M')}")
        st.write(f"**Patient Name:** {name}")
        st.write(f"**Age:** {age}")
        st.write(f"**Gender:** {gender_option}")

        st.write("**Symptoms:**")
        st.write(f"Fever: {'Yes' if fever else 'No'} | Cough: {'Yes' if cough else 'No'} | Headache: {'Yes' if headache else 'No'}")
        st.write(f"Sore Throat: {'Yes' if soreThroat else 'No'} | Body Pain: {'Yes' if bodyPain else 'No'} | Vomiting: {'Yes' if vomiting else 'No'}")
        st.write(f"Rash: {'Yes' if rash else 'No'} | Breathing Problem: {'Yes' if breathing else 'No'}")

        st.write(f"**Predicted Disease:** {disease}")
        st.write(f"**Prediction Confidence:** {confidence:.2f}%")

        st.subheader("?? Prescription")
        for item in prescription:
            st.markdown(f"- {item}")

        st.warning(
            "?? This prediction is for educational purposes only and not for real-life medical diagnosis. Please consult a qualified doctor."
        )

        # -------------------------
        # DOWNLOADABLE REPORT
        # -------------------------
        report_text = f"""
AI Healthcare Medical Report
-----------------------------
Date: {datetime.now().strftime("%d-%m-%Y %H:%M")}

Patient Name: {name}
Age: {age}
Gender: {gender_option}

Symptoms:
Fever: {'Yes' if fever else 'No'}
Cough: {'Yes' if cough else 'No'}
Headache: {'Yes' if headache else 'No'}
Sore Throat: {'Yes' if soreThroat else 'No'}
Body Pain: {'Yes' if bodyPain else 'No'}
Vomiting: {'Yes' if vomiting else 'No'}
Rash: {'Yes' if rash else 'No'}
Breathing Problem: {'Yes' if breathing else 'No'}

Predicted Disease: {disease}
Prediction Confidence: {confidence:.2f}%

Prescription:
{', '.join(prescription)}

Disclaimer:
This prediction is for educational purposes only. Always consult a qualified doctor.
"""

        st.download_button(
            label="? Download Report",
            data=report_text,
            file_name="medical_report.txt",
            mime="text/plain"
        )

        # -------------------------
        # SATISFYING END MESSAGE
        # -------------------------
        st.balloons()
        st.success("?? Report Generated Successfully! Thank you for using the AI Healthcare System. Stay safe! ?")