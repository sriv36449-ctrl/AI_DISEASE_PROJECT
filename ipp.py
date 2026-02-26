import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="AI Healthcare System",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 AI-Based Disease Prediction & Consultation System")
st.markdown("---")

# -------------------- LOAD DATA --------------------
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

# -------------------- SIDEBAR --------------------
st.sidebar.header("📌 Project Information")
st.sidebar.write("Algorithm: Random Forest")
st.sidebar.write(f"Model Accuracy: {accuracy*100:.2f}%")
st.sidebar.write("Purpose: Educational Only")

# -------------------- PATIENT INPUT --------------------
st.subheader("👤 Enter Patient Details")

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

# -------------------- PREDICTION --------------------
if st.button("🧾 Generate Medical Report"):

    if name.strip() == "":
        st.warning("⚠️ Please enter patient name")
    else:
        input_data = [[age, gender, fever, cough, headache,
                       soreThroat, bodyPain, vomiting, rash, breathing]]

        prediction = model.predict(input_data)
        disease = le.inverse_transform(prediction)[0]
        confidence = model.predict_proba(input_data).max() * 100

        st.success("✅ Prediction Completed")

        st.subheader("📋 Medical Report")
        st.write(f"**Date:** {datetime.now().strftime('%d-%m-%Y %H:%M')}")
        st.write(f"**Patient Name:** {name}")
        st.write(f"**Age:** {age}")
        st.write(f"**Gender:** {gender_option}")

        st.write(f"**Predicted Disease:** {disease}")
        st.write(f"**Prediction Confidence:** {confidence:.2f}%")

        # Prescription
        prescription_dict = {
            "COVID-19": ["Isolation", "Paracetamol", "Steam inhalation", "Doctor consultation"],
            "Dengue": ["Hydration", "Paracetamol (No aspirin)", "Rest"],
            "Malaria": ["Antimalarial drugs", "Hydration", "Doctor supervision"],
            "Flu": ["Rest", "Fluids", "Paracetamol"],
            "Cold": ["Steam inhalation", "Warm fluids", "Rest"],
            "Typhoid": ["Antibiotics (Doctor prescribed)", "Hydration", "Soft diet"]
        }

        prescription = prescription_dict.get(disease, ["Consult Doctor"])

        st.subheader("💊 Prescription")
        for item in prescription:
            st.markdown(f"- {item}")

        st.warning("⚠️ This system is for educational purposes only.")

        # Store predicted disease for appointment module
        st.session_state.predicted_disease = disease

        # Download report
        report_text = f"""
AI Healthcare Medical Report
Date: {datetime.now().strftime("%d-%m-%Y %H:%M")}
Patient Name: {name}
Age: {age}
Gender: {gender_option}
Predicted Disease: {disease}
Prediction Confidence: {confidence:.2f}%
Prescription: {', '.join(prescription)}
Disclaimer: Educational use only.
"""

        st.download_button(
            label="⬇ Download Report",
            data=report_text,
            file_name="medical_report.txt",
            mime="text/plain"
        )

# -------------------- APPOINTMENT MODULE --------------------
if "predicted_disease" in st.session_state:

    st.markdown("---")
    st.subheader("🏥 Doctor Consultation & Appointment Booking")

    with st.form("appointment_form"):

        col1, col2 = st.columns(2)

        with col1:
            patient_name = st.text_input("Full Name")
            phone = st.text_input("Phone Number")
            email = st.text_input("Email Address")

        with col2:
            doctor_type = st.selectbox(
                "Select Doctor",
                ["General Physician", "Cardiologist",
                 "Dermatologist", "Neurologist", "Pediatrician"]
            )
            appointment_date = st.date_input("Select Appointment Date")
            appointment_time = st.time_input("Select Appointment Time")

        problem = st.text_area("Describe Your Problem")

        submit = st.form_submit_button("📅 Book Appointment")

    if submit:
        if patient_name.strip() == "" or phone.strip() == "":
            st.warning("⚠️ Please fill required details")
        else:
            st.success("✅ Appointment Booked Successfully!")

            st.subheader("🧾 Appointment Confirmation")
            st.write(f"**Patient Name:** {patient_name}")
            st.write(f"**Doctor:** {doctor_type}")
            st.write(f"**Date:** {appointment_date}")
            st.write(f"**Time:** {appointment_time}")
            st.write(f"**Problem:** {problem}")

            st.info("📌 Please arrive 10 minutes before your appointment.")
            st.balloons()
            st.success("🎉 Thank you for using our AI Healthcare System! Stay healthy 💙")
