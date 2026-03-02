import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="AI Smart Healthcare System",
    page_icon="🩺",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stApp {background-color: #e6f2ff;}
    h1 {color: #003366; text-align: center;}
    h2 {color: #0059b3;}
    .stButton>button {background-color: #0059b3; color: white; border-radius:10px; height:3em;}
    </style>
    """, unsafe_allow_html=True
)

st.title("🩺 AI Smart Healthcare System")
st.markdown("---")

# -------------------- SAMPLE DATA --------------------
# This creates the CSV internally, no Excel needed
data_dict = {
    "Age":[25,30,40,22,55],
    "Gender":[1,0,1,0,1],
    "Fever":[1,0,1,0,1],
    "Cough":[1,1,0,0,1],
    "Headache":[0,1,1,0,1],
    "SoreThroat":[0,1,0,0,1],
    "BodyPain":[1,0,1,1,1],
    "Vomiting":[0,0,1,0,1],
    "Rash":[0,0,0,0,1],
    "Breathing":[1,0,1,0,1],
    "Disease":["Flu","Cold","Dengue","Cold","Malaria"]
}
df = pd.DataFrame(data_dict)

# -------------------- ENCODE AND MODEL --------------------
le = LabelEncoder()
df["Disease"] = le.fit_transform(df["Disease"])
X = df.drop("Disease", axis=1)
y = df["Disease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)

# -------------------- SIDEBAR --------------------
st.sidebar.header("📌 Project Info")
st.sidebar.write(f"Algorithm: Random Forest")
st.sidebar.write(f"Model Accuracy: {accuracy*100:.2f}%")
st.sidebar.write("Purpose: Educational Only")

# -------------------- STEP 1: PATIENT INPUT --------------------
st.subheader("👤 Step 1: Enter Patient Details")
col1, col2 = st.columns(2)

with col1:
    name = st.text_input("Patient Name")
    age = st.number_input("Age", 1, 100)
    gender_option = st.selectbox("Gender", ["Female", "Male"])
    gender = 1 if gender_option=="Male" else 0
    fever = 1 if st.selectbox("Fever", ["No","Yes"])=="Yes" else 0
    cough = 1 if st.selectbox("Cough", ["No","Yes"])=="Yes" else 0

with col2:
    headache = 1 if st.selectbox("Headache", ["No","Yes"])=="Yes" else 0
    soreThroat = 1 if st.selectbox("Sore Throat", ["No","Yes"])=="Yes" else 0
    bodyPain = 1 if st.selectbox("Body Pain", ["No","Yes"])=="Yes" else 0
    vomiting = 1 if st.selectbox("Vomiting", ["No","Yes"])=="Yes" else 0
    rash = 1 if st.selectbox("Rash", ["No","Yes"])=="Yes" else 0
    breathing = 1 if st.selectbox("Breathing Problem", ["No","Yes"])=="Yes" else 0

st.markdown("---")

# -------------------- STEP 2: DISEASE PREDICTION --------------------
if st.button("🧾 Step 2: Generate Medical Report"):

    if name.strip() == "":
        st.warning("⚠️ Please enter patient name")
    else:
        input_data = [[age, gender, fever, cough, headache, soreThroat, bodyPain, vomiting, rash, breathing]]
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
            "COVID-19": ["Isolation","Paracetamol","Steam inhalation","Doctor consultation"],
            "Dengue": ["Hydration","Paracetamol (No aspirin)","Rest"],
            "Malaria": ["Antimalarial drugs","Hydration","Doctor supervision"],
            "Flu": ["Rest","Fluids","Paracetamol"],
            "Cold": ["Steam inhalation","Warm fluids","Rest"],
            "Typhoid": ["Antibiotics (Doctor prescribed)","Hydration","Soft diet"]
        }
        prescription = prescription_dict.get(disease, ["Consult Doctor"])
        st.subheader("💊 Prescription")
        for item in prescription:
            st.markdown(f"- {item}")

        st.warning("⚠️ This system is for educational purposes only.")

        # Store for appointment
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
        st.download_button("⬇ Download Report", data=report_text, file_name="medical_report.txt", mime="text/plain")

st.markdown("---")

# -------------------- STEP 3: APPOINTMENT --------------------
if "predicted_disease" in st.session_state:
    st.subheader("🏥 Step 3: Doctor Appointment Booking")

    with st.form("appointment_form"):
        col1, col2 = st.columns(2)
        with col1:
            patient_name = st.text_input("Full Name")
            phone = st.text_input("Phone Number")
            email = st.text_input("Email Address")
        with col2:
            doctor_type = st.selectbox("Select Doctor", ["General Physician","Cardiologist","Dermatologist","Neurologist","Pediatrician"])
            appointment_date = st.date_input("Select Appointment Date")
            appointment_time = st.time_input("Select Appointment Time")
        problem = st.text_area("Describe Your Problem")
        submit = st.form_submit_button("📅 Book Appointment")

    if submit:
        if patient_name.strip()=="" or phone.strip()=="":
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

st.markdown("---")

# -------------------- STEP 4: FINAL QUOTES --------------------
st.subheader("💙 Stay Healthy, Stay Safe!")
st.markdown(
"""
> "Prevention is better than cure."  
> "Your health is your wealth."  
> "Early detection saves lives."  
"""
)
