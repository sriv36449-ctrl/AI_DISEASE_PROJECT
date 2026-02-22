import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

st.set_page_config(page_title="AI Healthcare System", layout="centered")

st.title("AI Healthcare Disease Prediction System")

# -----------------------
# Load Dataset
# -----------------------
@st.cache_data
def load_data():
    data = pd.read_csv("disease_data.csv")
    return data

data = load_data()

# Encode disease column
le = LabelEncoder()
data["Disease"] = le.fit_transform(data["Disease"])

X = data.drop("Disease", axis=1)
y = data["Disease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

st.subheader("Enter Patient Details")

name = st.text_input("Patient Name")
age = st.number_input("Age", 1, 100)
gender = st.selectbox("Gender (0 = Female, 1 = Male)", [0, 1])

fever = st.selectbox("Fever", [0, 1])
cough = st.selectbox("Cough", [0, 1])
headache = st.selectbox("Headache", [0, 1])
soreThroat = st.selectbox("Sore Throat", [0, 1])
bodyPain = st.selectbox("Body Pain", [0, 1])
vomiting = st.selectbox("Vomiting", [0, 1])
rash = st.selectbox("Rash", [0, 1])
breathing = st.selectbox("Breathing Problem", [0, 1])

# -----------------------
# Prediction
# -----------------------
if st.button("Generate Report"):

    input_data = [[age, gender, fever, cough, headache,
                   soreThroat, bodyPain, vomiting, rash, breathing]]

    prediction = model.predict(input_data)
    disease = le.inverse_transform(prediction)[0]

    # -----------------------
    # Prescription Logic
    # -----------------------
    prescription_dict = {
        "COVID-19": "Isolation, Paracetamol, Steam inhalation, Doctor consultation",
        "Dengue": "Hydration, Paracetamol (No aspirin), Rest",
        "Malaria": "Antimalarial drugs, Hydration, Doctor supervision",
        "Flu": "Rest, Fluids, Paracetamol",
        "Cold": "Steam inhalation, Warm fluids, Rest",
        "Typhoid": "Antibiotics (Doctor prescribed), Hydration, Soft diet"
    }

    prescription = prescription_dict.get(disease, "Consult Doctor")

    # -----------------------
    # Report Section
    # -----------------------
    st.success("Prediction Complete")

    st.subheader("Medical Report")

    st.write("Date:", datetime.now().strftime("%d-%m-%Y %H:%M"))
    st.write("Patient Name:", name)
    st.write("Age:", age)
    st.write("Predicted Disease:", disease)

    st.subheader("Prescription")

    st.info(prescription)

    # -----------------------
    # Download Report
    # -----------------------
    report_text = f"""
    AI Healthcare Medical Report
    -----------------------------
    Date: {datetime.now().strftime("%d-%m-%Y %H:%M")}

    Patient Name: {name}
    Age: {age}
    Predicted Disease: {disease}

    Prescription:
    {prescription}
    """

    st.download_button(
        label="Download Report",
        data=report_text,
        file_name="medical_report.txt",
        mime="text/plain"
    )