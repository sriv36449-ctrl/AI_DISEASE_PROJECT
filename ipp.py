import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# -----------------------------------
# Page Configuration (MUST BE FIRST)
# -----------------------------------
st.set_page_config(
    page_title="AI Disease Prediction System",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 AI-Based Disease Prediction System")
st.markdown("### Predict diseases using Machine Learning")
st.markdown("---")

# -----------------------------------
# Sidebar Section
# -----------------------------------
st.sidebar.header("📌 Project Information")

st.sidebar.markdown("""
**Project Title:** AI Healthcare Disease Prediction  
**Technology Used:** Python, Streamlit, Scikit-learn  
**Algorithm Used:** Random Forest Classifier  
**Developed By:** Vaishnavy,srimathi,sri vaishnavi 
""")

# -----------------------------------
# Load Dataset
# -----------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("disease_data.csv")

data = load_data()

# Encode disease column
le = LabelEncoder()
data["Disease"] = le.fit_transform(data["Disease"])

X = data.drop("Disease", axis=1)
y = data["Disease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

st.sidebar.success(f"Model Accuracy: {accuracy*100:.2f}%")

# -----------------------------------
# Patient Input Section
# -----------------------------------
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# -----------------------------------
# Page Configuration (MUST BE FIRST)
# -----------------------------------
st.set_page_config(
    page_title="AI Disease Prediction System",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 AI-Based Disease Prediction System")
st.markdown("### Predict diseases using Machine Learning")
st.markdown("---")

# -----------------------------------
# Sidebar Section
# -----------------------------------
st.sidebar.header("📌 Project Information")

st.sidebar.markdown("""
**Project Title:** AI Healthcare Disease Prediction  
**Technology Used:** Python, Streamlit, Scikit-learn  
**Algorithm Used:** Random Forest Classifier  
**Developed By:** Vaishnavy  
""")

# -----------------------------------
# Load Dataset
# -----------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("disease_data.csv")

data = load_data()

# Encode disease column
le = LabelEncoder()
data["Disease"] = le.fit_transform(data["Disease"])

X = data.drop("Disease", axis=1)
y = data["Disease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

st.sidebar.success(f"Model Accuracy: {accuracy*100:.2f}%")

# -----------------------------------
# Patient Input Section
# -----------------------------------
st.subheader("👤 Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    name = st.text_input("Patient Name")
    age = st.number_input("Age", 1, 100)
    gender = st.selectbox("Gender (0 = Female, 1 = Male)", [0, 1])
    fever = st.selectbox("Fever", [0, 1])
    cough = st.selectbox("Cough", [0, 1])

with col2:
    headache = st.selectbox("Headache", [0, 1])
    soreThroat = st.selectbox("Sore Throat", [0, 1])
    bodyPain = st.selectbox("Body Pain", [0, 1])
    vomiting = st.selectbox("Vomiting", [0, 1])
    rash = st.selectbox("Rash", [0, 1])
    breathing = st.selectbox("Breathing Problem", [0, 1])

st.markdown("---")

# -----------------------------------
# Prediction Section
# -----------------------------------
if st.button("🧾 Generate Medical Report"):

    input_data = [[age, gender, fever, cough, headache,
                   soreThroat, bodyPain, vomiting, rash, breathing]]

    prediction = model.predict(input_data)
    disease = le.inverse_transform(prediction)[0]

    # Prescription dictionary
    prescription_dict = {
        "COVID-19": "Isolation, Paracetamol, Steam inhalation, Doctor consultation",
        "Dengue": "Hydration, Paracetamol (No aspirin), Rest",
        "Malaria": "Antimalarial drugs, Hydration, Doctor supervision",
        "Flu": "Rest, Fluids, Paracetamol",
        "Cold": "Steam inhalation, Warm fluids, Rest",
        "Typhoid": "Antibiotics (Doctor prescribed), Hydration, Soft diet"
    }

    prescription = prescription_dict.get(disease, "Consult Doctor")

    st.success("Prediction Complete ✅")

    # -----------------------------------
    # Report Section
    # -----------------------------------
    st.subheader("📋 Medical Report")

    st.write("Date:", datetime.now().strftime("%d-%m-%Y %H:%M"))
    st.write("Patient Name:", name)
    st.write("Age:", age)
    st.write("Predicted Disease:", disease)

    st.subheader("💊 Prescription")
    st.info(prescription)

    # -----------------------------------
    # Download Report
    # -----------------------------------
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
        label="⬇ Download Report",
        data=report_text,
        file_name="medical_report.txt",
        mime="text/plain"
    )

# -----------------------------------
# About Section
# -----------------------------------
st.markdown("---")
st.subheader("📖 About This Project")

st.write("""
This project uses Machine Learning (Random Forest Algorithm) 
to predict diseases based on patient symptoms.

It helps in early detection and provides quick medical guidance.
""")

st.subheader("🚀 Future Scope")

st.write("""
- Add more diseases
- Improve dataset size
- Connect with hospital database
- Convert into mobile application
""")

# -----------------------------------
# Prediction Section
# -----------------------------------
if st.button("🧾 Generate Medical Report"):

    input_data = [[age, gender, fever, cough, headache,
                   soreThroat, bodyPain, vomiting, rash, breathing]]

    prediction = model.predict(input_data)
    disease = le.inverse_transform(prediction)[0]

    # Prescription dictionary
    prescription_dict = {
        "COVID-19": "Isolation, Paracetamol, Steam inhalation, Doctor consultation",
        "Dengue": "Hydration, Paracetamol (No aspirin), Rest",
        "Malaria": "Antimalarial drugs, Hydration, Doctor supervision",
        "Flu": "Rest, Fluids, Paracetamol",
        "Cold": "Steam inhalation, Warm fluids, Rest",
        "Typhoid": "Antibiotics (Doctor prescribed), Hydration, Soft diet"
    }

    prescription = prescription_dict.get(disease, "Consult Doctor")

    st.success("Prediction Complete ✅")

    # -----------------------------------
    # Report Section
    # -----------------------------------
    st.subheader("📋 Medical Report")

    st.write("Date:", datetime.now().strftime("%d-%m-%Y %H:%M"))
    st.write("Patient Name:", name)
    st.write("Age:", age)
    st.write("Predicted Disease:", disease)

    st.subheader("💊 Prescription")
    st.info(prescription)

    # -----------------------------------
    # Download Report
    # -----------------------------------
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
        label="⬇ Download Report",
        data=report_text,
        file_name="medical_report.txt",
        mime="text/plain"
    )

# -----------------------------------
# About Section
# -----------------------------------
st.markdown("---")
st.subheader("📖 About This Project")

st.write("""
This project uses Machine Learning (Random Forest Algorithm) 
to predict diseases based on patient symptoms.

It helps in early detection and provides quick medical guidance.
""")

st.subheader("🚀 Future Scope")

st.write("""
- Add more diseases
- Improve dataset size
- Connect with hospital database
- Convert into mobile application
""")
