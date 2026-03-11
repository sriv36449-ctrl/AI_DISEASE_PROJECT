import streamlit as st
from reportlab.pdfgen import canvas

st.set_page_config(page_title="Smart Healthcare System", page_icon="🏥", layout="wide")

# ---------------- STYLE ----------------

st.markdown("""
<style>

.stApp{
background-color:#d6f0ff;
}

.title{
text-align:center;
font-size:50px;
color:#003366;
font-weight:bold;
}

.subtitle{
text-align:center;
color:#003366;
font-size:20px;
margin-bottom:30px;
}

.card{
background:white;
padding:25px;
border-radius:15px;
box-shadow:0px 4px 15px rgba(0,0,0,0.1);
margin-bottom:25px;
}

.card-title{
font-size:24px;
color:#003366;
margin-bottom:10px;
font-weight:bold;
}

.stButton>button{
background:red;
color:white;
border-radius:10px;
height:40px;
font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------

st.markdown("<div class='title'>🏥 Smart AI Healthcare Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Advanced Digital Healthcare Platform</div>", unsafe_allow_html=True)

# ---------------- SYMPTOM CHECKER ----------------

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='card-title'>🩺 Symptom Checker</div>", unsafe_allow_html=True)

symptoms = [
"Fever","Cough","Headache","Fatigue","Stomach Pain","Sore Throat",
"Runny Nose","Chest Pain","Shortness of Breath","Dizziness",
"Nausea","Vomiting","Sneezing","Body Pain"
]

selected = st.multiselect("Select Symptoms", symptoms)

disease=""
prescription=""

# ---------------- PREDICT BUTTON ----------------

if st.button("Predict Disease"):

    if "Fever" in selected and "Cough" in selected:
        disease="Flu"
        prescription="Paracetamol, Vitamin C, Rest"

    elif "Runny Nose" in selected and "Sneezing" in selected:
        disease="Common Cold"
        prescription="Cetirizine, Steam inhalation"

    elif "Headache" in selected and "Nausea" in selected:
        disease="Migraine"
        prescription="Ibuprofen, Rest"

    elif "Stomach Pain" in selected and "Vomiting" in selected:
        disease="Food Poisoning"
        prescription="ORS, Antacid"

    elif "Chest Pain" in selected and "Shortness of Breath" in selected:
        disease="Possible Heart Disease"
        prescription="Immediate cardiologist consultation"

    if disease!="":
        st.success(f"Predicted Disease: {disease}")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PATIENT DETAILS ----------------

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='card-title'>👤 Patient Details</div>", unsafe_allow_html=True)

name = st.text_input("Patient Name")
age = st.number_input("Age",1,120)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- MEDICAL REPORT ----------------

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='card-title'>📄 Medical Report</div>", unsafe_allow_html=True)

if st.button("Generate Medical Report") and disease!="":

    pdf="medical_report.pdf"

    c=canvas.Canvas(pdf)

    c.drawString(100,750,"SMART HEALTHCARE MEDICAL REPORT")
    c.drawString(100,720,f"Patient Name: {name}")
    c.drawString(100,700,f"Age: {age}")
    c.drawString(100,680,f"Symptoms: {', '.join(selected)}")
    c.drawString(100,660,f"Predicted Disease: {disease}")
    c.drawString(100,640,f"Prescription: {prescription}")

    c.save()

    st.success("Medical Report Generated")

    with open(pdf,"rb") as f:
        st.download_button("Download Medical Report",f,"medical_report.pdf")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- DOCTOR APPOINTMENT ----------------

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='card-title'>📅 Doctor Appointment</div>", unsafe_allow_html=True)

doctor = st.selectbox(
"Choose Specialist Doctor",
["General Physician","Cardiologist","Neurologist","Dermatologist"]
)

date = st.date_input("Appointment Date")

if st.button("Book Appointment"):
    st.success(f"Appointment booked with {doctor} on {date}")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- VIDEO CONSULTATION ----------------

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='card-title'>🎥 Online Video Consultation</div>", unsafe_allow_html=True)

col1,col2,col3 = st.columns(3)

with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/3774/3774299.png", width=100)
    st.subheader("👨‍⚕️ Dr Smith")
    st.write("General Physician")
    st.success("Available")
    st.markdown("[Start Video Call](https://meet.google.com/)")

with col2:
    st.image("https://cdn-icons-png.flaticon.com/512/3774/3774361.png", width=100)
    st.subheader("👩‍⚕️ Dr Emily")
    st.write("Cardiologist")
    st.success("Available")
    st.markdown("[Start Video Call](https://zoom.us/)")

with col3:
    st.image("https://cdn-icons-png.flaticon.com/512/3774/3774296.png", width=100)
    st.subheader("👨‍⚕️ Dr John")
    st.write("Neurologist")
    st.warning("Offline")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- HOSPITAL INFO ----------------

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='card-title'>🏥 Hospital Information</div>", unsafe_allow_html=True)

st.write("Smart City Hospital")
st.write("Location: City Center")
st.write("Timing: 9 AM - 8 PM")
st.write("Emergency Contact: +123456789")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- HEALTH MESSAGE ----------------

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='card-title'>💚 Stay Healthy</div>", unsafe_allow_html=True)

st.success("Your health is important. Early diagnosis helps prevent serious diseases.")

st.markdown("</div>", unsafe_allow_html=True)
