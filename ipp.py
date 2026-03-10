import streamlit as st
from reportlab.pdfgen import canvas

st.set_page_config(page_title="Smart Healthcare System", page_icon="🏥", layout="wide")

# ---------------- UI DESIGN ----------------

st.markdown("""
<style>

.stApp{
background: linear-gradient(135deg,#e0f2ff,#b9e6ff,#dff4ff);
color:#1e3a8a;
}

.title{
text-align:center;
font-size:50px;
color:#1e3a8a;
font-weight:bold;
}

.subtitle{
text-align:center;
color:#1e3a8a;
font-size:20px;
margin-bottom:40px;
}

.card{
background:white;
padding:25px;
border-radius:15px;
box-shadow:0px 4px 20px rgba(0,0,0,0.15);
margin-bottom:25px;
animation: fade 1s;
}

@keyframes fade{
0%{opacity:0}
100%{opacity:1}
}

.card-title{
font-size:24px;
color:#1e3a8a;
margin-bottom:10px;
font-weight:bold;
}

.stButton>button{
background:#ef4444;
color:white;
border-radius:10px;
height:40px;
font-weight:bold;
}

strong{
color:#ef4444;
}

</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🏥 Smart AI Healthcare Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Advanced Digital Hospital Platform</div>", unsafe_allow_html=True)

# ---------------- SYMPTOMS ----------------

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='card-title'>🩺 Symptom Checker</div>", unsafe_allow_html=True)

symptoms = [
"Fever","Cough","Headache","Fatigue","Stomach Pain","Sore Throat","Runny Nose",
"Chest Pain","Shortness of Breath","Dizziness","Nausea","Vomiting","Body Pain",
"Chills","Loss of Appetite","Sweating","Muscle Pain","Joint Pain","Back Pain",
"Blurred Vision","Ear Pain","Skin Rash","Diarrhea","Constipation","Heartburn",
"Weight Loss","Weight Gain","Insomnia","Anxiety","Depression","Palpitations",
"Swelling","Frequent Urination","Burning Urination","Dry Mouth","Hair Loss",
"Itching","Red Eyes","Sneezing","Numbness"
]

selected = st.multiselect("Select Symptoms", symptoms)

disease="General Checkup Recommended"
risk="Low"

if st.button("Predict Disease"):

    if "Fever" in selected and "Cough" in selected:
        disease="Flu"

    elif "Runny Nose" in selected and "Sneezing" in selected:
        disease="Common Cold"

    elif "Headache" in selected and "Nausea" in selected:
        disease="Migraine"

    elif "Stomach Pain" in selected and "Heartburn" in selected:
        disease="Gastric Infection"

    elif "Vomiting" in selected and "Diarrhea" in selected:
        disease="Food Poisoning"

    elif "Frequent Urination" in selected and "Weight Loss" in selected:
        disease="Possible Diabetes"

    elif "Chest Pain" in selected and "Shortness of Breath" in selected:
        disease="Possible Heart Disease"

    elif "Skin Rash" in selected and "Itching" in selected:
        disease="Skin Allergy"

    elif "Red Eyes" in selected and "Blurred Vision" in selected:
        disease="Eye Infection"

    elif "Burning Urination" in selected and "Frequent Urination" in selected:
        disease="Urinary Tract Infection"

    symptom_count=len(selected)

    if symptom_count<=3:
        risk="Low"
    elif symptom_count<=6:
        risk="Medium"
    else:
        risk="High"

    st.warning(f"Possible Condition: {disease}")
    st.info(f"Risk Level: {risk}")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- MEDICAL REPORT ----------------

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='card-title'>📄 Medical Report</div>", unsafe_allow_html=True)

name=st.text_input("Patient Name")
age=st.number_input("Age",1,120)

if st.button("Generate Medical Report"):

    st.success("Medical Report Generated")

    st.write("Patient Name:",name)
    st.write("Age:",age)
    st.write("Symptoms:",selected)
    st.write("Predicted Disease:",disease)
    st.write("Risk Level:",risk)

    pdf="medical_report.pdf"

    c=canvas.Canvas(pdf)

    c.drawString(100,750,"SMART HEALTHCARE REPORT")
    c.drawString(100,700,f"Name: {name}")
    c.drawString(100,680,f"Age: {age}")
    c.drawString(100,660,f"Symptoms: {', '.join(selected)}")
    c.drawString(100,640,f"Disease: {disease}")
    c.drawString(100,620,f"Risk Level: {risk}")

    c.save()

    with open(pdf,"rb") as f:
        st.download_button("Download Medical Report",f,"medical_report.pdf")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PRESCRIPTION ----------------

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='card-title'>💊 Prescription</div>", unsafe_allow_html=True)

condition=st.selectbox("Select Condition",
["Flu","Cold","Migraine","Gastric Infection","Food Poisoning"])

if condition=="Flu":
    st.info("Tablets: Paracetamol, Vitamin C")

elif condition=="Cold":
    st.info("Tablets: Cetirizine")

elif condition=="Migraine":
    st.info("Tablets: Ibuprofen")

elif condition=="Gastric Infection":
    st.info("Tablets: Antacid")

elif condition=="Food Poisoning":
    st.info("Tablets: Oral Rehydration Salts")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- APPOINTMENT ----------------

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='card-title'>📅 Doctor Appointment</div>", unsafe_allow_html=True)

doctor=st.selectbox("Choose Specialist Doctor",
["General Physician","Cardiologist","Neurologist","Pediatrician",
"Gastroenterologist","Dermatologist","Orthopedic","Psychiatrist"])

date=st.date_input("Appointment Date")

if st.button("Book Appointment"):
    st.success(f"Appointment booked with {doctor} on {date}")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- VIDEO CONSULTATION ----------------

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='card-title'>🎥 Online Video Consultation</div>", unsafe_allow_html=True)

col1,col2,col3=st.columns(3)

with col1:
    st.subheader("👨‍⚕️ Dr Smith")
    st.write("General Physician")
    st.success("Online")
    st.markdown("[Start Video Call](https://meet.google.com/)")

with col2:
    st.subheader("👩‍⚕️ Dr Emily")
    st.write("Cardiologist")
    st.success("Online")
    st.markdown("[Start Video Call](https://zoom.us/)")

with col3:
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

# ---------------- POSITIVE MESSAGE ----------------

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='card-title'>💚 Stay Healthy</div>", unsafe_allow_html=True)

st.success("Your health is important. Early diagnosis helps prevent serious diseases. Maintain a healthy lifestyle and consult doctors when needed.")

st.markdown("</div>", unsafe_allow_html=True)