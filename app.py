import streamlit as st
import pandas as pd
import joblib
import time

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# ----------------------------
# Custom CSS for background and styling
# ----------------------------
st.markdown("""
<style>
/* Whole page background */
body {
    background-color: #f0f2f6;  /* light gray */
}

/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #e6f2ff; /* light blue */
}

/* Header fonts */
h1, h2, h3 {
    font-family: 'Arial Black', sans-serif;
}

/* Button styling */
.stButton>button {
    background-color: #ff4d4d;  /* red button */
    color: white;
    font-size:16px;
    border-radius: 12px;
    padding: 10px 20px;
}

/* Prediction result cards */
div[style*="background-color:#ffcccc"] {
    background-color: rgba(255, 0, 0, 0.2) !important;
}
div[style*="background-color:#ccffcc"] {
    background-color: rgba(0, 255, 0, 0.2) !important;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Load model objects
# ----------------------------
model = joblib.load("Disease.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# ----------------------------
# Header
# ----------------------------
st.markdown("""
<h1 style='text-align:center;color:red;'>❤️ Heart Disease Prediction System</h1>
<p style='text-align:center;'>AI based health risk analyzer</p>
<hr>
""", unsafe_allow_html=True)

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("ℹ️ About App")
st.sidebar.info("""
This app predicts heart disease risk using Machine Learning.

Developer: Muhammad Hammad  
Tech: Python, Scikit-learn, Streamlit
""")

# ----------------------------
# Input layout
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 40)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)

with col2:
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.slider("Max Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

st.markdown("<br>", unsafe_allow_html=True)

# ----------------------------
# Predict button
# ----------------------------
predict_btn = st.button("🔍 Predict Heart Risk", use_container_width=True)

# ----------------------------
# Prediction
# ----------------------------
if predict_btn:

    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    df_input = pd.DataFrame([raw_input])

    # Align columns
    for col in columns:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[columns]

    # Show spinner
    with st.spinner("Analyzing your health data..."):
        time.sleep(1.3)

        scaled_input = scaler.transform(df_input)
        prediction = model.predict(scaled_input)[0]

        # Show probability if available
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(scaled_input)[0][1] * 100
        else:
            probability = None

    # ----------------------------
    # Result UI
    # ----------------------------
    if prediction == 1:
        st.markdown(f"""
        <div style='background-color:#ffcccc;padding:25px;border-radius:12px'>
        <h2 style='color:red;'>⚠️ High Risk of Heart Disease</h2>
        <p>Please consult a cardiologist as soon as possible.</p>
        <b>Risk Probability:</b> {probability:.2f}% 
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background-color:#black;padding:25px;border-radius:12px'>
        <h2 style='color:green;'>✅ Low Risk of Heart Disease</h2>
        <p>You are currently safe. Maintain a healthy lifestyle.</p>
        <b>Risk Probability:</b> {probability:.2f}%
        </div>
        """, unsafe_allow_html=True)











#import streamlit as st
# import pandas as pd
# import joblib 

# model=joblib.load('Disease.pkl')
# scaler=joblib.load('scaler.pkl')
# column=joblib.load('columns.pkl')

# st.title('Heart Strok Prediction')
# st.markdown('Provide the following details to check your heart stroke risk:')

# # Collect user input
# age = st.slider("Age", 18, 100, 40)
# sex = st.selectbox("Sex", ["M", "F"])
# chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
# resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
# cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
# fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
# resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
# max_hr = st.slider("Max Heart Rate", 60, 220, 150)
# exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
# oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
# st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# # When Predict is clicked
# if st.button ('Predict'):
#     # Create a raw input dictionary
#     input={
#         'Age': age,
#         'RestingBP': resting_bp,
#         'Cholesterol': cholesterol,
#         'FastingBS': fasting_bs,
#         'MaxHR': max_hr,
#         'Oldpeak': oldpeak,
#         'Sex_' + sex: 1,              #this covert according to user input ex:if M then Sex_M:1
#         'ChestPainType_' + chest_pain: 1,
#         'RestingECG_' + resting_ecg: 1,
#         'ExerciseAngina_' + exercise_angina: 1,
#         'ST_Slope_' + st_slope: 1

#     }

# df=pd.DataFrame([input])
# for cols in column:
#     if cols not in df.columns:
#         df[cols]=0

# df=df[column]
# scaler_df=scaler.transform(df)
# prediction=model.predict(scaler_df)



# if prediction == 1:
#         st.error("⚠️ High Risk of Heart Disease")
# else:
#     st.success("✅ Low Risk of Heart Disease")



# import streamlit as st
# import pandas as pd
# import joblib

# # Load saved model, scaler, and expected columns
# model = joblib.load("Heart Disease.pkl")
# scaler = joblib.load("heart_scaler.pkl")
# expected_columns = joblib.load("heart_columns.pkl")

# st.title("Heart Stroke Prediction by akarsh")
# st.markdown("Provide the following details to check your heart stroke risk:")

# # Collect user input
# age = st.slider("Age", 18, 100, 40)
# sex = st.selectbox("Sex", ["M", "F"])
# chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
# resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
# cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
# fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
# resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
# max_hr = st.slider("Max Heart Rate", 60, 220, 150)
# exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
# oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
# st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# # When Predict is clicked
# if st.button("Predict"):

#     # Create a raw input dictionary
#     raw_input = {
#         'Age': age,
#         'RestingBP': resting_bp,
#         'Cholesterol': cholesterol,
#         'FastingBS': fasting_bs,
#         'MaxHR': max_hr,
#         'Oldpeak': oldpeak,
#         'Sex_' + sex: 1,
#         'ChestPainType_' + chest_pain: 1,
#         'RestingECG_' + resting_ecg: 1,
#         'ExerciseAngina_' + exercise_angina: 1,
#         'ST_Slope_' + st_slope: 1
#     }

#     # Create input dataframe
#     input_df = pd.DataFrame([raw_input])

#     # Fill in missing columns with 0s
#     for col in expected_columns:
#         if col not in input_df.columns:
#             input_df[col] = 0

#     # Reorder columns
#     input_df = input_df[expected_columns]

#     # Scale the input
#     scaled_input = scaler.fit_transform(input_df)

#     # Make prediction
#     prediction = model.predict(scaled_input)[0]

#     # Show result
#     if prediction == 1:
#         st.error("⚠️ High Risk of Heart Disease")
#     else:
#         st.success("✅ Low Risk of Heart Disease")