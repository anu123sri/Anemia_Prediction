import streamlit as st
import pandas as pd
import pickle
import os

# Get the absolute path to the directory of the current script
current_script_directory = os.path.dirname(os.path.abspath(__file__))

# Load the Random Forest model
model_path = os.path.join(current_script_directory, 'random_forest_model.pkl')
with open(model_path, 'rb') as model_file:
    random_forest_model = pickle.load(model_file)

# Streamlit App
st.title("Anemia Prediction ")

# Add a background image
st.markdown(
    """
    <style>
        body {
            background-image: url("https://example.com/your-background-image.jpg");
            background-size: cover;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Input features from the user
st.sidebar.title("Input Features")  # Sidebar title
gender = st.sidebar.radio("Select Gender", ["Male", "Female"])  # Use radio button for binary choice
hemoglobin = st.sidebar.slider("Hemoglobin", min_value=0.0, max_value=20.0, value=10.0, step=0.1)
mch = st.sidebar.slider("MCH", min_value=0.0, max_value=100.0, value=30.0, step=0.1)
mchc = st.sidebar.slider("MCHC", min_value=0.0, max_value=100.0, value=35.0, step=0.1)
mcv = st.sidebar.slider("MCV", min_value=0.0, max_value=150.0, value=80.0, step=0.1)

# Create a DataFrame with user input
user_input = pd.DataFrame({
    'Gender': [1 if gender == "Male" else 0],  # Convert categorical to numerical
    'Hemoglobin': [hemoglobin],
    'MCH': [mch],
    'MCHC': [mchc],
    'MCV': [mcv]
})

# Make predictions
prediction = random_forest_model.predict(user_input)

# Display the prediction with styled text
st.subheader("Prediction:")

# Apply different colors and font sizes based on prediction
if prediction[0] == 1:
    st.markdown("<p style='color: red; font-size: 20px;'>High Risk of Anemia</p>", unsafe_allow_html=True)

else:
    st.markdown("<p style='color: green; font-size: 20px;'>Low Risk of Anemia</p>", unsafe_allow_html=True)
