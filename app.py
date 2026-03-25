import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Title
st.title("AI Citizen Complaint Classifier")

st.write("Enter your complaint and get predicted category")

# Input box
input_text = st.text_area("Enter Complaint")

# Button
if st.button("Predict"):
    if input_text.strip() != "":
        transformed = vectorizer.transform([input_text])
        result = model.predict(transformed)
        st.success(f"Predicted Category: {result[0]}")
    else:
        st.warning("Please enter some text")