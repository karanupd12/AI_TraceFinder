import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="TraceFinder Predictor", page_icon="🔎", layout="centered")

st.title("🔎 TraceFinder – Baseline Model Prediction")
st.write("Upload a CSV file to generate predictions using the trained baseline model.")

# Check if model exists
MODEL_PATH = "models/random_forest.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found! Please train the model first by running `train_baseline.py`.")
else:
    # Load model
    model = joblib.load(MODEL_PATH)

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        # Read CSV
        data = pd.read_csv(uploaded_file)
        st.subheader("📄 Preview of Uploaded Data")
        st.dataframe(data.head())

        # Predict
        st.subheader("📊 Predictions")
        try:
            predictions = model.predict(data)
            data['Prediction'] = predictions
            st.write(data)

            # Option to download results
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")
