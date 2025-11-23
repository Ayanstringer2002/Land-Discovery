import streamlit as st
import pandas as pd
import joblib
import os

# -------- Load Model --------
MODEL_PATH = "models/model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model file not found. Please train the model first.")
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("🏗 AI Land Recommendation System")
st.markdown("Analyze land based on **Safety**, **Infrastructure**, and **Environment** factors.")

st.sidebar.title("Navigation")
option = st.sidebar.radio("Select Mode", ["Single Prediction", "Bulk CSV Upload"])

# ------------------------------------------
# ✅ SINGLE LAND PREDICTION
# ------------------------------------------

if option == "Single Prediction":

    st.subheader("📌 Enter Land Parameters")

    safety = st.slider("Safety Score", 0.0, 10.0, 5.0)
    infrastructure = st.slider("Infrastructure Score", 0.0, 10.0, 5.0)
    environment = st.slider("Environmental Quality", 0.0, 10.0, 5.0)

    # Add more features here if your model uses more
    if st.button("Predict Land Suitability"):

        input_data = pd.DataFrame({
            "safety_score": [safety],
            "infrastructure_score": [infrastructure],
            "environment_score": [environment]
        })

        prediction = model.predict(input_data)[0]

        st.success(f"✅ Predicted Suitability Score: {round(prediction, 2)}")

        if prediction > 7:
            st.markdown("🏆 **Highly Recommended Land**")
        elif prediction > 4:
            st.markdown("✅ **Moderately Suitable Land**")
        else:
            st.markdown("⚠ **Not Recommended Land**")

# ------------------------------------------
# ✅ BULK CSV PREDICTION
# ------------------------------------------

if option == "Bulk CSV Upload":

    st.subheader("📂 Upload CSV File")
    uploaded_file = st.file_uploader("Upload your land CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("📄 Uploaded Data Preview", df.head())

        if st.button("Run Prediction on File"):

            predictions = model.predict(df)
            df["Predicted_Score"] = predictions

            st.write("✅ Prediction Results", df)

            # Save results
            output_file = "data/predictions_output.csv"
            df.to_csv(output_file, index=False)

            st.success(f"Predictions saved to: {output_file}")
            st.download_button("📥 Download Results", df.to_csv(index=False), "predictions.csv")
