import streamlit as st
import pandas as pd
import joblib

# ------------------------------
# Load Model
# ------------------------------
model = joblib.load("titanic_model.joblib")

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="🚢 Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide"
)

# ------------------------------
# Title Section
# ------------------------------
st.title("🚢 Titanic Survival Prediction App")
st.markdown("""
Welcome!  
This app predicts whether a passenger would have **survived** the Titanic disaster based on their details.  
Provide the passenger information in the sidebar and click **Predict Survival**.
""")

# ------------------------------
# Sidebar Inputs
# ------------------------------
st.sidebar.header("Passenger Details")

pclass = st.sidebar.selectbox("Ticket Class (Pclass)", [1, 2, 3])
sex = st.sidebar.selectbox("Sex", ["male", "female"])
age = st.sidebar.slider("Age", 0, 100, 29)
sibsp = st.sidebar.slider("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.sidebar.slider("Parents/Children Aboard", 0, 10, 0)
fare = st.sidebar.number_input("Ticket Fare", 0.0, 600.0, 32.2)
embarked = st.sidebar.selectbox("Port of Embarkation", ["C", "Q", "S"])

# ------------------------------
# Preprocess Inputs
# ------------------------------
sex = 0 if sex == "male" else 1
embarked_dict = {"C": 0, "Q": 1, "S": 2}
embarked = embarked_dict[embarked]

# Final Input
input_data = pd.DataFrame(
    [[pclass, sex, age, sibsp, parch, fare, embarked]],
    columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
)

# ------------------------------
# Prediction
# ------------------------------
if st.button("🔮 Predict Survival"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][prediction]

    st.subheader("📊 Prediction Result")
    if prediction == 1:
        st.success(f"The passenger **SURVIVED** 🎉 (Probability: {probability:.2f})")
    else:
        st.error(f"The passenger **DID NOT SURVIVE** 💔 (Probability: {probability:.2f})")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.markdown("💡 *This project demonstrates the power of Machine Learning applied to real-world historical data.*")
