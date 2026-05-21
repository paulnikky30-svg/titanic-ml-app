import streamlit as st
import pickle
import pandas as pd

# Load trained model
import os
import pickle

model_path = os.path.join(os.path.dirname(__file__), "titanic_rf_model.pkl")
model = pickle.load(open(model_path, "rb"))

st.title("🚢 Titanic Survival Prediction App")

st.write("Enter passenger details below:")

# Inputs
pclass = st.selectbox("Pclass", [1, 2, 3])
sex = st.selectbox("Sex (0=Male, 1=Female)", [0, 1])
age = st.number_input("Age", 0, 100, 25)
sibsp = st.number_input("SibSp", 0, 10, 0)
parch = st.number_input("Parch", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 500.0, 10.0)
embarked = st.selectbox("Embarked (S=0, C=1, Q=2)", [0, 1, 2])

# Input format (same as training)
input_data = pd.DataFrame([[
    1, pclass, sex, age, sibsp, parch, fare, embarked
]], columns=[
    "PassengerId","Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"
])

# Predict button
if st.button("Predict"):
    result = model.predict(input_data)

    if result[0] == 1:
        st.success("🎉 Survived")
    else:
        st.error("💀 Not Survived")
