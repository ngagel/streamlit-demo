import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Titanic Survival Rate", page_icon="🚢")


def preprocess_data(pclass, sex, age, sibsp, parch, fare, embarked):
    # transform data
    sex = "female" if "Female" else "male"
    embarked = embarked[0]

    # collect data as dict
    data = {
        "Pclass": [pclass],
        "Sex": [sex],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Embarked": [embarked]
    }
    df = pd.DataFrame(data)
    # st.dataframe(df) 

    # preprocessing pipeline 
    ### encode sex
    le_sex = joblib.load("../encoder/le_sex.joblib")
    df["Sex"] = le_sex.transform(df["Sex"])
    ### encode embarked
    le_embarked = joblib.load("../encoder/le_embarked.joblib")
    df["Embarked"] = le_embarked.transform(df["Embarked"])
    ### impute age
    imp_age = joblib.load("../encoder/imp_age.joblib")
    df["Age"] = imp_age.transform(df[["Age"]]) 
    ### impute fare
    imp_fare = joblib.load("../encoder/imp_fare.joblib")
    df["Fare"] = imp_fare.transform(df[["Fare"]])

    return df

def predict(df):
    # predict label
    clf = joblib.load("../model/decision_tree.joblib") 
    return clf.predict_proba(df)



st.title("Titanic Survival Rate Prediction :ice_cube: :ship:")
st.subheader("Test your survival rate here :sunglasses:")
with st.form("form"):
    # create input fields
    pclass = st.selectbox("Ticket Class", ("1", "2", "3"),
                        index=None, placeholder="Select ticket class")

    sex = st.selectbox("Sex", ("Male", "Female"),
                    index=None, placeholder="Select sex")

    age = st.number_input("Input age", 0, 100)

    sibsp = st.number_input("Input # of siblings/spouse", 0, 10)

    parch = st.number_input("Input # of parent/children", 0, 10)

    fare = st.number_input("Ticket Fare")

    embarked = st.selectbox("Port of Embarkation", ("Cherbourg", "Queenstown", "Southampton"),
                            index=None, placeholder="Select port of embarkation")
    
    # submit btn
    submit = st.form_submit_button("Predict")
    if submit:
        df = preprocess_data(pclass, sex, age, sibsp, parch, fare, embarked)
        result = predict(df)
        print(result)
        st.write(f"Your survival rate in Titanic is: {result[0][1] * 100:.2f}%")


