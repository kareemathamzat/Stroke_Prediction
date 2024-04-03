import streamlit as st
import pickle
import numpy as np

def load_model():
     with open('saved_stroke_steps.pkl', 'rb') as stroke_file:
        stroke_data = pickle.load(stroke_file)
     return stroke_data

stroke_data = load_model()


    
rf_classifier_loaded = stroke_data['model']
le_ever_married = stroke_data['le_ever_married']

def show_app():
    st.title("Stroke prediction app")
    st.write("""#### We need some information to predict if you have a stroke or not""")

    married = (
    "Yes",
    "No"
    )

    hyper_dict = {
    "Yes" : 1,
    "No" : 0
    }

    heart_dict = {
    "Yes" : 1,
    "No" : 0
    }


    age = st.slider("Age",0,100,18)
    avg_glucose_level = st.slider("Glucose Level",0.00,400.00)

    hypertensive = st.selectbox("Hypertensive ?",hyper_dict.keys())
    hypertension = hyper_dict[hypertensive]


    heart_condition = st.selectbox("Heart_disease ?",heart_dict.keys())
    heart_disease = heart_dict[heart_condition]


    ever_married = st.selectbox("Married ?",married)
    ok = st.button('Predict')
    if ok:
        x = np .array([[age,hypertension,heart_disease,ever_married,avg_glucose_level]])
        x[:,3] =le_ever_married.transform(x[:,3])
        x = x.astype(float)

        y_prediction = rf_classifier_loaded.predict(x)
        
        if y_prediction == 1:
                st.subheader('Prediction: You are at risk of stroke')
        else:
                st.subheader('Prediction: You are not at risk of stroke')



    

    

show_app()