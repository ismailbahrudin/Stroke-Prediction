import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Simple Stroke Prediction App
This app predicts the **Stroke patient** 
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    gender == st.sidebar.selectbox(
    "Choose a gender",   
    ('Male', 'Female'),
    )

    smoke == st.sidebar.selectbox(
    "Smoking Status",   
    ('never smoked', 'formerly smoked','smoke'),
    )

    age = st.sidebar.slider('Age', 0, 90, 40)

    married == st.sidebar.selectbox(
    "Ever Married?",   
    ('Yes', 'No'),
    )

    residence == st.sidebar.selectbox(
    "Residence Type",   
    ('Rural', 'Urban'),
    )
    glucose = st.sidebar.slider('Average Glucose Level', 50, 300, 150)
    
    data = {'gender': gender,
            'smoker': smoke,
	    'age': age,
            'married': married,
            'residence': residence,
            'glucose': glucose}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

stroke = pd.read_csv('https://raw.githubusercontent.com/ismailbahrudin/Stroke-Prediction/main/healthcare-dataset-stroke-data.csv')
X = stroke.drop(['id','hypertension','heart_disease','work_type','bmi','stroke'],axis=1)
Y = stroke['stroke']


clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
dataf= pd.DataFrame(['Stroke','Non-Stroke'])
st.write(dataf)

st.subheader('Prediction')
#st.write(iris.target_names[prediction])
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
