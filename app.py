import streamlit as st
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('ohe_geo.pkl','rb') as file:
    ohe_geo = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)


st.title('Customer Churn prediction')

geography = st.selectbox('Geography',ohe_geo.categories_[0])
gender = st.selectbox('Gender',label_encoder_gender.classes_)
age = st.slider('Age',18,92)
balance= st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])

input_data = {
    'CreditScore':credit_score,
    'Geography':geography,
    'Gender':gender,
    'Age':age,
    'Tenure':tenure,
    'Balance':balance,
    'NumOfProducts':num_of_products,
    'HasCrCard':has_cr_card,
    'IsActiveMember':is_active_member,
    'EstimatedSalary':estimated_salary
}

data = pd.DataFrame([input_data])
geo = ohe_geo.transform(data[['Geography']])
data['Gender'] = label_encoder_gender.transform(data['Gender'])
geography = pd.DataFrame(geo.toarray(),columns=ohe_geo.get_feature_names_out(['Geography']))
data = pd.concat([data.drop('Geography',axis=1),geography],axis=1)
data_scaled = scaler.transform(data)
prediction = model.predict(data_scaled)
prediction_probab = prediction[0][0]
result = "The Customer will not Churn" if prediction_probab < 0.5 else "The Customer will Churn"
st.write(f"The Customer Churn Probability is {prediction_probab:.2f}")
st.write(result)


