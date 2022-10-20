import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model


#function that takes in the predictions and produces a decision
def get_nn_preds(predictions):
    #if the prob is less than 0.5 the append 0
    if predictions<0.5:
        prediction_values=0
    #if the probability is greater than 0.5 then append 1
    else:
        prediction_values=1
    return prediction_values

#Read in the data
df=pd.read_csv('Data/car.csv',header=None)
#add in the columns
df.columns=['buying','maint','doors','persons','lug_boot','safety','value']
#drop the output
df.drop('value',axis=1,inplace=True)

#Encode the data
for i in df.columns:
    le=LabelEncoder()
    df[i]=le.fit_transform(df[i])
#Establish each section of the dashboard
col1,col2,col3=st.columns(3)
#column 1 takes the selectbox inputs
with col1:
    buying=st.selectbox('Buying Input',list(set(df['buying'])))
    main=st.selectbox('Maintenance Input',list(set(df['maint'])))
#column 2 uses radio buttons to create inputs
with col2:
    doors=st.radio('Number of Doors',list(set(df['doors'])))
    lug_boot=st.radio('Lug Boot Input',list(set(df['lug_boot'])))
    safety=st.radio('Safety Input',list(set(df['safety'])))
#column three uses select slider widgets
with col3:
    doors=st.select_slider('Number of Doors',list(set(df['doors'])))
    persons=st.select_slider('Persons Input',list(set(df['persons'])))
#alter the input for the neural network
input=[[buying,main,doors,persons,lug_boot,safety]]
#load the model into the dash
model=load_model('Models/nn_model.sav')
#get the probability
prob=model.predict(input)[0][0]
#Form a decision using the model
decision=get_nn_preds(prob)
print(decision)
#Change number to Words
decision_dict={0:'Unacceptable Value',1:'Acceptable Value'}
decision_v=decision_dict[decision]
#Display value
st.header(decision_v)
