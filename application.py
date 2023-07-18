#Importing necessary packages and modules
import streamlit as st
import warnings
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Creating the dataframe
df=pd.read_csv('Pulsar.csv')

#Splitting X and y values
X=df.iloc[:,:-1]
y=df.iloc[:,-1]

#Scaling X values
sc=StandardScaler()
X=sc.fit_transform(X)

#Model selection
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1,test_size=0.3)

#Ignoring warnings
warnings.filterwarnings("ignore")

#Developing the model
rfc=RandomForestClassifier(criterion="entropy",max_features=None,min_samples_split=10,n_estimators=30,random_state=1)
rfc.fit(X_train,y_train)

#Developing the application
hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.title("Pulsar Star Prediction Application")
def create_form():
    with st.form(key="my_form"):
        v1 = st.number_input("Mean of the integrated profile:")
        v2 = st.number_input("SD of the integrated profile:")
        v3 = st.number_input("EK of the integrated profile:")
        v4 = st.number_input("Skewness of the integrated profile:")
        v5 = st.number_input("Mean of the DM-SNR Curve:")
        v6 = st.number_input("SD of the DM-SNR Curve:")
        v7 = st.number_input("EK of the DM-SNR Curve:")
        v8 = st.number_input("Skewness of the DM-SNR Curve:")
        submit_button = st.form_submit_button("Predict")
    if submit_button:
        if rfc.predict(sc.transform([[v1,v2,v3,v4,v5,v6,v7,v8]]))==0:
            st.error("Not a Pulsar Star")
        else:
            st.success("Pulsar Star")
create_form()