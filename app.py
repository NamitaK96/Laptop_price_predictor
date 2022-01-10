import streamlit as st
import pickle
import sklearn
import xgboost
import numpy as np

# import the model
df = pickle.load(open('df.pkl', 'rb'))
pipe = pickle.load(open('pipe.pkl', 'rb'))

st.title('Laptop Price Predictor')

# select brand
brand = st.selectbox('Brand', df['Company'].unique())

# type of laptop
type = st.selectbox('Type', df['TypeName'].unique())

# Ram
ram = st.selectbox('Ram', df['Ram'].unique())

# Weight
weight = st.number_input('Weight of Laptop')

# touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# IPS
ips = st.selectbox('IPS', ['No', 'Yes'])

# screen size
screen_size = st.number_input('Screen Size')

# screen resolution
resolution = st.selectbox('Screen Resolution', ['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

# cpu
cpu = st.selectbox('CPU', df['Cpu_name'].unique())

hdd = st.selectbox('HDD(in GB)', [0,128,256,512,1024,2048])

ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

gpu = st.selectbox('GPU', df['Gpu_name'].unique())

os = st.selectbox('OS', df['os'].unique())

if st.button('Predict Price'):
    ppi = None

    if touchscreen == 'Yes':
        touchscreen = 1

    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1

    else:
        ips = 0

    x_resolution = int(resolution.split('x')[0])
    y_resolution = int(resolution.split('x')[1])

    ppi = ((x_resolution)**2 + (y_resolution)**2)**0.5/ screen_size

    query = np.array([brand,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])

    query = query.reshape(1,12)

    st.title("The predicted price of this configuration is " + str(int(np.exp(pipe.predict(query)[0]))))

