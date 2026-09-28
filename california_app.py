import numpy as np
import joblib
import streamlit as st

Modelx = joblib.load('model.joblib') 

Model = Modelx['model']

Values = Modelx['columns']
st.title('california Housing App')

v = []

for value in Values:
    vx =st.number_input(value)
    v.append(vx)
    
if st.button('Predict'):
    input_data = np.array(v).reshape(1, -1)
    prediction = Model.predict(input_data)
    st.write(f'Predicted Value: {prediction[0]}')
