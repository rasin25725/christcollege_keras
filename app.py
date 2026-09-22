import streamlit as st
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("machine_temperature_rnn.keras")

st.title("Machine Temperature Predictor")

st.write("Enter the previous two machine readings:")

temp1 = st.number_input("Previous Timestamp 1 - Temperature")
vibration1 = st.number_input("Previous Timestamp 1 - Vibration")

temp2 = st.number_input("Previous Timestamp 2 - Temperature")
vibration2 = st.number_input("Previous Timestamp 2 - Vibration")

if st.button("Predict Next Temperature"):

    input_data = np.array([
        [temp1, vibration1],
        [temp2, vibration2]
    ])

    input_data = input_data.reshape(1, 2, 2)

    prediction = model.predict(input_data, verbose=0)

    st.subheader("Predicted Next Machine Temperature:")

    st.success(f"{prediction[0][0]:.2f} °C")
