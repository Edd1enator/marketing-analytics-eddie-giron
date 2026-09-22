# Archivo base para el despliegue del Agente en Streamlit
import streamlit as st
from sklearn.linear_model import LinearRegression
import numpy as np

st.title("Configuracion inicial")
st.write("Primera prueba de uso streamlit y ambiente de MA2026")

gasto = st.slider("Seleccione nivel de gasto en publicidad", 10, 200, 50)

variable_x = np.array([[10], [20], [30], [40], [50]])
variable_y = np.array([15,25, 35, 45, 55])
modelo_lr = LinnearRegression()

modelo_lr.fit(variable_x, variable_y)

if st.button("predecir"):
    resultado = modelo_lr.predict([[gasto]])

    # streamlit muestra un mensaje de exito en verde bonito usando success
    st.success(f"Las ventas proyectadas para una inversion de ${gasto} son:${resultado[0]}")