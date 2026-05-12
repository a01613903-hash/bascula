# Importar librerías necesarias
import numpy as np
import streamlit as st
import pandas as pd

# Insertamos título
st.write(''' # ⚡ ODS 7: Energía Asequible y No Contaminante ''')
# Insertamos texto con formato
st.markdown("""
Esta aplicación utiliza **Machine Learning** para predecir la eficiencia de una planta fotovoltaica
basada en la intensidad de la radiación solar, alineado con el **ODS 7: Energía Asequible y No Contaminante**.
""")
# Insertamos una imagen
st.image("planta.jpg")

# Definimos cómo ingresará los datos el usuario
st.sidebar.header("Parámetros de Radiación")
# Límites basados en el dataset generado (200 a 1050)
radiacion_input = st.sidebar.slider("Intensidad de Radiación Solar", 200.0, 1050.0, 600.0)

# Cargamos el archivo con los datos
df = pd.read_csv('energia_ODS7.csv', encoding='latin-1')
# Seleccionamos las variables
X = df[['VAR_2']]
y = df['VAR_4']

# Creamos y entrenamos el modelo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
LR = LinearRegression()
LR.fit(X_train, y_train)

# Hacemos la predicción con el modelo y la radiación seleccionada
b1 = LR.coef_
b0 = LR.intercept_
prediccion = b0 + b1[0]*radiacion_input

# Presentamos los resultados
st.subheader('Energía Estimada')
st.write(f'La energía de salida estimada es: {prediccion:.2f} Watts')

if prediccion < 50:
    st.success("Estado: Baja Eficiencia")
elif prediccion < 80:
    st.warning("Estado: Eficiencia Moderada")
else:
    st.info("Estado: Alta Eficiencia")
