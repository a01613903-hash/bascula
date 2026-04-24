# Importar librerías necesarias
import numpy as np
import streamlit as st
import pandas as pd

# Insertamos título
st.write(''' # ODS 4: Educación de Calidad ''')
# Insertamos texto con formato
st.markdown("""
Esta aplicación utiliza **Machine Learning** para predecir el rendimiento académico
basado en las horas de estudio, alineado con el **ODS 4: Educación de Calidad**.
""")

# Insertamos una imagen (opcional, reemplaza con tu ruta o URL)
# st.image("escuela.jpg", caption="Impacto del estudio en el rendimiento académico.")

# Definimos cómo ingresará los datos el usuario
# Usaremos un deslizador
st.sidebar.header("Parámetros Académicos")
# Definimos los parámetros de nuestro deslizador:
  # Límite inferior: 0 horas.
  # Límite superior: 40 horas. Límite razonable para estudio semanal
  # Valor inicial: 15 horas. Promedio saludable en estudiantes
horas_input = st.sidebar.slider("Horas de Estudio Semanales", 0.0, 40.0, 15.0)

# Cargamos el archivo con los datos (.csv)
df = pd.read_csv('Calificaciones_ODS4.csv', encoding='latin-1')
# Seleccionamos las variables
X = df[['Horas_Estudio']]
y = df['Calificacion_Final']

# Creamos y entrenamos el modelo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
LR = LinearRegression()
LR.fit(X_train,y_train)

# Hacemos la predicción con el modelo y las horas seleccionadas por el usuario
b1 = LR.coef_
b0 = LR.intercept_ 
prediccion = b0 + b1[0]*horas_input

# Presentamos los resultados
st.subheader('Pronóstico de Rendimiento')
st.write(f'La calificación estimada es: {prediccion:.2f}/100')

if prediccion < 50:
        st.error("Estado: Riesgo de Reprobación")
elif prediccion < 75:
        st.warning("Estado: Rendimiento Promedio")
else:
        st.success("Estado: Excelente Rendimiento")
