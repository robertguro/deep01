#Importamos las librerias necesarias
import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split

#Condifuración de la página
st.set_page_config(page_title='Deep Learning', page_icon=':bar_chart:', layout='wide', initial_sidebar_state='expanded')
#Página principal
st.title('Ejercicio # 3 Deep Learning') 
#Menu de opciones
st.sidebar.title('Menú de opciones')
#Lista de opciones
opciones = ['Cargar Datos', 'Desarrollo Ejercicio # 3']
#Selección de la opción
opcion = st.sidebar.selectbox('Seleccione una opción', opciones)

#Carga de datos
@st._cache_data
def cargar_datos(archivo):
    if archivo:
        if archivo.name.endswith('csv'):
            df = pd.read_csv(archivo)
        elif archivo.name.endswith('xlsx'):
            df = pd.read_excel(archivo)
        else:
            raise ValueError('Formato de archivo no soportado')
        return df
    else:
        return None

if opcion == 'Cargar Datos':
    st.subheader('Cargar datos')
    archivo = st.sidebar.file_uploader('Seleccione un archivo', type=['csv', 'xlsx'])
    if archivo:
        df = cargar_datos(archivo)
        st.session_state.df = df
        st.info('Datos cargados correctamente')
    else:
        st.write('No hay datos para mostrar')
elif opcion == 'Desarrollo Ejercicio # 3':
    
    df = st.session_state.df
    iteraciones = st.sidebar.slider('Seleccione el número de iteraciones', 1, 500, 100)
    random_state = st.sidebar.slider('Seleccione el valor de random_state', 0, 100, 42)
    alpha = st.sidebar.slider('Seleccione el valor de alpha', 0.0001, 10.0, 0.0001)

    #Agregando make_classification
    from sklearn.datasets import make_classification
    X = make_classification(n_samples=50000,n_features=15, n_redundant=0, n_informative=2, random_state=42, n_clusters_per_class=1)
    X = pd.DataFrame(X[0], columns=['X'+str(i) for i in range(1,16)])
    X
    y = np.random.randint(0, 15, 50000)
    y = pd.DataFrame(y, columns=['Y'])
    y
     #Dividimos los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    #Entrenamos el modelo
    perceptron = Perceptron(max_iter=iteraciones, random_state=random_state, alpha=alpha)
    perceptron.fit(X_train, y_train)

    #Mostar resultados
    st.write('Precisión del modelo: {:.2f}'.format(perceptron.score(X_test, y_test)))
    # Parametros del modelo
    st.write('Parametros del modelo: ', perceptron.coef_)
    # Intercepción del modelo
    st.write('Intercepción del modelo: ', perceptron.intercept_)
    # mostramos los parametros del modelo
    st.write('Parametros del modelo: ', perceptron.get_params())
    #para mostrar la matriz de confusión
    from sklearn.metrics import confusion_matrix
    y_pred = perceptron.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    st.write('Matriz de confusión: ', cm)

    # Realizando predicciones
    y_pred = perceptron.predict(X)
    st.write('Predicciones: ', y_pred)
