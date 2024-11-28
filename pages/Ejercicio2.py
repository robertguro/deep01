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
st.title('Ejercicio # 2 Deep Learning') 
#Subtitulo
st.subheader('En este ejercicio se implementará un modelo de Perceptron para clasificar un conjunto de datos de 10 entradas y 5 salidas.')
st.subheader('El problema trata de predecir el tipo de vehículo en base a las entradas dadas } .')
st.subheader('El número de experimentos es de 5000, algunos parámetros pueden ser modificados por el usuario.')

#Menu de opciones
st.sidebar.title('Menú de opciones')
#Lista de opciones
opciones = ['Cargar Datos', 'Desarrollo Ejercicio # 2', 'Codigo']
#Selección de la opción
opcion = st.sidebar.selectbox('Seleccione una opción', opciones)

#Carga de datos
@st._cache_data
def cargar_datos(archivo):
    if archivo:
        if archivo.name.endswith('csv'):
            df = pd.read_csv(archivo, sep=';')
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
elif opcion == 'Desarrollo Ejercicio # 2':
    
    #Recuperamos el archivo y generamos las opciones
    df = st.session_state.df
    st.write('El archivo contiene {} filas y {} columnas'.format(df.shape[0], df.shape[1]))
    st.write(df.head())
    iteraciones = st.sidebar.slider('Seleccione el número de iteraciones', 1, 500, 100)
    random_state = st.sidebar.slider('Seleccione el valor de random_state', 0, 100, 42)
    alpha = st.sidebar.slider('Seleccione el valor de alpha', 0.0001, 10.0, 0.0001)

    #quitamos las 2 columnas que no son necesarias para el análisis (highway-mpg y city-mpg)
    df = df.drop(['highway_mpg', 'city_mpg'], axis=1)
    st.write('El archivo contiene {} filas y {} columnas'.format(df.shape[0], df.shape[1]))
    st.write(df.head())

    #Separamos las variables dependientes e independientes
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1:].values

    #Generamos dummy variables independientes
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    X_d = df.drop(['year','engine_hp','engine_cylinders','number_of_doors','popularity','msrp'],axis=1)
    X_d = pd.get_dummies(X_d)
        
    #Generamos dummy variables dependientes
    y_d = df['vehicle_style'].replace({'Convertible': 0, 'Hatchback': 1, 'Pickup': 2, 'Sedan': 3, 'Wagon': 4})
    
    X = pd.concat([X_d, df[['year','engine_hp','engine_cylinders','number_of_doors','popularity','msrp']]], axis=1)
    y = y_d
    #Normalizamos los datos
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X = sc.fit_transform(X)
    st.write('Datos normalizados y aplicando OneHotEncoder')
    
    # Mostramos X e y
    st.write('X:', X)
    st.write('y:', y)

    #Entrenamiento de la red neuronal
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

elif opcion == 'Codigo':
    st.write('A continuación se muestra el código del ejercicio # 2')
    #mostrando el codigo:
    code = '''
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
    st.title('Ejercicio # 2 Deep Learning') 
    #Subtitulo
    st.subheader('En este ejercicio se implementará un modelo de Perceptron para clasificar un conjunto de datos de 10 entradas y 5 salidas.')
    st.subheader('El problema trata de predecir el tipo de vehículo en base a las entradas dadas } .')
    st.subheader('El número de experimentos es de 5000, algunos parámetros pueden ser modificados por el usuario.')

    #Menu de opciones
    st.sidebar.title('Menú de opciones')
    #Lista de opciones
    opciones = ['Cargar Datos', 'Desarrollo Ejercicio # 2', 'Codigo']
    #Selección de la opción
    opcion = st.sidebar.selectbox('Seleccione una opción', opciones)

    #Carga de datos
    @st._cache_data
    def cargar_datos(archivo):
        if archivo:
            if archivo.name.endswith('csv'):
                df = pd.read_csv(archivo, sep=';')
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
    elif opcion == 'Desarrollo Ejercicio # 2':
        
        #Recuperamos el archivo y generamos las opciones
        df = st.session_state.df
        st.write('El archivo contiene {} filas y {} columnas'.format(df.shape[0], df.shape[1]))
        st.write(df.head())
        iteraciones = st.sidebar.slider('Seleccione el número de iteraciones', 1, 500, 100)
        random_state = st.sidebar.slider('Seleccione el valor de random_state', 0, 100, 42)
        alpha = st.sidebar.slider('Seleccione el valor de alpha', 0.0001, 10.0, 0.0001)

        #quitamos las 2 columnas que no son necesarias para el análisis (highway-mpg y city-mpg)
        df = df.drop(['highway_mpg', 'city_mpg'], axis=1)
        st.write('El archivo contiene {} filas y {} columnas'.format(df.shape[0], df.shape[1]))
        st.write(df.head())

        #Separamos las variables dependientes e independientes
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1:].values

        #Generamos dummy variables independientes
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        X_d = df.drop(['year','engine_hp','engine_cylinders','number_of_doors','popularity','msrp'],axis=1)
        X_d = pd.get_dummies(X_d)
            
        #Generamos dummy variables dependientes
        y_d = df['vehicle_style'].replace({'Convertible': 0, 'Hatchback': 1, 'Pickup': 2, 'Sedan': 3, 'Wagon': 4})
        
        X = pd.concat([X_d, df[['year','engine_hp','engine_cylinders','number_of_doors','popularity','msrp']]], axis=1)
        y = y_d
        #Normalizamos los datos
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X = sc.fit_transform(X)
        st.write('Datos normalizados y aplicando OneHotEncoder')
        
        # Mostramos X e y
        st.write('X:', X)
        st.write('y:', y)

        #Entrenamiento de la red neuronal
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
    '''
    st.code(code, language="python")
    