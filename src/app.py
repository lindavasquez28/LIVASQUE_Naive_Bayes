from utils import db_connect
engine = db_connect()

# your code here
# # LINDA VASQUEZ Naive Bayes

# ## Carga de datos

import pandas as pd

url = "https://raw.githubusercontent.com/4GeeksAcademy/naive-bayes-project-tutorial/main/playstore_reviews.csv"
datos = pd.read_csv(url)

# ## 2. Estudio de variables y su contenido (+ limpieza)

datos = datos.drop("package_name", axis=1)

# Limpiaeza de texto (minúsculas y sin espacios en los extremos)
datos["review"] = datos["review"].str.strip().str.lower()

from sklearn.model_selection import train_test_split

X = datos["review"]
y = datos["polarity"]

# 80% para entrenar y 20% para probar, random_state=42 para que siempre de igual.
X_entrena, X_prueba, y_entrena, y_prueba = train_test_split(X, y, test_size=0.2, random_state=42)

# CountVectorizer para convertir las frases en números
from sklearn.feature_extraction.text import CountVectorizer

# stop_words="english" quita conectores innecesarios
vectorizador = CountVectorizer(stop_words="english")

# Transformacion de textos a números
X_entrena_num = vectorizador.fit_transform(X_entrena).toarray()
X_prueba_num = vectorizador.transform(X_prueba).toarray()

# ## 3. Modelo Naive Bayes

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# MODELO MultinomialNB
modelo_multi = MultinomialNB()
modelo_multi.fit(X_entrena_num, y_entrena)

predicciones_multi = modelo_multi.predict(X_prueba_num)

precision = accuracy_score(y_prueba, predicciones_multi)
print("Precisión Multinomial:", precision)

from sklearn.naive_bayes import GaussianNB, BernoulliNB

# MODELO GaussianNB
modelo_gauss = GaussianNB()
modelo_gauss.fit(X_entrena_num, y_entrena)
print("Precisión Gaussiano:", accuracy_score(y_prueba, modelo_gauss.predict(X_prueba_num)))

# MODELO BernoulliNB
modelo_bernoulli = BernoulliNB()
modelo_bernoulli.fit(X_entrena_num, y_entrena)
print("Precisión Bernoulli:", accuracy_score(y_prueba, modelo_bernoulli.predict(X_prueba_num)))

# ## 4. Optimización del modelo

import numpy as np
from sklearn.model_selection import RandomizedSearchCV

parametros = {
    "alpha": np.linspace(0.01, 10.0, 200),
    "fit_prior": [True, False]
}

busqueda = RandomizedSearchCV(modelo_multi, parametros, n_iter=50, cv=5, random_state=42)
busqueda.fit(X_entrena_num, y_entrena)

print("Los mejores parámetros se encontraron fueron:", busqueda.best_params_)

modelo_final = MultinomialNB(
    alpha=busqueda.best_params_["alpha"], 
    fit_prior=busqueda.best_params_["fit_prior"]
)
modelo_final.fit(X_entrena_num, y_entrena)

predicciones_finales = modelo_final.predict(X_prueba_num)
print("Nueva precisión optimizada:", accuracy_score(y_prueba, predicciones_finales))

# ## 5. Guardado del modelo

from pickle import dump

# Los dos puntos y el slash (../) le dicen a Python que vaya una carpeta hacia atrás
dump(modelo_final, open("../models/modelo_naive_bayes_optimizado.sav", "wb"))

print("¡Modelo guardado con éxito! Tu intuición fue correcta.")

# ## 6. Alternativa: Random Forest

from sklearn.ensemble import RandomForestClassifier

# 1. Inicializamos el modelo (usamos random_state=42 para que tus resultados sean los mismos si lo corres de nuevo)
modelo_bosque = RandomForestClassifier(random_state=42)

# 2. Entrenamos el modelo asegurándonos de usar las variables NUMÉRICAS (X_entrena_num)
modelo_bosque.fit(X_entrena_num, y_entrena)

# 3. Hacemos las predicciones con los datos de prueba numéricos (X_prueba_num)
predicciones_bosque = modelo_bosque.predict(X_prueba_num)

# 4. Calculamos y mostramos la precisión para compararla con la de Naive Bayes
precision_bosque = accuracy_score(y_prueba, predicciones_bosque)
print("Precisión del Random Forest:", precision_bosque)

# Lo guardo tambien por si acaso
dump(modelo_bosque, open("../models/modelo_random_forest.sav", "wb"))
print("¡Modelo Random Forest guardado con éxito!")

# ### Tras entrenar y comparar el modelo de Bosque Aleatorio (Random Forest) con el Naive Bayes Multinomial, obtuve los siguientes resultados:
# 
# ### Precisión Naive Bayes Optimizada: 82.1%
# 
# ### Precisión Random Forest: 79.8%
# 
# ### Aunque Random Forest es un algoritmo robusto para detectar relaciones complejas, para análisis de sentimientos el modelo Naive Bayes es superior.

