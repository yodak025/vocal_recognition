
# imports
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

# load datasets and split into test and train

iris = datasets.load_iris()
X = iris.data
y = iris.target

#  30% de test

# split train test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print("hello")
print(f"X_train : {X_train} \n X_test : {X_test} \n y_train : {y_train} \n y_test : {y_test}")

# Creamos el modelo de NB

model_NB = GaussianNB()

#entrenamos

model_NB.fit(X_train, y_train)

print("Media de la gaussiana para cada clase:\n Columnas = Características \n Filas = Clases\n")
print(model_NB.theta_)
print("Desviación estándar de la gaussiana para cada clase:\n Columnas = Características \n Filas = Clases\n")
print(model_NB.var_)

y_hat_NB = model_NB.predict(X_test)
print(y_hat_NB)
#Calculamos el accuracy
accuracy = accuracy_score(y_test, y_hat_NB)
print("Accuracy:", accuracy)

