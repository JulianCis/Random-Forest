import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

# Cargar el archivo y leerlo
data = pd.read_csv('bankloan.csv')

# Eliminar columnas irrelevantes para el modelo
# En este caso, el ID no aporta nada, ni tampoco el zip code ya que no tenemos algo con qué comparar
# si el zip code determina alguna zona donde haya mayor nivel socioeconómico
data.drop(['ID', 'ZIP.Code'], axis=1, inplace=True)

print(data)
# Separar variables independientes X, el cual sólo tiene datos numéricos y la variable objetivo y, en este caso, Personal.Loan
X = data.drop('Personal.Loan', axis=1)
y = data['Personal.Loan']

# Dividir el conjunto de datos ya limpios en entrenamiento y prueba (80% para entrenamiento, y 20% para prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Dividir el conjunto de entrenamiento en entrenamiento y validación (80% para entrenamiento, y 20% para validación)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42, stratify=y_train)

# Creamos el modelo de random forest con una misma semilla para tener los mismos valores cuando lo corremos
random_forest = RandomForestClassifier(random_state=42)

# Definir los hiperparámetros, podemos ajustar los estimadores (número de árboles) y la profundidad máxima de cada uno 
# de los árboles de decisión dentro del random forest
param_grid = {
    'n_estimators': [30, 50],
    'max_depth': [3, 5, 8, 10]
}

# Usamos grid search para buscar los mejores hiperparámetros usando los datos de validación que habíamos apartado
# Le subimos el cross validation para evitar overfitting
grid_search = GridSearchCV(estimator=random_forest, param_grid=param_grid, cv=5, scoring='accuracy')

# Entrenamos al modelo con los datos de entrenamiento
grid_search.fit(X_train, y_train)

# Obtenemos los mejores hiperparámetros y con base en esto, igual el mejor modelo
best_rf = grid_search.best_estimator_
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Imprimimos Los mejores hiperparámetros, sacados con grid search
print("Mejores hiperparámetros:", best_params)
print("Mejor puntuación de validación usando grid search:", best_score)

# Evaluar el modelo con el conjunto de entrenamiento
train_score = best_rf.score(X_train, y_train)
print("Puntuación de entrenamiento:", train_score)

# Evaluar el modelo con el conjunto de validación
val_score = best_rf.score(X_val, y_val)
print("Puntuación de validación:", val_score)

# Evaluar el modelo con el conjunto de prueba
test_score = best_rf.score(X_test, y_test)
print("Puntuación de prueba:", test_score)

# Hacemos predicciones con validación
y_val_pred = best_rf.predict(X_val)

# Hacemos predicciones con prueba
y_test_pred = best_rf.predict(X_test)

# Usamos la matriz de confusión para el conjunto de validación
conf_matrix_val = confusion_matrix(y_val, y_val_pred)
print("Matriz de confusión (validación):\n", conf_matrix_val)

# Matriz de confusión para el conjunto de prueba
conf_matrix_test = confusion_matrix(y_test, y_test_pred)
print("Matriz de confusión (prueba):\n", conf_matrix_test)

# Evaluación con métricas adicionales como acccuracy, precision, recall y f1, es importante saber que la evaluación funciona
# como un conjunto de todas las métricas
print("Evaluación del conjunto de prueba:\n")
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

# Imprimimos las métricas
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Reporte de clasificación más detallado
classification_rep = classification_report(y_test, y_test_pred)
print("Reporte de clasificación:\n", classification_rep)


# En conclusión, se puede ver queaún usando cross validation con 5 folds, este puede llegar a generar overfitting
# Y tambén, random forest es bueno para este tipo de problemas de clasificación