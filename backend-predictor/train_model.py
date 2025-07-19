import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Datos simulados para entrenamiento
data = pd.DataFrame({
    'dias_desde_ultimo_login': [1, 5, 10, 30, 60],
    'total_capitulos_leidos': [100, 80, 50, 10, 0],
    'promedio_dias_entre_sesiones': [1.0, 2.5, 4.0, 10.0, 20.0],
    'capitulos_creados': [3, 2, 1, 0, 0],
    'seguidores': [50, 20, 10, 1, 0],
    'siguiendo': [100, 80, 30, 5, 1],
    'comentarios_realizados': [20, 10, 5, 0, 0],
    'tiempo_lectura_total': [1000, 800, 400, 50, 10],
    'notificaciones_activadas': [1, 1, 1, 0, 0],
    'abandono': [0, 0, 0, 1, 1]
})

X = data.drop(columns='abandono')
y = data['abandono']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(model, 'app/model.pkl')
