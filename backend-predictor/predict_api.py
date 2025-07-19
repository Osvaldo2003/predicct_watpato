import joblib
import numpy as np

# Cargar el modelo entrenado
model = joblib.load("app/model.pkl")

def predecir_abandono(datos_usuario):
    columnas = [
        'dias_desde_ultimo_login',
        'total_capitulos_leidos',
        'promedio_dias_entre_sesiones',
        'capitulos_creados',
        'seguidores',
        'siguiendo',
        'comentarios_realizados',
        'tiempo_lectura_total',
        'notificaciones_activadas'
    ]
    valores = [datos_usuario.get(col, 0) for col in columnas]
    entrada = np.array([valores])
    prob = model.predict_proba(entrada)[0][1]
    etiqueta = "Alto" if prob > 0.75 else "Medio" if prob > 0.4 else "Bajo"
    return {
        "riesgo_abandono": etiqueta,
        "probabilidad": round(float(prob), 2)
    }
