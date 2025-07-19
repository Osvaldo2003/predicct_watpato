import json
import pandas as pd
from app.predictor import predecir_abandono

# Cargar métricas desde el archivo generado por extract_user_metrics.py
with open('user_metrics.json', 'r') as f:
    data = json.load(f)

# Convertir a DataFrame para manejar los datos cómodamente
df = pd.DataFrame(data)

# Ejecutar predicción para cada usuario
resultados = []
for _, row in df.iterrows():
    datos_usuario = row.drop('user_id').to_dict()
    resultado = predecir_abandono(datos_usuario)
    resultado['user_id'] = row['user_id']
    resultados.append(resultado)

# Mostrar resultados en consola
for r in resultados:
    print(f"Usuario {r['user_id']}: Riesgo = {r['riesgo_abandono']}, Probabilidad = {r['probabilidad']}")

# Exportar resultados a un archivo JSON
with open('predicciones_abandono.json', 'w') as f:
    json.dump(resultados, f, indent=2)

print("✅ Predicciones guardadas en predicciones_abandono.json")
