# test_predict.py
import requests

# URL del endpoint local de predicción
url = "http://localhost:5000/predict"

# Datos de prueba para un usuario
payload = {
    "instances": [
        {
            "frecuencia_lectura": 1,
            "horas_lectura": 0.3,
            "quiere_recomendaciones": 0
        },
        {
            "frecuencia_lectura": 4,
            "horas_lectura": 2.5,
            "quiere_recomendaciones": 1
        }
    ]
}

# Enviar POST request
response = requests.post(url, json=payload)

# Mostrar respuesta
if response.status_code == 200:
    print("✅ Respuesta de la API:")
    print(response.json())
else:
    print(f"❌ Error {response.status_code}: {response.text}")
