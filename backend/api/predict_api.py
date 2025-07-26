# backend/api/predict_api.py

from flask import Flask, jsonify
import pandas as pd
import joblib
import json
from pathlib import Path

app = Flask(__name__)

# --------------------------
# CONFIG
# --------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # <-- subimos un nivel (de /api a /backend)
MODEL_PATH = BASE_DIR / "model" / "abandono_model.pkl"
DATA_PATH = BASE_DIR / "data" / "encuesta_usuarios.csv"
META_PATH = BASE_DIR / "model" / "metadata.json"

# --------------------------
# CARGA
# --------------------------
print("⚙️ Cargando modelo y metadata...")
model = joblib.load(MODEL_PATH)

with open(META_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

features = metadata["features"]

# --------------------------
# ENDPOINT PRINCIPAL
# --------------------------
@app.route("/api/predicciones", methods=["GET"])
def get_predictions():
    try:
        df = pd.read_csv(DATA_PATH)
        df.columns = [col.strip() for col in df.columns]

        # Mapear datos
        df = df.rename(columns={
            "¿Con qué frecuencia lees libros digitales?": "frecuencia_lectura",
            "¿Cuántas horas al día dedicas a la lectura digital?": "horas_lectura",
            "¿Te gustaría recibir recomendaciones basadas en los libros que ya has leído?": "quiere_recomendaciones"
        })

        map_frecuencia = {
            "Casi nunca": 0,
            "Una vez al mes": 1,
            "Una vez por semana": 2,
            "Varias veces a la semana": 3,
            "A diario": 4
        }
        map_horas = {
            "Menos de 1 hora": 0.3,
            "Entre 1 y 2 horas": 1.5,
            "Entre 2 y 4 horas": 3,
            "Más de 4 horas": 5
        }
        map_recomienda = {
            "No me interesa": 0,
            "Prefiero explorar por mi cuenta": 0,
            "Tal vez": 0,
            "Sí, mucho": 1
        }

        df["frecuencia_lectura"] = df["frecuencia_lectura"].map(map_frecuencia)
        df["horas_lectura"] = df["horas_lectura"].map(map_horas)
        df["quiere_recomendaciones"] = df["quiere_recomendaciones"].map(map_recomienda)

        df = df.dropna(subset=features)
        if df.empty:
            return jsonify({"error": "No hay datos válidos para predecir."}), 400

        X = df[features]
        df["prob_abandono"] = model.predict_proba(X)[:, 1]
        df["abandono_predicho"] = model.predict(X)

        salida = df[["frecuencia_lectura", "horas_lectura", "quiere_recomendaciones", "prob_abandono", "abandono_predicho"]]
        return salida.to_dict(orient="records")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
