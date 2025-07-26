# backend-predictivo/predict_api.py
import joblib
import json
from flask import Flask, request, jsonify
from pathlib import Path
import pandas as pd

MODEL_PATH = Path("backend-predictivo/model/abandono_model.pkl")
META_PATH  = Path("backend-predictivo/model/metadata.json")

app = Flask(__name__)

print("📦 Cargando modelo...")
model = joblib.load(MODEL_PATH)
with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

# Para simplificar la demo, esperamos que el frontend ya te mande los features numéricos/binarizados.
# Si quieres, puedes replicar el preprocesamiento aquí exactamente como en el train_model.py
EXPECTED_FEATURES = (
    meta["features_numeric"] + meta["features_binary"] + meta["features_categorical"]
)

@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok", model=meta["best_model"])

@app.route("/predict", methods=["POST"])
def predict():
    """
    Espera un JSON con:
    {
      "instances": [
        {
           "freq_lectura_num": 1,
           "freq_offline_num": 2,
           ...
           "pref_formato": "Semanal"
        },
        ...
      ]
    }
    """
    data = request.get_json()
    if data is None or "instances" not in data:
        return jsonify(error="Debes enviar 'instances'"), 400

    X = pd.DataFrame(data["instances"])
    # Verificación básica de columnas
    missing = set(EXPECTED_FEATURES) - set(X.columns)
    if missing:
        return jsonify(error=f"Faltan columnas: {list(missing)}"), 400

    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)

    riesgo_txt = []
    for p in proba:
        if p >= 0.75:
            riesgo_txt.append("Alto")
        elif p >= 0.5:
            riesgo_txt.append("Medio")
        else:
            riesgo_txt.append("Bajo")

    out = []
    for i in range(len(pred)):
        out.append({
            "probabilidad": float(proba[i]),
            "clase": int(pred[i]),
            "riesgo": riesgo_txt[i]
        })

    return jsonify(predictions=out)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
