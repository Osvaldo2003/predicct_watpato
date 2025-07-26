# backend/model/train_model.py
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# --------------------------
# CONFIG
# --------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "encuesta_usuarios.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "abandono_model.pkl"
META_PATH = MODEL_DIR / "metadata.json"

SEED = 42
TEST_SIZE = 0.2

# --------------------------
# CARGA Y LIMPIEZA DE DATOS
# --------------------------
print("📥 Leyendo CSV de encuesta...")
if not DATA_PATH.exists():
    raise FileNotFoundError(f"❌ No se encontró el archivo: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

# Normalización de nombres
df.columns = [col.strip() for col in df.columns]

# Renombrar columnas clave
df = df.rename(columns={
    "¿Con qué frecuencia lees libros digitales?": "frecuencia_lectura",
    "¿Cuántas horas al día dedicas a la lectura digital?": "horas_lectura",
    "¿Te gustaría recibir recomendaciones basadas en los libros que ya has leído?": "quiere_recomendaciones"
})

# Mapas
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

# Aplicar mapeos
df["frecuencia_lectura"] = df["frecuencia_lectura"].map(map_frecuencia)
df["horas_lectura"] = df["horas_lectura"].map(map_horas)
df["quiere_recomendaciones"] = df["quiere_recomendaciones"].map(map_recomienda)

# Validación de datos
df = df.dropna(subset=["frecuencia_lectura", "horas_lectura", "quiere_recomendaciones"])

if df.shape[0] < 10:
    raise ValueError("❌ No hay suficientes datos válidos después de limpiar el CSV.")

# --------------------------
# GENERAR ETIQUETA (abandono)
# --------------------------
cond_1 = df["frecuencia_lectura"] <= 1
cond_2 = df["horas_lectura"] < 0.5
cond_3 = df["quiere_recomendaciones"] == 0
df["abandono"] = ((cond_1.astype(int) + cond_2.astype(int) + cond_3.astype(int)) >= 2).astype(int)

# --------------------------
# ENTRENAMIENTO
# --------------------------
features = ["frecuencia_lectura", "horas_lectura", "quiere_recomendaciones"]
X = df[features]
y = df["abandono"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)

print("⚙️ Entrenando modelo...")
model = RandomForestClassifier(random_state=SEED, class_weight="balanced")
model.fit(X_train, y_train)

# --------------------------
# EVALUACIÓN
# --------------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)

print("\n📊 Reporte de clasificación:")
print(classification_report(y_test, y_pred))
print(f"ROC AUC: {auc:.4f}")

# --------------------------
# GUARDADO
# --------------------------
joblib.dump(model, MODEL_PATH)
print(f"\n✅ Modelo guardado en: {MODEL_PATH}")

metadata = {
    "features": features,
    "target": "abandono",
    "descripcion": "Predicción de abandono basado en hábitos de lectura digital.",
    "criterios": "abandono si lectura baja, pocas horas y no desea recomendaciones"
}
with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print(f"✅ Metadata guardada en: {META_PATH}")
