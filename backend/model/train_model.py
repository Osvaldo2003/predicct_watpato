# backend-predictivo/train_model.py
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# --------------------------
# CONFIG
# --------------------------
DATA_PATH = Path("backend-predictivo/data/encuesta_usuarios.csv")
MODEL_DIR  = Path("backend-predictivo/model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "abandono_model.pkl"
META_PATH  = MODEL_DIR / "metadata.json"

SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# --------------------------
# MAPEOS / LISTAS DE VARIABLES
# Ajusta los textos EXACTOS a los de tu CSV
# --------------------------

ORD_MAP_FRECUENCIA_LECTURA = {
    "Nunca": 0,
    "Rara vez": 1,
    "A veces": 2,
    "Frecuentemente": 3,
    "Todos los días": 4
}

ORD_MAP_FRECUENCIA_OFFLINE = {
    "Nunca": 0,
    "Casi nunca": 1,
    "A veces": 2,
    "Frecuentemente": 3,
    "Siempre": 4
}

ORD_MAP_PUBLICA = {
    "Nunca": 0,
    "Rara vez": 1,
    "Ocasionalmente": 2,
    "Frecuentemente": 3
}

LIKERT_1_5 = {
    "Nada importante": 1,
    "Poco importante": 2,
    "Neutral": 3,
    "Importante": 4,
    "Muy importante": 5
}

INFLUENCIA_1_5 = {
    "Nada": 1,
    "Poco": 2,
    "Neutral": 3,
    "Bastante": 4,
    "Mucho": 5
}

FACILIDAD_1_5 = {
    "Muy difícil": 1,
    "Difícil": 2,
    "Neutral": 3,
    "Fácil": 4,
    "Muy fácil": 5
}

SI_NO_MAP = {"Sí": 1, "Si": 1, "No": 0}

# Palabras clave para parsear la columna “¿Qué aspectos te resultan más molestos...?”
MOLESTIAS_KWS = {
    "anuncios": ["anuncio", "ads", "publicidad"],
    "pago": ["pago", "suscripción", "monetización"],
    "censura": ["censura", "bloqueo", "restricción"],
    "buscador": ["buscador", "búsqueda", "search"],
    "recomendaciones": ["recomendaciones", "recomienda"],
}

# --------------------------
# FUNCIONES AUXILIARES
# --------------------------

def normalize_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()

def contains_any(text, kws):
    return any(kw in text for kw in kws)

def build_molestias_df(series):
    """Serie de textos -> DataFrame con columnas binarias por categoría de molestia."""
    texts = series.fillna("").astype(str).str.lower()
    out = {}
    for key, kws in MOLESTIAS_KWS.items():
        out[f"molestia_{key}"] = texts.apply(lambda t: 1 if contains_any(t, kws) else 0)
    return pd.DataFrame(out)

def map_with_default(series, mapping, default=np.nan):
    return series.map(mapping).fillna(default)

def to_numeric(series):
    return pd.to_numeric(series, errors="coerce")

def build_target(df):
    """
    Regla heurística para etiquetar 'abandono'.
    Puedes ajustar los umbrales según tu criterio/validación.
    """
    # Variables ya numéricas/convertidas
    freq_lect = df["freq_lectura_num"]
    horas = df["horas_lectura_digital_num"]
    notifs = df["util_notificaciones_num"]
    personalizacion = df["importancia_personalizacion_num"]
    interact_autores = df["interaccion_autores_num"]
    suscripcion = df["paga_suscripcion_bin"]
    molestia_ads = df["molestia_anuncios"]
    molestia_pago = df["molestia_pago"]
    molestia_cens = df["molestia_censura"]

    cond_1 = (freq_lect <= 1) | (horas < 0.5)
    cond_2 = ((molestia_ads == 1) | (molestia_pago == 1) | (molestia_cens == 1)) & (suscripcion == 0)
    cond_3 = (personalizacion <= 2) & (notifs <= 2) & (interact_autores <= 2)

    abandono = ((cond_1.astype(int) + cond_2.astype(int) + cond_3.astype(int)) >= 2).astype(int)
    return abandono

# --------------------------
# CARGA Y LIMPIEZA
# --------------------------
print("📥 Cargando CSV...")
df_raw = pd.read_csv(DATA_PATH)

# Renombra aquí a claves cortas y consistentes (AJUSTA los nombres exactos a los tuyos):
rename_cols = {
    "¿Con qué frecuencia lees libros digitales?": "freq_lectura",
    "¿Cuántos capítulos sueles leer en una sola sesión?  ": "caps_por_sesion",
    "¿Prefieres leer historias completas o en capítulos semanales?  ": "pref_formato",
    "¿Utilizas plataformas digitales para leer historias?  ": "usa_plataformas",
    "¿Qué aspectos te resultan más molestos en las plataformas de lectura actuales?  ": "molestias",
    "¿Con qué frecuencia lees sin conexión a internet?  ": "freq_offline",
    "¿Qué tan importante es para ti poder personalizar tu perfil de usuario?  ": "importancia_personalizacion",
    "¿Te gustaría poder interactuar directamente con los autores que sigues?  ": "interaccion_autores",
    "¿Cuánto influye el sistema de recomendaciones en tu elección de lectura?  ": "infl_recomendaciones",
    "¿Cuán importante es para ti que las historias sean gratuitas?  ": "importancia_gratis",
    "¿Cuántas horas al día dedicas a la lectura digital?  ": "horas_lectura_digital",
    "¿Qué tan seguido escribes o publicas tus propias historias?  ": "frecuencia_publica",
    "¿Consideras importante que las plataformas de lectura protejan a sus usuarios de contenido ofensivo o discriminatorio?  ": "proteccion_contenido",
    "¿Qué tan útil consideras un sistema de notificaciones para libros o autores que sigues?  ": "util_notificaciones",
    "¿Qué tan fácil te resulta encontrar nuevos libros o autores en las plataformas actuales?  ": "facilidad_descubrimiento",
    "¿Te gustaría participar en desafíos de escritura o eventos literarios en línea?  ": "participa_desafios",
    "¿Qué tan importante es para ti poder descargar libros para leer sin conexión?": "importancia_descarga",
    "¿Cuántos anuncios consideras aceptables al leer un capítulo en línea?  ": "anuncios_aceptables",
    "¿Estarías dispuesto a pagar una suscripción para eliminar anuncios?  ": "paga_suscripcion",
    "¿Te interesaría usar una plataforma que valore y promueva escritores emergentes?  ": "valora_emergentes",
    "¿Cuánto influyen las reseñas y calificaciones en tu decisión de leer una historia?  ": "infl_resenas",
    "¿Te gustaría compartir la lectura de un libro con un amigo o pareja y dejarse notas o marcadores visibles mutuamente?": "compartir_lectura",
    "¿Qué tan importante es para ti la diversidad en los temas de las historias?  ": "importancia_diversidad",
    "¿Prefieres historias con imágenes, música o contenido interactivo?  ": "prefiere_interactivo",
    "¿Te gustaría recibir recomendaciones basadas en los libros que ya has leído?  ": "quiere_recomendaciones",
}
df = df_raw.rename(columns=rename_cols)

# Normalización de textos y mapeos
df["freq_lectura_num"] = map_with_default(df["freq_lectura"], ORD_MAP_FRECUENCIA_LECTURA, default=2)
df["freq_offline_num"] = map_with_default(df["freq_offline"], ORD_MAP_FRECUENCIA_OFFLINE, default=2)
df["frecuencia_publica_num"] = map_with_default(df["frecuencia_publica"], ORD_MAP_PUBLICA, default=1)

df["importancia_personalizacion_num"] = map_with_default(df["importancia_personalizacion"], LIKERT_1_5, default=3)
df["interaccion_autores_num"] = map_with_default(df["interaccion_autores"], LIKERT_1_5, default=3)
df["infl_recomendaciones_num"] = map_with_default(df["infl_recomendaciones"], INFLUENCIA_1_5, default=3)
df["importancia_gratis_num"] = map_with_default(df["importancia_gratis"], LIKERT_1_5, default=4)
df["proteccion_contenido_num"] = map_with_default(df["proteccion_contenido"], LIKERT_1_5, default=4)
df["util_notificaciones_num"] = map_with_default(df["util_notificaciones"], LIKERT_1_5, default=3)
df["facilidad_descubrimiento_num"] = map_with_default(df["facilidad_descubrimiento"], FACILIDAD_1_5, default=3)
df["importancia_descarga_num"] = map_with_default(df["importancia_descarga"], LIKERT_1_5, default=4)
df["valora_emergentes_num"] = map_with_default(df["valora_emergentes"], LIKERT_1_5, default=4)
df["infl_resenas_num"] = map_with_default(df["infl_resenas"], INFLUENCIA_1_5, default=3)
df["compartir_lectura_num"] = map_with_default(df["compartir_lectura"], LIKERT_1_5, default=3)
df["importancia_diversidad_num"] = map_with_default(df["importancia_diversidad"], LIKERT_1_5, default=4)

df["usa_plataformas_bin"] = map_with_default(df["usa_plataformas"], SI_NO_MAP, default=1)
df["participa_desafios_bin"] = map_with_default(df["participa_desafios"], SI_NO_MAP, default=0)
df["paga_suscripcion_bin"] = map_with_default(df["paga_suscripcion"], SI_NO_MAP, default=0)
df["prefiere_interactivo_bin"] = map_with_default(df["prefiere_interactivo"], SI_NO_MAP, default=0)
df["quiere_recomendaciones_bin"] = map_with_default(df["quiere_recomendaciones"], SI_NO_MAP, default=1)

df["horas_lectura_digital_num"] = to_numeric(df["horas_lectura_digital"])
df["caps_por_sesion_num"] = to_numeric(df["caps_por_sesion"])
df["anuncios_aceptables_num"] = to_numeric(df["anuncios_aceptables"])

# One-hot para el formato (completo/semanal)
df["pref_formato"] = df["pref_formato"].fillna("Desconocido")

molestias_df = build_molestias_df(df["molestias"])
df = pd.concat([df, molestias_df], axis=1)

# Construimos target
print("🏷️ Construyendo etiqueta 'abandono' (heurística ajustable)...")
df["abandono"] = build_target(df)

# --------------------------
# FEATURES
# --------------------------
num_features = [
    "freq_lectura_num", "freq_offline_num", "frecuencia_publica_num",
    "importancia_personalizacion_num", "interaccion_autores_num",
    "infl_recomendaciones_num", "importancia_gratis_num",
    "horas_lectura_digital_num", "proteccion_contenido_num",
    "util_notificaciones_num", "facilidad_descubrimiento_num",
    "importancia_descarga_num", "anuncios_aceptables_num",
    "valora_emergentes_num", "infl_resenas_num",
    "compartir_lectura_num", "importancia_diversidad_num"
]

bin_features = [
    "usa_plataformas_bin", "participa_desafios_bin", "paga_suscripcion_bin",
    "prefiere_interactivo_bin", "quiere_recomendaciones_bin"
] + list(molestias_df.columns)

cat_features = ["pref_formato"]  # la pasamos a OneHot

X = df[num_features + bin_features + cat_features]
y = df["abandono"].astype(int)

# Divisiones
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
)

# --------------------------
# PREPROCESAMIENTO
# --------------------------
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ("num", "passthrough", num_features + bin_features),
    ]
)

# --------------------------
# MODELOS Y GRIDS
# --------------------------
models_and_params = [
    (
        LogisticRegression(max_iter=200, class_weight="balanced", random_state=SEED),
        {
            "clf__C": [0.1, 1, 10],
            "clf__penalty": ["l2"],
            "clf__solver": ["lbfgs", "liblinear"]
        }
    ),
    (
        DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
        {
            "clf__max_depth": [3, 5, 7, None],
            "clf__min_samples_split": [2, 5, 10]
        }
    ),
    (
        RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1),
        {
            "clf__n_estimators": [100, 300],
            "clf__max_depth": [None, 5, 10],
            "clf__min_samples_split": [2, 5]
        }
    )
]

best_model = None
best_score = -np.inf
best_name = None
best_grid = None

skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

for base_clf, params in models_and_params:
    pipe = Pipeline(steps=[
        ("pre", preprocess),
        ("clf", base_clf)
    ])
    grid = GridSearchCV(
        pipe,
        param_grid=params,
        scoring="roc_auc",
        cv=skf,
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)
    mean_auc = grid.best_score_
    print(f"🔎 {base_clf.__class__.__name__} AUC (CV): {mean_auc:.4f}")

    if mean_auc > best_score:
        best_score = mean_auc
        best_model = grid.best_estimator_
        best_name = base_clf.__class__.__name__
        best_grid = grid

# --------------------------
# EVALUACIÓN FINAL
# --------------------------
print(f"\n🏆 Mejor modelo: {best_name} (CV AUC={best_score:.4f})")
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]
print("\n📊 Classification report (test):")
print(classification_report(y_test, y_pred, digits=4))
print(f"ROC-AUC (test): {roc_auc_score(y_test, y_proba):.4f}")

# --------------------------
# GUARDADO
# --------------------------
joblib.dump(best_model, MODEL_PATH)
metadata = {
    "best_model": best_name,
    "cv_auc": float(best_score),
    "features_numeric": num_features,
    "features_binary": bin_features,
    "features_categorical": cat_features,
    "target_rule": "Heurística cond_1 + cond_2 + cond_3 >= 2 (editable en train_model.py)",
}
with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"\n✅ Modelo guardado en: {MODEL_PATH}")
print(f"✅ Metadata guardada en: {META_PATH}")
