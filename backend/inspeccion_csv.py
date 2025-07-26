# backend/inspeccion_csv.py

import pandas as pd
from pathlib import Path

# Lista de rutas posibles donde puede estar el archivo
possible_paths = [
    Path(__file__).resolve().parent.parent / "data" / "encuesta_usuarios.csv",
    Path(__file__).resolve().parent / "data" / "encuesta_usuarios.csv",
    Path(__file__).resolve().parent / "encuesta_usuarios.csv"
]

csv_path = None

# Buscar el archivo en las rutas posibles
for path in possible_paths:
    if path.exists():
        csv_path = path
        break

if not csv_path:
    print("❌ No se encontró el archivo 'encuesta_usuarios.csv' en ninguna de las rutas esperadas:")
    for path in possible_paths:
        print(f" - {path}")
else:
    print(f"✅ CSV encontrado en: {csv_path}")
    df = pd.read_csv(csv_path)

    # Mostrar columnas y valores únicos
    print("\n🔎 Columnas encontradas:")
    print(df.columns.tolist())

    print("\n🧪 Valores únicos por columna:")
    for col in df.columns:
        print(f"\n👉 {col}")
        print(df[col].dropna().unique())
