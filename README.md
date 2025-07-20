# Sistema de Predicción de Abandono de Usuarios

Este módulo permite predecir qué usuarios están en riesgo de abandonar Watpato, utilizando Machine Learning entrenado con datos reales de comportamiento guardados en la base de datos. El proceso incluye extracción de métricas, entrenamiento del modelo, predicción masiva y una API para predicción en tiempo real.

---

## Estructura del backend

```
backend-predictor/
├── app/
│   ├── __init__.py
│   ├── predictor.py              # Función predecir_abandono()
│   ├── utils.py                  # Funciones auxiliares (vacío por ahora)
│   └── sql/
│       └── metricas_usuario.sql  # Consulta para métricas de usuarios
├── extract_user_metrics.py       # Extrae métricas desde PostgreSQL
├── predict_api.py                # API Flask para predicción individual
├── predict_from_json.py          # Predicción masiva desde archivo JSON
├── train_model.py                # Entrena el modelo ML
├── requirements.txt              # Librerías necesarias
```

---

### 🛠 Requisitos

Antes de iniciar, instala las dependencias necesarias (una sola vez):

```bash
cd backend-predictor
pip install -r requirements.txt
```

---

### ⚙️ Paso 1 – Entrenar el modelo

Si aún no existe `app/model.pkl`, ejecuta el siguiente script para entrenar un modelo base:

```bash
python train_model.py
```

Esto generará el archivo `app/model.pkl` que será utilizado por la API y las predicciones.

---

### 📊 Paso 2 – Extraer métricas de usuarios

Este script se conecta a la base de datos PostgreSQL y extrae las métricas de comportamiento desde la tabla `user_action_logs`:

```bash
python extract_user_metrics.py
```

✅ Resultado: Se genera el archivo `user_metrics.json` con las métricas de cada usuario.

---

### 🔮 Paso 3 – Predecir abandono de forma masiva

Con el modelo entrenado y las métricas generadas, ejecuta:

```bash
python predict_from_json.py
```

✅ Esto generará:

* Salida en consola mostrando el riesgo por usuario.
* Un archivo `predicciones_abandono.json` con los resultados:

```json
[
  {
    "riesgo_abandono": "Alto",
    "probabilidad": 0.82,
    "user_id": 5
  },
  ...
]
```

---

### 🌐 Paso 4 – Usar la API Flask de predicción en tiempo real

Para hacer predicciones individuales vía API (ideal para frontend o herramientas externas), ejecuta:

```bash
python predict_api.py
```

La API quedará activa en:

```
POST http://localhost:5000/predict
```

#### 📥 Ejemplo de request:

```json
{
  "dias_desde_ultimo_login": 15,
  "total_capitulos_leidos": 20,
  "promedio_dias_entre_sesiones": 1.5,
  "capitulos_creados": 0,
  "seguidores": 2,
  "siguiendo": 4,
  "comentarios_realizados": 5,
  "tiempo_lectura_total": 500,
  "notificaciones_activadas": 1
}
```

#### 📤 Ejemplo de respuesta:

```json
{
  "riesgo_abandono": "Medio",
  "probabilidad": 0.56
}
```

---

### 🧠 Flujo General del Sistema

1. Los usuarios interactúan con Watpato (leer, comentar, etc.).
2. Esas acciones se almacenan en la tabla `user_action_logs`.
3. `extract_user_metrics.py` transforma los logs en métricas por usuario.
4. `train_model.py` entrena un modelo de abandono.
5. `predict_from_json.py` evalúa a todos los usuarios.
6. `predict_api.py` permite predicciones individuales vía API.

---

### ✅ Archivos importantes generados

| Archivo                      | Descripción                                 |
| ---------------------------- | ------------------------------------------- |
| `user_metrics.json`          | Métricas por usuario para predecir abandono |
| `app/model.pkl`              | Modelo entrenado con scikit-learn           |
| `predicciones_abandono.json` | Resultado de predicciones masivas           |

---

Este sistema es totalmente extensible y puede integrarse con dashboards en React, tareas programadas (cron jobs) o incluso sistemas de notificación push para prevenir abandono en tiempo real.
