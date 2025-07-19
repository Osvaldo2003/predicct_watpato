import psycopg2
import pandas as pd
import json
from datetime import datetime

conn = psycopg2.connect(
    dbname="postgres",
    user="postgres",
    password="iTtL2du6fNLrpUTizeOO",
    host="watpato.c9zcs6yb7c4x.us-east-1.rds.amazonaws.com",
    port="5432"
)

query = '''
SELECT
    user_id,
    MAX(timestamp) FILTER (WHERE action = 'inicio_sesion') AS ultima_sesion,
    COUNT(*) FILTER (WHERE action = 'lectura_capitulo') AS total_capitulos_leidos,
    COUNT(*) FILTER (WHERE action LIKE 'comentario_%') AS comentarios_realizados,
    COUNT(*) FILTER (WHERE action IN ('me_gusta_capitulo', 'me_gusta_libro')) AS likes_dados,
    COUNT(*) FILTER (WHERE action IN ('seguimiento_libro', 'seguimiento_usuario')) AS seguimientos,
    SUM(CASE 
        WHEN metadata ? 'tiempo_lectura' THEN (metadata->>'tiempo_lectura')::INT 
        ELSE 0 
    END) FILTER (WHERE action = 'lectura_capitulo') AS tiempo_lectura_total,
    COUNT(*) FILTER (WHERE action = 'notificacion_clicada') AS clics_notificacion
FROM user_action_logs
GROUP BY user_id;
'''

df = pd.read_sql_query(query, conn)

df['dias_desde_ultimo_login'] = df['ultima_sesion'].apply(
    lambda x: (datetime.utcnow() - x).days if x else 999
)

df['promedio_dias_entre_sesiones'] = df['dias_desde_ultimo_login'] / (df['total_capitulos_leidos'] + 1)

df['capitulos_creados'] = 0
df['seguidores'] = df['seguimientos'] // 2
df['siguiendo'] = df['seguimientos']
df['notificaciones_activadas'] = df['clics_notificacion'].apply(lambda x: 1 if x > 0 else 0)

final_df = df[[
    'user_id',
    'dias_desde_ultimo_login',
    'total_capitulos_leidos',
    'promedio_dias_entre_sesiones',
    'capitulos_creados',
    'seguidores',
    'siguiendo',
    'comentarios_realizados',
    'tiempo_lectura_total',
    'notificaciones_activadas'
]]

final_df.to_json("user_metrics.json", orient='records', lines=False, indent=2)

print("✅ user_metrics.json exportado correctamente.")
conn.close()
