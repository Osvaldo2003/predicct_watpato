-- Consulta para obtener métricas de abandono por usuario
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
