// src/components/AdminPanel.jsx
import React, { useEffect, useState } from "react";
import PredictionTable from "./PredictionTable";

const AdminPanel = () => {
  const [predictions, setPredictions] = useState([]);

  useEffect(() => {
    const fetchPredictions = async () => {
      try {
        const response = await fetch("http://127.0.0.1:5000/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            instances: [
              {
                frecuencia_lectura: 2,
                horas_lectura: 1.5,
                quiere_recomendaciones: 0,
              },
              {
                frecuencia_lectura: 5,
                horas_lectura: 3.2,
                quiere_recomendaciones: 1,
              },
              {
                frecuencia_lectura: 1,
                horas_lectura: 0.2,
                quiere_recomendaciones: 0,
              }
            ],
          }),
        });

        const data = await response.json();
        setPredictions(data.predictions || []);
      } catch (error) {
        console.error("❌ Error al obtener predicciones:", error);
      }
    };

    fetchPredictions();
  }, []);

  return (
    <div>
      <h1><strong>Watpato: Panel de Retención</strong></h1>
      <h2>📊 Predicción de Abandono</h2>
      <PredictionTable predictions={predictions} />
    </div>
  );
};

export default AdminPanel;
