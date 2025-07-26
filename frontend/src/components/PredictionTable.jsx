import React, { useEffect, useState } from 'react';

const PredictionTable = () => {
  const [predicciones, setPredicciones] = useState([]);

  useEffect(() => {
    fetch("http://localhost:5000/predicciones")
      .then((res) => res.json())
      .then((data) => setPredicciones(data))
      .catch((err) => console.error("Error al cargar predicciones:", err));
  }, []);

  return (
    <div>
      <h2>📊 Predicción de Abandono</h2>
      <table border="1">
        <thead>
          <tr>
            <th>#</th>
            <th>Clase</th>
            <th>Probabilidad</th>
            <th>Riesgo</th>
          </tr>
        </thead>
        <tbody>
          {predicciones.map((p) => (
            <tr key={p.id}>
              <td>{p.id}</td>
              <td>{p.clase}</td>
              <td>{p.probabilidad}</td>
              <td>{p.riesgo}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default PredictionTable;
