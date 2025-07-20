import React from "react";

const getColor = (riesgo) => {
  switch (riesgo) {
    case "Alto": return "#f87171";
    case "Medio": return "#facc15";
    case "Bajo": return "#4ade80";
    default: return "#e5e7eb";
  }
};

const UserRiskTable = ({ data }) => {
  return (
    <table style={{ width: "100%", borderCollapse: "collapse" }}>
      <thead>
        <tr>
          <th>User ID</th>
          <th>Riesgo</th>
          <th>Probabilidad</th>
        </tr>
      </thead>
      <tbody>
        {data.map((usuario, index) => (
          <tr key={index}>
            <td>{usuario.user_id}</td>
            <td style={{ color: getColor(usuario.riesgo_abandono), fontWeight: "bold" }}>
              {usuario.riesgo_abandono}
            </td>
            <td>{usuario.probabilidad}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
};

export default UserRiskTable;
