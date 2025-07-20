import React from 'react';
import userMetrics from '../user_metrics.json';

const AdminDashboard = () => {
  return (
    <div>
      <h1>Dashboard de Riesgo de Abandono</h1>
      <table>
        <thead>
          <tr>
            <th>User ID</th>
            <th>Riesgo</th>
            <th>Probabilidad</th>
          </tr>
        </thead>
        <tbody>
          {userMetrics.map((user, index) => (
            <tr key={index}>
              <td>{user.username}</td>
              <td>
                {user.probabilidad_abandono > 0.5 ? 'Alto' : 'Bajo'}
              </td>
              <td>{(user.probabilidad_abandono * 100).toFixed(1)}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default AdminDashboard;
