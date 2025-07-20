export async function predecirUsuario(data) {
  try {
    const response = await fetch(`${process.env.REACT_APP_API_URL}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });

    if (!response.ok) throw new Error("Error en la predicción");
    return await response.json();
  } catch (error) {
    console.error("Error en predictorAPI:", error);
    return { error: true };
  }
}
