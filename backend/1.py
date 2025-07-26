import pickle

with open('model/abandono_model.pkl', 'rb') as f:
    modelo = pickle.load(f)

print("✅ Tipo de objeto cargado:", type(modelo))
