from flask import Flask, request, jsonify
from flask_cors import CORS
from app.predictor import predecir_abandono

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        resultado = predecir_abandono(data)
        return jsonify(resultado)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
