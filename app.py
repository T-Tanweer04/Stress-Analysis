from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

with open('stress_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        humidity = float(request.form.get('humidity'))
        temperature = float(request.form.get('temperature'))
        step_count = float(request.form.get('step_count'))

        input_features = np.array([[humidity, temperature, step_count]])
        prediction = model.predict(input_features)
        stress_level = 'High' if prediction[0] >= 1 else 'Low'

        return render_template('form.html', prediction_text=f'Predicted Stress Level: {stress_level}')
    
    except ValueError as e:
        return render_template('form.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
