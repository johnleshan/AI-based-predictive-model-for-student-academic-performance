from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model/model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    input_data = np.array(data).reshape(1, -1)
    prediction = model.predict(input_data)[0]
    return render_template('index.html', prediction_text=f'Predicted Performance: {"Pass" if prediction == 1 else "Fail"}')

if __name__ == '__main__':
    app.run(debug=True, port=4444)