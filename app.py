from flask import Flask, render_template, request
import numpy as np
import joblib
import pickle

app = Flask(__name__)

# Load models
models = {
    "KNN": joblib.load("knn_model.pkl"),
    "Random Forest": joblib.load("rf_model.pkl"),
    "Decision Tree": joblib.load("dt_model.pkl")
}

# Load label encoder
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/homepage')
def homepage():
    return render_template('homepage.html')

@app.route('/ourteam')
def ourteam():
    return render_template('ourteam.html')

@app.route('/toolsandtech')
def toolsandtech():
    return render_template('toolsandtech.html')

@app.route('/input')
def input_page():
    return render_template('input.html', models=models.keys())

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read input values
        v1 = float(request.form['V1'])
        v2 = float(request.form['V2'])
        v3 = float(request.form['V3'])
        i1 = float(request.form['I1'])
        i2 = float(request.form['I2'])
        i3 = float(request.form['I3'])
        selected_model = request.form['model']

        features = np.array([[v1, v2, v3, i1, i2, i3]])

        # Get prediction
        model = models[selected_model]
        prediction_encoded = model.predict(features)
        prediction_label = le.inverse_transform(prediction_encoded)[0]

        return render_template('result.html',
                               model=selected_model,
                               prediction=prediction_label)

    except Exception as e:
        return render_template('result.html', error=f"Result: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)