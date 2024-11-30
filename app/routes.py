from flask import Blueprint, render_template, request, jsonify
from models.predict import predict_runtime

# Create a Blueprint
main = Blueprint('main', __name__)

# Home route
@main.route('/')
def index():
    return render_template('index.html')

# Prediction route
@main.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from the form
        feature1 = float(request.form['feature1'])
        feature2 = float(request.form['feature2'])
        feature3 = float(request.form['feature3'])

        # Make a prediction
        prediction = predict_runtime([feature1, feature2, feature3])

        # Return the prediction as JSON
        return jsonify({'predicted_runtime': round(prediction, 2)})

    except Exception as e:
        return jsonify({'error': str(e)})
