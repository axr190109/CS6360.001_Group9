from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import os

# Initialize Flask app and specify the template folder path
app = Flask(__name__, template_folder=os.path.join("app", "templates"))

# Load the trained model
model = tf.keras.models.load_model('models/runtime_model.keras')

def predict_runtime(features):
    """
    Predict the runtime based on input features.

    Args:
        features (list or array): Input features for prediction.

    Returns:
        float: Predicted runtime (non-negative).
    """
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    return max(0, prediction[0][0])

@app.route('/')
def index():
    """
    Render the homepage with the input form.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle form submission, predict runtime, and render results.
    """
    try:
        # Extract input features from the form
        features = [
            int(request.form['num_joins']),
            int(request.form['num_where_conditions']),
            int(request.form['num_aggregates']),
            int(request.form['num_subqueries']),
            int(request.form['query_length']),
            int(request.form['num_select_columns']),
            int(request.form['num_distinct_clauses']),
            int(request.form['presence_group_by']),
            int(request.form['presence_order_by']),
            int(request.form['nested_subquery_levels'])
        ]

        # Predict runtime
        predicted_runtime = predict_runtime(features)

        return render_template(
            'result.html',
            features=features,
            predicted_runtime=predicted_runtime
        )
    except Exception as e:
        return f"Error: {str(e)}", 400

if __name__ == '__main__':
    app.run(debug=True)
