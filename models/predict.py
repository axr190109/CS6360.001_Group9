import tensorflow as tf
import numpy as np
import random

# Load the models
runtime_model = tf.keras.models.load_model('runtime_model.keras')
size_model = tf.keras.models.load_model('size_model.keras')
error_model = tf.keras.models.load_model('error_model.keras')

def predict_runtime(features):
    """
    Predict the runtime based on input features.
    Args:
        features (list): Input features.
    Returns:
        float: Predicted runtime.
    """
    features = np.array(features).reshape(1, -1)
    return max(0, runtime_model.predict(features)[0][0])

def predict_result_size(features):
    """
    Predict the result size based on input features.
    Args:
        features (list): Input features.
    Returns:
        float: Predicted result size.
    """
    features = np.array(features).reshape(1, -1)
    return max(0, size_model.predict(features)[0][0])

def predict_error_class(features):
    """
    Predict the error class based on input features.
    Args:
        features (list): Input features.
    Returns:
        tuple: Predicted error class and its probabilities.
    """
    features = np.array(features).reshape(1, -1)
    probabilities = error_model.predict(features)[0]
    return np.argmax(probabilities), probabilities

def generate_example_features():
    """
    Generate example features for prediction.
    Returns:
        list: Example features.
    """
    num_joins = random.randint(0, 5)
    num_where_conditions = random.randint(0, 3)
    num_aggregates = random.randint(0, 2)
    num_subqueries = random.randint(0, 2)
    query_length = random.randint(50, 300)
    num_select_columns = random.randint(1, 10)
    num_distinct_clauses = random.randint(0, 1)
    presence_group_by = random.randint(0, 1)
    presence_order_by = random.randint(0, 1)
    nested_subquery_levels = random.randint(0, 2)
    return [
        num_joins, num_where_conditions, num_aggregates, num_subqueries, query_length,
        num_select_columns, num_distinct_clauses, presence_group_by, presence_order_by, nested_subquery_levels
    ]

if __name__ == "__main__":
    for _ in range(5):
        example_features = generate_example_features()
        print(f"Generated example features: {example_features}")
        runtime = predict_runtime(example_features)
        size = predict_result_size(example_features)
        error_class, probabilities = predict_error_class(example_features)
        print(f"Predicted runtime: {runtime:.2f}")
        print(f"Predicted result size: {size:.2f}")
        print(f"Predicted error class: {error_class} (Probabilities: {probabilities})\n")
