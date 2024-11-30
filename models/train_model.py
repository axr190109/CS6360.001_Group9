import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report
import logging

# Configure logging
logging.basicConfig(filename='train_model.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def log_metrics(metric_name, value):
    logging.info(f"{metric_name}: {value:.4f}")

# Load and preprocess the dataset
def load_data(filepath):
    """
    Load and preprocess data for training models.
    Args:
        filepath (str): Path to the CSV file.
    Returns:
        tuple: Split data for runtime, result size, and error class prediction models.
    """
    data = pd.read_csv(filepath)

    # Extract features
    X = data[['num_joins', 'num_where_conditions', 'num_aggregates', 'num_subqueries', 'query_length',
              'num_select_columns', 'num_distinct_clauses', 'presence_group_by', 'presence_order_by', 
              'nested_subquery_levels']]

    # Normalize features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Targets for runtime and result size
    y_runtime = (
        0.2 * data['num_joins'] +
        0.1 * data['num_where_conditions'] +
        0.15 * data['num_aggregates'] +
        0.25 * data['num_subqueries'] +
        0.005 * data['query_length']
    ).to_numpy()
    y_size = (
        50 * data['num_joins'] +
        30 * data['num_where_conditions'] +
        20 * data['num_aggregates'] +
        10 * data['num_subqueries'] +
        0.5 * data['query_length']
    ).to_numpy()

    # Error class (categorical)
    y_error_class = data['error_class']
    encoder = OneHotEncoder(sparse_output=False)
    y_error_class_encoded = encoder.fit_transform(y_error_class.values.reshape(-1, 1))

    return train_test_split(X_scaled, y_runtime, y_size, y_error_class_encoded, test_size=0.2, random_state=42)

# Train and evaluate models
def train_models(filepath):
    """
    Train models for runtime, result size, and error class prediction.
    Args:
        filepath (str): Path to the CSV file.
    """
    X_train, X_test, y_runtime_train, y_runtime_test, y_size_train, y_size_test, y_error_train, y_error_test = load_data(filepath)

    # Runtime prediction model
    runtime_model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)  # Regression output
    ])
    runtime_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    print("Training the runtime prediction model...")
    runtime_model.fit(X_train, y_runtime_train, epochs=20, batch_size=16, validation_data=(X_test, y_runtime_test))
    runtime_model.save('runtime_model.keras')
    print("Runtime prediction model training complete.")
    logging.info("Runtime prediction model training complete.")

    # Result size prediction model
    size_model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)  # Regression output
    ])
    size_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    print("Training the result size prediction model...")
    size_model.fit(X_train, y_size_train, epochs=20, batch_size=16, validation_data=(X_test, y_size_test))
    size_model.save('size_model.keras')
    print("Result size prediction model training complete.")
    logging.info("Result size prediction model training complete.")

    # Error class prediction model
    error_model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(y_error_train.shape[1], activation='softmax')  # Classification output
    ])
    error_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("Training the error class prediction model...")
    error_model.fit(X_train, y_error_train, epochs=20, batch_size=16, validation_data=(X_test, y_error_test))
    error_model.save('error_model.keras')
    print("Error class prediction model training complete.")
    logging.info("Error class prediction model training complete.")

    # Evaluate models
    evaluate_models(runtime_model, size_model, error_model, X_test, y_runtime_test, y_size_test, y_error_test)

def evaluate_models(runtime_model, size_model, error_model, X_test, y_runtime_test, y_size_test, y_error_test):
    """
    Evaluate the trained models.
    """
    print("\nEvaluating models...")

    # Evaluate runtime model
    y_runtime_pred = runtime_model.predict(X_test)
    runtime_mae = mean_absolute_error(y_runtime_test, y_runtime_pred)
    runtime_rmse = mean_squared_error(y_runtime_test, y_runtime_pred, squared=False)
    print(f"Runtime Model - MAE: {runtime_mae:.2f}")
    print(f"Runtime Model - RMSE: {runtime_rmse:.2f}")
    log_metrics("Runtime MAE", runtime_mae)
    log_metrics("Runtime RMSE", runtime_rmse)

    # Evaluate result size model
    y_size_pred = size_model.predict(X_test)
    size_mae = mean_absolute_error(y_size_test, y_size_pred)
    size_rmse = mean_squared_error(y_size_test, y_size_pred, squared=False)
    print(f"Size Model - MAE: {size_mae:.2f}")
    print(f"Size Model - RMSE: {size_rmse:.2f}")
    log_metrics("Size MAE", size_mae)
    log_metrics("Size RMSE", size_rmse)

    # Evaluate error class model
    y_error_pred = error_model.predict(X_test)
    y_error_pred_classes = np.argmax(y_error_pred, axis=1)
    y_error_test_classes = np.argmax(y_error_test, axis=1)
    error_accuracy = accuracy_score(y_error_test_classes, y_error_pred_classes)
    print(f"Error Class Model - Accuracy: {error_accuracy:.2f}")
    print("\nClassification Report:")
    report = classification_report(y_error_test_classes, y_error_pred_classes)
    print(report)
    log_metrics("Error Class Accuracy", error_accuracy)
    logging.info("\nClassification Report:\n" + report)

if __name__ == "__main__":
    train_models('../data/extracted_features.csv')
