import re
import pandas as pd

def extract_features(sql_query):
    """
    Extract features from an SQL query.
    Args:
        sql_query (str): SQL query string.
    Returns:
        dict: Dictionary of extracted features.
    """
    features = {
        "num_joins": len(re.findall(r'\bJOIN\b', sql_query, re.IGNORECASE)),
        "num_where_conditions": len(re.findall(r'\bWHERE\b', sql_query, re.IGNORECASE)),
        "num_aggregates": len(re.findall(r'\bCOUNT\b|\bSUM\b|\bAVG\b|\bMIN\b|\bMAX\b', sql_query, re.IGNORECASE)),
        "num_subqueries": len(re.findall(r'\bSELECT\b', sql_query, re.IGNORECASE)) - 1,
        "query_length": len(sql_query),
    }
    return features

def process_sql_file(filepath):
    """
    Process a file containing SQL queries and extract features for each query.
    Args:
        filepath (str): Path to the .sql file.
    Returns:
        pd.DataFrame: DataFrame containing features for each query.
    """
    with open(filepath, 'r') as file:
        queries = file.read().split(';')  # Split by semicolon to separate queries
        queries = [query.strip() for query in queries if query.strip()]  # Remove empty queries
    
    features_list = []
    for query in queries:
        features = extract_features(query)
        features_list.append(features)
    
    return pd.DataFrame(features_list)

if __name__ == "__main__":
    # Filepath to sample.sql
    sql_file = '../data/sample.sql'

    # Extract features and save to CSV
    df = process_sql_file(sql_file)
    print("Extracted features:")
    print(df)

    # Save features to a CSV file for model training
    df.to_csv('../data/extracted_features.csv', index=False)
