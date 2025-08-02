import pandas as pd
import streamlit as st
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def validate_users_format(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate that the users CSV has the required format.
    
    Args:
        df: Pandas DataFrame from uploaded users CSV
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if df.empty:
        return False, "The users file is empty."
    
    # Check if required columns exist
    required_columns = ['user_id', 'description']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    
    # Check if there are any empty values
    if df['user_id'].isna().any():
        return False, "The 'user_id' column contains empty values."
    
    if df['description'].isna().any():
        return False, "The 'description' column contains empty values."
    
    # Check if there are any duplicate user_ids
    if df['user_id'].duplicated().any():
        return False, "The 'user_id' column contains duplicate values."
    
    return True, ""


def validate_queries_format(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate that the queries CSV has the required format.
    
    Args:
        df: Pandas DataFrame from uploaded queries CSV
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if df.empty:
        return False, "The queries file is empty."
    
    # Check if required columns exist
    required_columns = ['user_id', 'question']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    
    # Check if there are any empty values
    if df['user_id'].isna().any():
        return False, "The 'user_id' column contains empty values."
    
    if df['question'].isna().any():
        return False, "The 'question' column contains empty values."
    
    return True, ""


def validate_csv_format(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate that the uploaded CSV has the required format (legacy function).
    
    Args:
        df: Pandas DataFrame from uploaded CSV
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if df.empty:
        return False, "The uploaded file is empty."
    
    # Check if 'query' column exists
    if 'query' not in df.columns:
        return False, "The CSV must contain a column named 'query'."
    
    # Check if there are any empty queries
    if df['query'].isna().any():
        return False, "The 'query' column contains empty values."
    
    # Check if there are any duplicate queries
    if df['query'].duplicated().any():
        return False, "The 'query' column contains duplicate values."
    
    return True, ""


def process_separate_csv_data(users_df: pd.DataFrame, queries_df: pd.DataFrame) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Process separate users and queries CSV data into the format expected by the API.
    
    Args:
        users_df: Pandas DataFrame with 'user_id' and 'description' columns
        queries_df: Pandas DataFrame with 'user_id' and 'question' columns
        
    Returns:
        Tuple of (users, queries) lists for API consumption
    """
    # Convert users to the expected format
    users = []
    for idx, row in users_df.iterrows():
        users.append({
            "user_id": row['user_id'],
            "description": row['description']
        })
    
    # Convert queries to the expected format
    queries = []
    for idx, row in queries_df.iterrows():
        queries.append({
            "user_id": row['user_id'],
            "question": row['question']
        })
    
    return users, queries


def process_csv_data(df: pd.DataFrame, user_description: str) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Process CSV data into the format expected by the API (legacy function).
    
    Args:
        df: Pandas DataFrame with 'query' column
        user_description: User's self-description
        
    Returns:
        Tuple of (users, queries) lists for API consumption
    """
    # Create a single user with the provided description
    users = [{
        "user_id": "demo_user",
        "description": user_description
    }]
    
    # Convert queries to the expected format
    queries = []
    for idx, row in df.iterrows():
        queries.append({
            "user_id": "demo_user",
            "question": row['query']
        })
    
    return users, queries


def create_new_results_dataframe(personalized_results: List[Dict]) -> pd.DataFrame:
    """
    Create a DataFrame from new API results format for display and download.
    
    Args:
        personalized_results: List of PersonalizedQueryResponse data from API
        
    Returns:
        Pandas DataFrame with results
    """
    results_data = []
    
    for result in personalized_results:
        results_data.append({
            'user_id': result.get('user_id', ''),
            'cluster_id': result.get('cluster_id', -1),
            'original_query': result.get('original_query', ''),
            'summarized_query': result.get('summarized_query', ''),
            'summarized_response': result.get('summarized_response', ''),
            'personalized_response': result.get('personalized_response', ''),
            'success': result.get('success', False),
            'error_message': result.get('error_message', '')
        })
    
    return pd.DataFrame(results_data)


def create_results_dataframe(
    clustered_queries: List[Dict], 
    personalized_rewrites: List[Dict]
) -> pd.DataFrame:
    """
    Create a DataFrame from API results for display and download (legacy function).
    
    Args:
        clustered_queries: List of clustered query data from API
        personalized_rewrites: List of personalized rewrite data from API
        
    Returns:
        Pandas DataFrame with results
    """
    results_data = []
    
    # Create a mapping of original queries to their personalized rewrites
    rewrite_map = {}
    for rewrite in personalized_rewrites:
        rewrite_map[rewrite['original_query']] = rewrite['personalized_query']
    
    # Process each cluster
    for cluster in clustered_queries:
        cluster_id = cluster['cluster_id']
        
        for query in cluster['queries']:
            original_query = query['question']
            rewritten_query = rewrite_map.get(original_query, "N/A")
            
            results_data.append({
                'cluster_id': cluster_id,
                'original_query': original_query,
                'rewritten_query': rewritten_query,
                'strategy': 'LLM - Reprompted'
            })
    
    return pd.DataFrame(results_data)


def format_cluster_display(cluster_data: Dict) -> str:
    """
    Format cluster data for display in Streamlit.
    
    Args:
        cluster_data: Cluster data from API
        
    Returns:
        Formatted string for display
    """
    cluster_id = cluster_data['cluster_id']
    query_count = len(cluster_data['queries'])
    return f"Cluster {cluster_id} ({query_count} prompts)"


def check_api_connection(api_client) -> bool:
    """
    Check if the API is accessible.
    
    Args:
        api_client: APIClient instance
        
    Returns:
        True if API is accessible, False otherwise
    """
    try:
        result = api_client.health_check()
        return True
    except Exception as e:
        logger.error(f"API connection check failed: {e}")
        return False 