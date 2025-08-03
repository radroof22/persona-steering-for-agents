import pandas as pd
from typing import Tuple, Dict, Any
from pathlib import Path
from backend.models.schema import User, Query


def load_mock_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load mock users and queries from parquet files.
    
    Returns:
        Tuple of (users_df, queries_df) DataFrames
    """
    data_dir = Path(__file__).parent.parent / "data"
    
    # Load parquet files
    users_df = pd.read_parquet(data_dir / "mock_users.parquet")
    queries_df = pd.read_parquet(data_dir / "mock_queries.parquet")
    
    return users_df, queries_df


def validate_data_format(users_df: pd.DataFrame, queries_df: pd.DataFrame) -> bool:
    """
    Validate that the data has the expected format.
    
    Args:
        users_df: DataFrame with user data
        queries_df: DataFrame with query data
        
    Returns:
        True if validation passes, raises ValueError otherwise
    """
    # Validate users DataFrame
    expected_user_columns = {"user_id", "description"}
    if set(users_df.columns) != expected_user_columns:
        raise ValueError(f"Users DataFrame must have columns: {expected_user_columns}")
    
    # Validate queries DataFrame
    expected_query_columns = {"user_id", "question"}
    if set(queries_df.columns) != expected_query_columns:
        raise ValueError(f"Queries DataFrame must have columns: {expected_query_columns}")
    
    # Check for missing values
    if users_df.isnull().any().any():
        raise ValueError("Users DataFrame contains missing values")
    
    if queries_df.isnull().any().any():
        raise ValueError("Queries DataFrame contains missing values")
    
    # Check that all query user_ids exist in users
    missing_users = set(queries_df["user_id"]) - set(users_df["user_id"])
    if missing_users:
        raise ValueError(f"Queries reference non-existent users: {missing_users}")
    
    return True


def merge_user_query_data(users_df: pd.DataFrame, queries_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge users and queries on user_id.
    
    Args:
        users_df: DataFrame with user data
        queries_df: DataFrame with query data
        
    Returns:
        Merged DataFrame with user descriptions and queries
    """
    # Validate data format first
    validate_data_format(users_df, queries_df)
    
    # Merge on user_id
    merged_df = queries_df.merge(users_df, on="user_id", how="left")
    
    return merged_df


def load_and_validate_data() -> pd.DataFrame:
    """
    Load, validate, and merge user and query data.
    
    Returns:
        Merged DataFrame with user descriptions and queries
    """
    users_df, queries_df = load_mock_data()
    return merge_user_query_data(users_df, queries_df)


def get_user_descriptions(users_df: pd.DataFrame) -> Dict[str, str]:
    """
    Create a mapping of user_id to user description.
    
    Args:
        users_df: DataFrame with user data
        
    Returns:
        Dictionary mapping user_id to description
    """
    return dict(zip(users_df["user_id"], users_df["description"])) 