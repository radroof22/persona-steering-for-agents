import pytest
import pandas as pd
from pathlib import Path
import tempfile
import os

from app.utils.loader import (
    load_mock_data, 
    validate_data_format, 
    merge_user_query_data,
    load_and_validate_data,
    get_user_descriptions
)


class TestDataLoader:
    """Test cases for data loading and validation."""
    
    def test_load_mock_data(self):
        """Test loading mock data from parquet files."""
        users_df, queries_df = load_mock_data()
        
        # Check that data is loaded
        assert isinstance(users_df, pd.DataFrame)
        assert isinstance(queries_df, pd.DataFrame)
        
        # Check that data is not empty
        assert len(users_df) > 0
        assert len(queries_df) > 0
        
        # Check column names
        assert set(users_df.columns) == {"user_id", "description"}
        assert set(queries_df.columns) == {"user_id", "question"}
    
    def test_validate_data_format_valid(self):
        """Test validation with valid data."""
        # Create valid test data
        users_df = pd.DataFrame({
            "user_id": ["user1", "user2"],
            "description": ["desc1", "desc2"]
        })
        
        queries_df = pd.DataFrame({
            "user_id": ["user1", "user2"],
            "question": ["question1", "question2"]
        })
        
        # Should not raise an exception
        assert validate_data_format(users_df, queries_df) is True
    
    def test_validate_data_format_invalid_columns(self):
        """Test validation with invalid column names."""
        # Create invalid test data
        users_df = pd.DataFrame({
            "user_id": ["user1"],
            "wrong_column": ["desc1"]
        })
        
        queries_df = pd.DataFrame({
            "user_id": ["user1"],
            "question": ["question1"]
        })
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="Users DataFrame must have columns"):
            validate_data_format(users_df, queries_df)
    
    def test_validate_data_format_missing_values(self):
        """Test validation with missing values."""
        # Create test data with missing values
        users_df = pd.DataFrame({
            "user_id": ["user1", "user2"],
            "description": ["desc1", None]
        })
        
        queries_df = pd.DataFrame({
            "user_id": ["user1", "user2"],
            "question": ["question1", "question2"]
        })
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="Users DataFrame contains missing values"):
            validate_data_format(users_df, queries_df)
    
    def test_validate_data_format_missing_users(self):
        """Test validation when queries reference non-existent users."""
        # Create test data with missing user reference
        users_df = pd.DataFrame({
            "user_id": ["user1"],
            "description": ["desc1"]
        })
        
        queries_df = pd.DataFrame({
            "user_id": ["user1", "user2"],  # user2 doesn't exist in users_df
            "question": ["question1", "question2"]
        })
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="Queries reference non-existent users"):
            validate_data_format(users_df, queries_df)
    
    def test_merge_user_query_data(self):
        """Test merging user and query data."""
        # Create test data
        users_df = pd.DataFrame({
            "user_id": ["user1", "user2"],
            "description": ["desc1", "desc2"]
        })
        
        queries_df = pd.DataFrame({
            "user_id": ["user1", "user2", "user1"],
            "question": ["question1", "question2", "question3"]
        })
        
        # Merge data
        merged_df = merge_user_query_data(users_df, queries_df)
        
        # Check merged result
        assert len(merged_df) == 3
        assert set(merged_df.columns) == {"user_id", "question", "description"}
        
        # Check that all queries have descriptions
        assert merged_df["description"].notna().all()
    
    def test_get_user_descriptions(self):
        """Test creating user description mapping."""
        # Create test data
        users_df = pd.DataFrame({
            "user_id": ["user1", "user2"],
            "description": ["desc1", "desc2"]
        })
        
        # Get descriptions mapping
        descriptions = get_user_descriptions(users_df)
        
        # Check mapping
        assert descriptions["user1"] == "desc1"
        assert descriptions["user2"] == "desc2"
        assert len(descriptions) == 2
    
    def test_load_and_validate_data_integration(self):
        """Test the complete data loading and validation pipeline."""
        # This test requires the mock data files to exist
        try:
            merged_df = load_and_validate_data()
            
            # Check that data is loaded and merged
            assert isinstance(merged_df, pd.DataFrame)
            assert len(merged_df) > 0
            assert set(merged_df.columns) == {"user_id", "question", "description"}
            
            # Check that all required columns are present
            assert merged_df["user_id"].notna().all()
            assert merged_df["question"].notna().all()
            assert merged_df["description"].notna().all()
            
            # Test with actual data characteristics
            unique_users = merged_df["user_id"].nunique()
            unique_questions = merged_df["question"].nunique()
            
            assert unique_users > 0
            assert unique_questions > 0
            
            # Check that we have some realistic data patterns
            # Some users should have multiple queries
            user_query_counts = merged_df["user_id"].value_counts()
            assert user_query_counts.max() > 1  # At least one user has multiple queries
            
            # Check that descriptions are meaningful
            descriptions = merged_df["description"].unique()
            assert len(descriptions) > 0
            for desc in descriptions:
                assert len(desc) > 10  # Descriptions should be reasonably long
            
        except FileNotFoundError:
            pytest.skip("Mock data files not found. Run create_mock_data.py first.")
    
    def test_mock_data_content_validation(self):
        """Test that the mock data contains expected content patterns."""
        try:
            users_df, queries_df = load_mock_data()
            
            # Check users data
            assert len(users_df) >= 10  # Should have at least 10 users
            assert all(users_df["user_id"].str.startswith("user_"))  # User IDs should follow pattern
            
            # Check that descriptions are diverse and meaningful
            descriptions = users_df["description"].tolist()
            assert len(set(descriptions)) > 5  # Should have diverse descriptions
            
            # Check queries data
            assert len(queries_df) >= 15  # Should have at least 15 queries
            assert all(queries_df["user_id"].str.startswith("user_"))  # User IDs should follow pattern
            
            # Check that queries are diverse
            questions = queries_df["question"].tolist()
            assert len(set(questions)) > 10  # Should have diverse questions
            
            # Check that some users have multiple queries
            user_query_counts = queries_df["user_id"].value_counts()
            assert user_query_counts.max() > 1  # At least one user has multiple queries
            
        except FileNotFoundError:
            pytest.skip("Mock data files not found. Run create_mock_data.py first.")
    
    def test_data_consistency_validation(self):
        """Test that the mock data is internally consistent."""
        try:
            users_df, queries_df = load_mock_data()
            
            # Check that all query user_ids exist in users
            query_user_ids = set(queries_df["user_id"])
            user_user_ids = set(users_df["user_id"])
            
            missing_users = query_user_ids - user_user_ids
            assert len(missing_users) == 0, f"Queries reference non-existent users: {missing_users}"
            
            # Check that all users have descriptions
            assert users_df["description"].notna().all()
            
            # Check that all queries have questions
            assert queries_df["question"].notna().all()
            
            # Check that user IDs are unique in users table
            assert users_df["user_id"].is_unique
            
        except FileNotFoundError:
            pytest.skip("Mock data files not found. Run create_mock_data.py first.") 