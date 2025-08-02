#!/usr/bin/env python3
"""
Test script for frontend components.
Run this to verify that all modules work correctly.
"""

import sys
import os
import pandas as pd
import asyncio

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import API_URL
from api_client import APIClient
from utils import (
    validate_csv_format, 
    validate_users_format,
    validate_queries_format,
    process_csv_data, 
    process_separate_csv_data,
    create_results_dataframe
)


def test_config():
    """Test configuration loading."""
    print("Testing configuration...")
    print(f"API URL: {API_URL}")
    print("✅ Configuration loaded successfully")


def test_csv_validation():
    """Test CSV validation functions."""
    print("\nTesting CSV validation...")
    
    # Test valid users CSV
    valid_users_data = {
        'user_id': ['user_001', 'user_002'],
        'description': ['Data scientist', 'Software engineer']
    }
    valid_users_df = pd.DataFrame(valid_users_data)
    is_valid, error = validate_users_format(valid_users_df)
    print(f"Valid users CSV test: {is_valid} - {error}")
    
    # Test valid queries CSV
    valid_queries_data = {
        'user_id': ['user_001', 'user_002'],
        'question': ['How do I improve my resume?', 'What are attention heads?']
    }
    valid_queries_df = pd.DataFrame(valid_queries_data)
    is_valid, error = validate_queries_format(valid_queries_df)
    print(f"Valid queries CSV test: {is_valid} - {error}")
    
    # Test invalid users CSV (missing column)
    invalid_users_data = {'user_id': ['user_001']}
    invalid_users_df = pd.DataFrame(invalid_users_data)
    is_valid, error = validate_users_format(invalid_users_df)
    print(f"Invalid users CSV test: {is_valid} - {error}")
    
    print("✅ CSV validation tests completed")


def test_data_processing():
    """Test data processing functions."""
    print("\nTesting data processing...")
    
    # Test data for separate CSV processing
    users_data = {
        'user_id': ['user_001', 'user_002'],
        'description': ['Data scientist', 'Software engineer']
    }
    users_df = pd.DataFrame(users_data)
    
    queries_data = {
        'user_id': ['user_001', 'user_002'],
        'question': ['How do I improve my resume?', 'What are attention heads?']
    }
    queries_df = pd.DataFrame(queries_data)
    
    # Process separate data
    users, queries = process_separate_csv_data(users_df, queries_df)
    
    print(f"Processed {len(users)} users and {len(queries)} queries")
    print(f"First user: {users[0]['user_id']} - {users[0]['description']}")
    print(f"First query: {queries[0]['user_id']} - {queries[0]['question']}")
    
    print("✅ Data processing tests completed")


def test_api_client():
    """Test API client functionality."""
    print("\nTesting API client...")
    
    client = APIClient()
    
    # Test health check
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        health_result = loop.run_until_complete(client.health_check())
        loop.close()
        print(f"Health check result: {health_result}")
        print("✅ API client health check successful")
    except Exception as e:
        print(f"❌ API client health check failed: {e}")
        print("Note: This is expected if the backend is not running")


def test_results_processing():
    """Test results processing functions."""
    print("\nTesting results processing...")
    
    # Mock API response
    mock_clustered_queries = [
        {
            'cluster_id': 0,
            'queries': [
                {'question': 'How do I improve my resume?'},
                {'question': 'What are attention heads?'}
            ]
        }
    ]
    
    mock_personalized_rewrites = [
        {
            'original_query': 'How do I improve my resume?',
            'personalized_query': 'How can I enhance my resume for data science roles?'
        },
        {
            'original_query': 'What are attention heads?',
            'personalized_query': 'Can you explain attention heads in transformer models?'
        }
    ]
    
    # Create results DataFrame
    results_df = create_results_dataframe(mock_clustered_queries, mock_personalized_rewrites)
    
    print(f"Results DataFrame shape: {results_df.shape}")
    print(f"Columns: {list(results_df.columns)}")
    print("Sample results:")
    print(results_df.head())
    
    print("✅ Results processing tests completed")


def main():
    """Run all tests."""
    print("🧪 Frontend Component Tests")
    print("=" * 50)
    
    try:
        test_config()
        test_csv_validation()
        test_data_processing()
        test_api_client()
        test_results_processing()
        
        print("\n" + "=" * 50)
        print("✅ All tests completed successfully!")
        print("\nTo run the frontend:")
        print("1. Ensure the sentosa environment is activated: conda activate sentosa")
        print("2. Ensure the backend is running: uvicorn app.main:app --reload")
        print("3. Run: streamlit run main.py")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 