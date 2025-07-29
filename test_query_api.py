#!/usr/bin/env python3
"""
Simple pytest script to verify the query answering API works correctly
"""

import pytest
import requests
import time

# API base URL
BASE_URL = "http://localhost:8000"

class TestQueryAPI:
    """Simple tests for query answering API"""
    
    def test_api_health(self):
        """Test if the API is running"""
        try:
            response = requests.get(f"{BASE_URL}/docs", timeout=5)
            assert response.status_code == 200, "API is not responding correctly"
        except requests.exceptions.ConnectionError:
            pytest.skip("Could not connect to API. Make sure the server is running on http://localhost:8000")

    def test_query_answering_basic(self):
        """Test basic query answering"""
        test_query = "What is artificial intelligence?"
        
        try:
            response = requests.post(
                f"{BASE_URL}/answer-query",
                json={"query": test_query},
                timeout=30  # 30 second timeout for model loading
            )
            
            if response.status_code == 200:
                result = response.json()
                assert result["query"] == test_query
                assert "answer" in result
                assert len(result["answer"]) > 0
                assert result["style_applied"] == False  # No user ID provided
            elif response.status_code == 503:
                pytest.skip("Query answering service not available (T5 model failed to load)")
            else:
                pytest.fail(f"Unexpected response: {response.status_code} - {response.text}")
                
        except requests.exceptions.Timeout:
            pytest.skip("Request timed out. The model might still be loading.")

    def test_query_answering_with_user(self):
        """Test query answering with user personalization"""
        # First register a test user
        test_user = {
            "user_id": "test_user_pytest",
            "sample_text_corpus": "I write technical documentation and prefer clear, concise explanations with examples.",
            "natural_language_description": "A technical writer who prefers formal, clear language"
        }
        
        try:
            # Register user
            response = requests.post(f"{BASE_URL}/register", json=test_user, timeout=10)
            if response.status_code not in [200, 400]:  # 400 means user already exists
                pytest.skip("Failed to register test user")
            
            # Test query with user ID
            test_query = "How do neural networks work?"
            
            response = requests.post(
                f"{BASE_URL}/answer-query",
                json={
                    "query": test_query,
                    "user_id": test_user["user_id"]
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                assert result["query"] == test_query
                assert result["user_id"] == test_user["user_id"]
                assert "answer" in result
                assert len(result["answer"]) > 0
                assert result["style_applied"] == True  # User ID provided
            elif response.status_code == 503:
                pytest.skip("Query answering service not available")
            else:
                pytest.fail(f"Unexpected response: {response.status_code} - {response.text}")
                
        except requests.exceptions.Timeout:
            pytest.skip("Request timed out.")

    def test_edge_cases(self):
        """Test edge cases"""
        # Test empty query
        response = requests.post(f"{BASE_URL}/answer-query", json={"query": ""}, timeout=10)
        assert response.status_code == 400, f"Should reject empty query, got: {response.status_code}"
        
        # Test missing query field
        response = requests.post(f"{BASE_URL}/answer-query", json={}, timeout=10)
        assert response.status_code == 422, f"Should reject missing query field, got: {response.status_code}"

# Pytest fixture to wait for API
@pytest.fixture(scope="session", autouse=True)
def wait_for_api():
    """Wait for API to be ready"""
    max_retries = 10
    for i in range(max_retries):
        try:
            response = requests.get(f"{BASE_URL}/docs", timeout=5)
            if response.status_code == 200:
                print("✅ API is ready")
                return
        except requests.exceptions.ConnectionError:
            if i == 0:
                print("⏳ Waiting for API to start...")
            time.sleep(2)
    
    pytest.skip("API server not running. Start with: python main.py")

if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "-s"]) 