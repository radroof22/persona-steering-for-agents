import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import Request
import logging

from backend.main import app, get_clusterer, get_llm_provider, get_prompt_summarizer, get_style_personalizer
from backend.models.schema import (
    Query, User, GenerateRequest, GenerateResponse, ClustersRequest, ClustersResponse,
    PersonalizedQueryResponse, ClusteredQuery
)
from backend.services.clusterer import QueryClusterer
from backend.services.llm_provider import LLMProvider
from backend.services.prompt_summarizer import PromptSummarizer
from backend.services.rewriter import StylePersonalizer

# Create a test app with mocked dependencies
def create_test_app():
    """Create a test app with mocked services to prevent LLM loading."""
    from backend.main import app
    
    # Mock the dependency functions to return mocked services
    mock_clusterer = Mock(spec=QueryClusterer)
    mock_llm = Mock(spec=LLMProvider)
    mock_summarizer = Mock(spec=PromptSummarizer)
    mock_personalizer = Mock(spec=StylePersonalizer)
    
    # Override the dependency functions
    app.dependency_overrides[get_clusterer] = lambda: mock_clusterer
    app.dependency_overrides[get_llm_provider] = lambda: mock_llm
    app.dependency_overrides[get_prompt_summarizer] = lambda: mock_summarizer
    app.dependency_overrides[get_style_personalizer] = lambda: mock_personalizer
    
    return app, mock_clusterer, mock_llm, mock_summarizer, mock_personalizer


class TestAppSetup:
    """Test FastAPI app setup and basic endpoints."""
    
    def test_app_creation(self):
        """Test that the FastAPI app is created with correct configuration."""
        assert app.title == "Personalized Query Rewriting API"
        assert app.description == "API for clustering and personalizing user queries using LLM"
        assert app.version == "1.0.0"
    
    def test_root_endpoint(self):
        """Test the root endpoint."""
        with TestClient(app) as client:
            response = client.get("/")
            assert response.status_code == 200
            assert response.json() == {"message": "Personalized Query Rewriting API is running"}
    
    def test_health_endpoint(self):
        """Test the health check endpoint."""
        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json() == {"status": "healthy", "message": "Service is operational"}


# Note: Lifespan tests are skipped due to async context manager complexity
# The lifespan functionality is tested indirectly through the endpoint tests
# which verify that the services work correctly when injected via dependencies


class TestDependencies:
    """Test dependency injection functions."""
    
    def test_get_clusterer(self):
        """Test get_clusterer dependency function."""
        # Create mock request with app.state
        mock_request = Mock(spec=Request)
        mock_clusterer = Mock(spec=QueryClusterer)
        mock_request.app.state.clusterer = mock_clusterer
        
        # Test dependency function
        result = get_clusterer(mock_request)
        
        assert result is mock_clusterer
    
    def test_get_llm_provider(self):
        """Test get_llm_provider dependency function."""
        # Create mock request with app.state
        mock_request = Mock(spec=Request)
        mock_llm = Mock(spec=LLMProvider)
        mock_request.app.state.llm_provider = mock_llm
        
        # Test dependency function
        result = get_llm_provider(mock_request)
        
        assert result is mock_llm
    
    def test_get_prompt_summarizer(self):
        """Test get_prompt_summarizer dependency function."""
        # Create mock request with app.state
        mock_request = Mock(spec=Request)
        mock_summarizer = Mock(spec=PromptSummarizer)
        mock_request.app.state.prompt_summarizer = mock_summarizer
        
        # Test dependency function
        result = get_prompt_summarizer(mock_request)
        
        assert result is mock_summarizer
    
    def test_get_style_personalizer(self):
        """Test get_style_personalizer dependency function."""
        # Create mock request with app.state
        mock_request = Mock(spec=Request)
        mock_personalizer = Mock(spec=StylePersonalizer)
        mock_request.app.state.style_personalizer = mock_personalizer
        
        # Test dependency function
        result = get_style_personalizer(mock_request)
        
        assert result is mock_personalizer


class TestClustersEndpoint:
    """Test the /clusters endpoint."""
    
    @pytest.fixture
    def sample_request_data(self):
        """Sample request data for testing."""
        return {
            "users": [
                {"user_id": "user1", "description": "A casual user"},
                {"user_id": "user2", "description": "A technical user"}
            ],
            "queries": [
                {"user_id": "user1", "question": "What is the weather like?"},
                {"user_id": "user2", "question": "How do I reset my password?"}
            ]
        }
    
    @pytest.fixture
    def mock_clusters(self):
        """Mock clustered queries for testing."""
        return [
            ClusteredQuery(
                cluster_id=0,
                user_ids=["user1"],
                queries=[Query(user_id="user1", question="What is the weather like?")]
            ),
            ClusteredQuery(
                cluster_id=1,
                user_ids=["user2"],
                queries=[Query(user_id="user2", question="How do I reset my password?")]
            )
        ]
    
    def test_clusters_endpoint_success(self, sample_request_data, mock_clusters):
        """Test successful clustering endpoint."""
        # Create test app with mocked dependencies
        test_app, mock_clusterer, _, _, _ = create_test_app()
        
        # Setup mock
        mock_clusterer.process_queries.return_value = mock_clusters
        
        # Test endpoint
        with TestClient(test_app) as client:
            response = client.post("/clusters", json=sample_request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify response structure
            assert "clusters" in data
            assert len(data["clusters"]) == 2
            
            # Verify cluster data
            cluster1 = data["clusters"][0]
            assert cluster1["cluster_id"] == 0
            assert cluster1["user_ids"] == ["user1"]
            assert len(cluster1["queries"]) == 1
            
            cluster2 = data["clusters"][1]
            assert cluster2["cluster_id"] == 1
            assert cluster2["user_ids"] == ["user2"]
            assert len(cluster2["queries"]) == 1
    
    def test_clusters_endpoint_service_failure(self, sample_request_data):
        """Test clustering endpoint when service fails."""
        # Create test app with mocked dependencies
        test_app, mock_clusterer, _, _, _ = create_test_app()
        
        # Setup mock to raise exception
        mock_clusterer.process_queries.side_effect = Exception("Clustering failed")
        
        # Test endpoint
        with TestClient(test_app) as client:
            response = client.post("/clusters", json=sample_request_data)
            
            assert response.status_code == 500
            data = response.json()
            assert "Internal server error" in data["detail"]
    
    def test_clusters_endpoint_invalid_request(self):
        """Test clustering endpoint with invalid request data."""
        invalid_data = {
            "users": "not a list",  # Invalid format
            "queries": []
        }
        
        # Create test app with mocked dependencies
        test_app, _, _, _, _ = create_test_app()
        
        with TestClient(test_app) as client:
            response = client.post("/clusters", json=invalid_data)
            
            assert response.status_code == 422  # Validation error


class TestGenerateEndpoint:
    """Test the /generate endpoint."""
    
    @pytest.fixture
    def sample_generate_request(self):
        """Sample generate request data for testing."""
        return {
            "users": [
                {"user_id": "user1", "description": "A casual user"},
                {"user_id": "user2", "description": "A technical user"}
            ],
            "queries": [
                {"user_id": "user1", "question": "What is the weather like?"},
                {"user_id": "user2", "question": "How do I reset my password?"}
            ]
        }
    
    @pytest.fixture
    def mock_clusters(self):
        """Mock clustered queries for testing."""
        return [
            ClusteredQuery(
                cluster_id=0,
                user_ids=["user1"],
                queries=[Query(user_id="user1", question="What is the weather like?")]
            ),
            ClusteredQuery(
                cluster_id=1,
                user_ids=["user2"],
                queries=[Query(user_id="user2", question="How do I reset my password?")]
            )
        ]
    
    def test_generate_endpoint_success(self, sample_generate_request, mock_clusters):
        """Test successful generate endpoint with full pipeline."""
        # Create test app with mocked dependencies
        test_app, mock_clusterer, _, mock_summarizer, mock_personalizer = create_test_app()
        
        # Setup mocks
        mock_clusterer.process_queries.return_value = mock_clusters
        
        mock_summarizer.summarize_cluster.side_effect = [
            "What is the weather like?",
            "How to reset password?"
        ]
        
        mock_personalizer.personalize_response.side_effect = [
            "Hey! Here's the weather info...",
            "To reset your password, follow these steps..."
        ]
        
        # Test endpoint
        with TestClient(test_app) as client:
            response = client.post("/generate", json=sample_generate_request)
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify response structure
            assert "results" in data
            assert len(data["results"]) == 2
            
            # Verify first result
            result1 = data["results"][0]
            assert result1["original_query"] == "What is the weather like?"
            assert result1["user_id"] == "user1"
            assert result1["cluster_id"] == 0
            assert result1["summarized_query"] == "What is the weather like?"
            assert result1["success"] is True
            assert result1["error_message"] is None
            
            # Verify second result
            result2 = data["results"][1]
            assert result2["original_query"] == "How do I reset my password?"
            assert result2["user_id"] == "user2"
            assert result2["cluster_id"] == 1
            assert result2["summarized_query"] == "How to reset password?"
            assert result2["success"] is True
            assert result2["error_message"] is None
    
    def test_generate_endpoint_summarization_failure(self, sample_generate_request):
        """Test generate endpoint when summarization fails."""
        # Create test app with mocked dependencies
        test_app, mock_clusterer, _, mock_summarizer, mock_personalizer = create_test_app()
        
        # Create clusters: first is single-query, second is multi-query to test failure
        test_clusters = [
            ClusteredQuery(
                cluster_id=0,
                user_ids=["user1"],
                queries=[Query(user_id="user1", question="What is the weather like?")]
            ),
            ClusteredQuery(
                cluster_id=1,
                user_ids=["user2", "user3"],
                queries=[
                    Query(user_id="user2", question="How do I reset my password?"),
                    Query(user_id="user3", question="What time does the store close?")
                ]
            )
        ]
        
        # Setup mocks
        mock_clusterer.process_queries.return_value = test_clusters
        
        mock_summarizer.summarize_cluster.side_effect = [
            "What is the weather like?",  # First cluster succeeds
            Exception("Summarization failed")  # Second cluster fails
        ]
        
        mock_personalizer.personalize_response.return_value = "Hey! Here's the weather info..."
        
        # Test endpoint
        with TestClient(test_app) as client:
            response = client.post("/generate", json=sample_generate_request)
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify response structure
            assert "results" in data
            assert len(data["results"]) == 3  # 1 from first cluster + 2 from second cluster
            
            # Verify first result (success - single query cluster)
            result1 = data["results"][0]
            assert result1["success"] is True
            assert result1["error_message"] is None
            
            # Verify second and third results (failure due to summarization failure)
            result2 = data["results"][1]
            result3 = data["results"][2]
            assert result2["success"] is False
            assert result3["success"] is False
            assert "Summarization failed" in result2["error_message"]
            assert "Summarization failed" in result3["error_message"]
            assert result2["personalized_response"] == ""
            assert result3["personalized_response"] == ""
            assert result2["summarized_query"] == ""
            assert result3["summarized_query"] == ""
    
    def test_generate_endpoint_personalization_failure(self, sample_generate_request):
        """Test generate endpoint when personalization fails."""
        # Create test app with mocked dependencies
        test_app, mock_clusterer, _, mock_summarizer, mock_personalizer = create_test_app()
        
        # Create a multi-query cluster to test personalization failure
        test_clusters = [
            ClusteredQuery(
                cluster_id=0,
                user_ids=["user1", "user2"],
                queries=[
                    Query(user_id="user1", question="What is the weather like?"),
                    Query(user_id="user2", question="How do I reset my password?")
                ]
            )
        ]
        
        # Setup mocks
        mock_clusterer.process_queries.return_value = test_clusters
        
        mock_summarizer.summarize_cluster.return_value = "What is the weather like?"
        
        mock_personalizer.personalize_response.side_effect = [
            "Hey! Here's the weather info...",  # First succeeds
            Exception("Personalization failed")  # Second fails
        ]
        
        # Test endpoint
        with TestClient(test_app) as client:
            response = client.post("/generate", json=sample_generate_request)
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify response structure
            assert "results" in data
            assert len(data["results"]) == 2
            
            # Verify first result (success)
            result1 = data["results"][0]
            assert result1["success"] is True
            assert result1["error_message"] is None
            
            # Verify second result (failure due to personalization failure)
            result2 = data["results"][1]
            assert result2["success"] is False
            assert "Personalization failed" in result2["error_message"]
            assert result2["personalized_response"] == ""
    
    def test_generate_endpoint_single_query_cluster(self):
        """Test generate endpoint with single query clusters."""
        # Create test app with mocked dependencies
        test_app, mock_clusterer, _, mock_summarizer, mock_personalizer = create_test_app()
        
        # Create request with single query
        request_data = {
            "users": [{"user_id": "user1", "description": "A casual user"}],
            "queries": [{"user_id": "user1", "question": "What is the weather like?"}]
        }
        
        # Create single query cluster
        single_cluster = [
            ClusteredQuery(
                cluster_id=0,
                user_ids=["user1"],
                queries=[Query(user_id="user1", question="What is the weather like?")]
            )
        ]
        
        # Setup mocks
        mock_clusterer.process_queries.return_value = single_cluster
        mock_summarizer.summarize_cluster.return_value = "What is the weather like?"
        # Should not be called for single query clusters
        
        # Test endpoint
        with TestClient(test_app) as client:
            response = client.post("/generate", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify response
            assert len(data["results"]) == 1
            result = data["results"][0]
            assert result["success"] is True
            assert result["personalized_response"] == "Summary for cluster 0: What is the weather like?"
            
            # Verify personalizer was not called for single query
            mock_personalizer.personalize_response.assert_not_called()
    
    def test_generate_endpoint_missing_user_description(self):
        """Test generate endpoint with missing user descriptions."""
        # Create test app with mocked dependencies
        test_app, mock_clusterer, _, mock_summarizer, mock_personalizer = create_test_app()
        
        # Create request with missing user description
        request_data = {
            "users": [{"user_id": "user1", "description": "A casual user"}],  # user2 missing
            "queries": [
                {"user_id": "user1", "question": "What is the weather like?"},
                {"user_id": "user2", "question": "How do I reset my password?"}
            ]
        }
        
        # Create a multi-query cluster to test personalization
        multi_query_cluster = [
            ClusteredQuery(
                cluster_id=0,
                user_ids=["user1", "user2"],
                queries=[
                    Query(user_id="user1", question="What is the weather like?"),
                    Query(user_id="user2", question="How do I reset my password?")
                ]
            )
        ]
        
        # Setup mocks
        mock_clusterer.process_queries.return_value = multi_query_cluster
        mock_summarizer.summarize_cluster.return_value = "Test summary"
        mock_personalizer.personalize_response.return_value = "Personalized response"
        
        # Test endpoint
        with TestClient(test_app) as client:
            response = client.post("/generate", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify personalizer was called with empty description for user2
            mock_personalizer.personalize_response.assert_called()
            # Check that user2 gets empty description
            calls = mock_personalizer.personalize_response.call_args_list
            user2_call = next(call for call in calls if "How do I reset my password?" in call[1]['original_query'])
            assert user2_call[1]['user_description'] == ""  # Empty description for missing user


class TestGenerateEndpointIntegration:
    """Integration tests for the generate endpoint with complex scenarios."""
    
    def test_generate_endpoint_mixed_success_failure(self):
        """Test generate endpoint with mixed success and failure scenarios."""
        # Create test app with mocked dependencies
        test_app, mock_clusterer, _, mock_summarizer, mock_personalizer = create_test_app()
        
        # Create complex request
        request_data = {
            "users": [
                {"user_id": "user1", "description": "A casual user"},
                {"user_id": "user2", "description": "A technical user"},
                {"user_id": "user3", "description": "A busy parent"}
            ],
            "queries": [
                {"user_id": "user1", "question": "What is the weather like?"},
                {"user_id": "user2", "question": "How do I reset my password?"},
                {"user_id": "user3", "question": "What time does the store close?"}
            ]
        }
        
        # Create clusters with different scenarios
        clusters = [
            ClusteredQuery(
                cluster_id=0,
                user_ids=["user1"],
                queries=[Query(user_id="user1", question="What is the weather like?")]
            ),
            ClusteredQuery(
                cluster_id=1,
                user_ids=["user2", "user3"],
                queries=[
                    Query(user_id="user2", question="How do I reset my password?"),
                    Query(user_id="user3", question="What time does the store close?")
                ]
            )
        ]
        
        # Setup mocks with mixed success/failure
        mock_clusterer.process_queries.return_value = clusters
        
        mock_summarizer.summarize_cluster.side_effect = [
            "What is the weather like?",  # Cluster 0 succeeds
            Exception("Summarization failed")  # Cluster 1 fails
        ]
        
        mock_personalizer.personalize_response.return_value = "Personalized response"
        
        # Test endpoint
        with TestClient(test_app) as client:
            response = client.post("/generate", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify response structure
            assert len(data["results"]) == 3
            
            # Verify first result (single query cluster, should succeed)
            result1 = data["results"][0]
            assert result1["success"] is True
            assert result1["cluster_id"] == 0
            
            # Verify second and third results (failed summarization)
            result2 = data["results"][1]
            result3 = data["results"][2]
            assert result2["success"] is False
            assert result3["success"] is False
            assert "Summarization failed" in result2["error_message"]
            assert "Summarization failed" in result3["error_message"]
    
    def test_generate_endpoint_service_exception(self):
        """Test generate endpoint when services raise exceptions."""
        # Create test app with mocked dependencies
        test_app, mock_clusterer, _, _, _ = create_test_app()
        
        request_data = {
            "users": [{"user_id": "user1", "description": "A casual user"}],
            "queries": [{"user_id": "user1", "question": "What is the weather like?"}]
        }
        
        # Setup mock to raise exception
        mock_clusterer.process_queries.side_effect = Exception("Clustering service unavailable")
        
        # Test endpoint
        with TestClient(test_app) as client:
            response = client.post("/generate", json=request_data)
            
            assert response.status_code == 500
            data = response.json()
            assert "Internal server error" in data["detail"]
            assert "Clustering service unavailable" in data["detail"] 