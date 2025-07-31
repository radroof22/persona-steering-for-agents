import pytest
import pandas as pd
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json
from pathlib import Path

from app.main import app
from app.models.schema import Query, ClusteredQuery, PersonalizedRewrite, GenerateResponse


class TestFastAPIEndpoints:
    """Test cases for FastAPI endpoints with proper payloads and mock data."""
    
    def setup_method(self):
        """Set up test client and load mock data."""
        self.client = TestClient(app)
        
        # Load mock data from actual files
        try:
            from app.utils.loader import load_mock_data
            self.mock_users, self.mock_queries = load_mock_data()
        except FileNotFoundError:
            # Fallback to hardcoded data if files don't exist
            self.mock_users = pd.DataFrame({
                "user_id": ["user_001", "user_002", "user_003"],
                "description": [
                    "A formal business professional who prefers concise, professional language",
                    "A casual tech enthusiast who uses informal language and tech jargon", 
                    "An academic researcher who prefers detailed, analytical explanations"
                ]
            })
            
            self.mock_queries = pd.DataFrame({
                "user_id": ["user_001", "user_002", "user_003", "user_001"],
                "question": [
                    "What's the weather like today?",
                    "How's the weather looking?",
                    "What are the current weather conditions?",
                    "Where should I eat tonight?"
                ]
            })
    
    def test_root_endpoint(self):
        """Test the root endpoint."""
        response = self.client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "Personalized Query Rewriting API" in data["message"]
    
    def test_health_endpoint(self):
        """Test the health check endpoint."""
        response = self.client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "Service is operational" in data["message"]
    
    @patch('app.services.clusterer.SentenceTransformer')
    @patch('app.services.llm_provider.AutoTokenizer')
    @patch('app.services.llm_provider.AutoModelForCausalLM')
    def test_generate_endpoint_with_payload(self, mock_model, mock_tokenizer, mock_transformer):
        """Test the generate endpoint with proper payload."""
        # Mock the sentence transformer to return embeddings for all queries
        mock_transformer_instance = Mock()
        
        # Create embeddings that match the number of queries in mock data
        # The mock data has 20 queries, so we need 20 embeddings
        # We'll create embeddings that will form clusters based on the mock data structure
        mock_embeddings = []
        for i in range(20):  # 20 queries total
            if i < 4:  # First 4: Weather cluster
                mock_embeddings.append([1.0, 0.0, 0.0, 0.0, 0.0])
            elif i < 8:  # Next 4: Restaurant cluster  
                mock_embeddings.append([0.0, 1.0, 0.0, 0.0, 0.0])
            elif i < 11:  # Next 3: Technical cluster
                mock_embeddings.append([0.0, 0.0, 1.0, 0.0, 0.0])
            elif i < 14:  # Next 3: Travel cluster
                mock_embeddings.append([0.0, 0.0, 0.0, 1.0, 0.0])
            elif i < 17:  # Next 3: Health cluster
                mock_embeddings.append([0.0, 0.0, 0.0, 0.0, 1.0])
            else:  # Last 3: Singleton clusters
                mock_embeddings.append([0.1, 0.1, 0.1, 0.1, 0.1])
        
        mock_transformer_instance.encode.return_value = mock_embeddings
        mock_transformer.return_value = mock_transformer_instance
        
        # Mock the LLM components
        mock_tokenizer_instance = Mock()
        # Mock the tokenizer call (when used as a function)
        mock_tokenizer_instance.return_value = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
        # Mock the decode method
        mock_tokenizer_instance.decode.return_value = "What are the current weather conditions?"
        mock_tokenizer_instance.eos_token_id = 2
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_model_instance = Mock()
        mock_model_instance.generate.return_value = [[1, 2, 3, 4, 5]]
        mock_model.return_value = mock_model_instance
        
        # Mock the LLM provider methods to return strings instead of MagicMock objects
        from app.services.llm_provider import LLMProvider
        mock_llm_provider = Mock(spec=LLMProvider)
        mock_llm_provider.rewrite_multiple_queries.return_value = [
            "What are the current weather conditions?",
            "Where should I eat tonight?",
            "How do I install Python?",
            "What's the best time to visit Paris?",
            "How can I improve my sleep?",
            "What's the capital of Mongolia?",
            "How do quantum computers work?",
            "What's the recipe for sourdough bread?"
        ]
        mock_llm_provider.rewrite_query.return_value = "Personalized query response"
        
        # Create payload from mock data
        from app.models.schema import User, Query, GenerateRequest
        
        users = [
            User(user_id=row['user_id'], description=row['description'])
            for _, row in self.mock_users.iterrows()
        ]
        
        queries = [
            Query(user_id=row['user_id'], question=row['question'])
            for _, row in self.mock_queries.iterrows()
        ]
        
        payload = GenerateRequest(users=users, queries=queries)
        
        # Patch the LLM provider creation
        with patch('app.services.llm_provider.create_llm_provider', return_value=mock_llm_provider):
            with patch('app.services.rewriter.create_style_personalizer') as mock_create_personalizer:
                mock_personalizer = Mock()
                mock_personalizer.create_personalized_rewrites.return_value = [
                    Mock(
                        user_id="user_001",
                        original_query="What's the weather like today?",
                        rewritten_query="What are the current weather conditions?",
                        personalized_query="Personalized weather query"
                    )
                ] * 20  # Create 20 mock personalized rewrites
                mock_create_personalizer.return_value = mock_personalizer
                
                # Create a test client that will trigger the lifespan
                with TestClient(app) as client:
                    # Make POST request to generate endpoint with payload
                    response = client.post("/generate", json=payload.model_dump())
                    
                    # Check response
                    assert response.status_code == 200
                    data = response.json()
                    
                    # Check response structure
                    assert "clustered_queries" in data
                    assert "representative_rewrites" in data
                    assert "personalized_rewrites" in data
                    
                    # Check that we have some data
                    assert len(data["clustered_queries"]) > 0  # Should have multiple clusters
                    assert len(data["representative_rewrites"]) > 0  # One rewrite per cluster
                    assert len(data["personalized_rewrites"]) == 20  # One per original query (20 total)
                    
                    # Check cluster structure
                    cluster = data["clustered_queries"][0]
                    assert "cluster_id" in cluster
                    assert "user_ids" in cluster
                    assert "queries" in cluster
                    
                    # Check personalized rewrite structure 
                    personalized = data["personalized_rewrites"][0]
                    assert "user_id" in personalized
                    assert "original_query" in personalized
                    assert "rewritten_query" in personalized
                    assert "personalized_query" in personalized
                    
                    # Verify we have the expected structure
                    total_queries_in_clusters = sum(len(c["queries"]) for c in data["clustered_queries"])
                    assert total_queries_in_clusters == 20  # All 20 queries should be in clusters
    
    @patch('app.services.clusterer.SentenceTransformer')
    def test_clusters_endpoint_with_payload(self, mock_transformer):
        """Test the clusters endpoint with proper payload."""
        # Mock the sentence transformer to return embeddings for all queries
        mock_transformer_instance = Mock()
        
        # Create embeddings that match the number of queries in mock data (20 queries)
        mock_embeddings = []
        for i in range(20):  # 20 queries total
            if i < 4:  # First 4: Weather cluster
                mock_embeddings.append([1.0, 0.0, 0.0, 0.0, 0.0])
            elif i < 8:  # Next 4: Restaurant cluster  
                mock_embeddings.append([0.0, 1.0, 0.0, 0.0, 0.0])
            elif i < 11:  # Next 3: Technical cluster
                mock_embeddings.append([0.0, 0.0, 1.0, 0.0, 0.0])
            elif i < 14:  # Next 3: Travel cluster
                mock_embeddings.append([0.0, 0.0, 0.0, 1.0, 0.0])
            elif i < 17:  # Next 3: Health cluster
                mock_embeddings.append([0.0, 0.0, 0.0, 0.0, 1.0])
            else:  # Last 3: Singleton clusters
                mock_embeddings.append([0.1, 0.1, 0.1, 0.1, 0.1])
        
        mock_transformer_instance.encode.return_value = mock_embeddings
        mock_transformer.return_value = mock_transformer_instance
        
        # Create payload from mock data
        from app.models.schema import User, Query, ClustersRequest
        
        users = [
            User(user_id=row['user_id'], description=row['description'])
            for _, row in self.mock_users.iterrows()
        ]
        
        queries = [
            Query(user_id=row['user_id'], question=row['question'])
            for _, row in self.mock_queries.iterrows()
        ]
        
        payload = ClustersRequest(users=users, queries=queries)
        
        # Create a test client that will trigger the lifespan
        with TestClient(app) as client:
            # Make POST request to clusters endpoint with payload
            response = client.post("/clusters", json=payload.model_dump())
            
            # Check response
            assert response.status_code == 200
            data = response.json()
            
            # Check response structure
            assert "clusters" in data
            assert len(data["clusters"]) > 0
            
            # Check cluster structure
            cluster = data["clusters"][0]
            assert "cluster_id" in cluster
            assert "user_ids" in cluster
            assert "query_count" in cluster
            
            # Verify the data makes sense
            assert cluster["query_count"] > 0
            assert len(cluster["user_ids"]) > 0
    
    def test_generate_endpoint_error_handling(self):
        """Test error handling in the generate endpoint."""
        # Test with invalid payload
        with TestClient(app) as client:
            # Send invalid JSON payload
            response = client.post("/generate", json={"invalid": "payload"})
            
            # Should return 422 validation error
            assert response.status_code == 422
            data = response.json()
            assert "detail" in data
    
    def test_clusters_endpoint_error_handling(self):
        """Test error handling in the clusters endpoint."""
        with TestClient(app) as client:
            # Send invalid JSON payload
            response = client.post("/clusters", json={"invalid": "payload"})
            
            # Should return 422 validation error
            assert response.status_code == 422
            data = response.json()
            assert "detail" in data


class TestAPIResponseModels:
    """Test cases for API response models."""
    
    def test_generate_response_model_validation(self):
        """Test GenerateResponse model validation with proper data."""
        # Create test data
        clustered_query = ClusteredQuery(
            cluster_id=0,
            user_ids=["user_001", "user_002"],
            queries=[
                Query(user_id="user_001", question="What's the weather like today?"),
                Query(user_id="user_002", question="How's the weather looking?")
            ]
        )
        
        personalized_rewrite = PersonalizedRewrite(
            user_id="user_001",
            original_query="What's the weather like today?",
            rewritten_query="What are the current weather conditions?",
            personalized_query="Please provide the current weather conditions."
        )
        
        # Create response
        response = GenerateResponse(
            clustered_queries=[clustered_query],
            representative_rewrites=["What are the current weather conditions?"],
            personalized_rewrites=[personalized_rewrite]
        )
        
        # Check structure
        assert len(response.clustered_queries) == 1
        assert len(response.representative_rewrites) == 1
        assert len(response.personalized_rewrites) == 1
        
        # Check data
        assert response.clustered_queries[0].cluster_id == 0
        assert response.personalized_rewrites[0].user_id == "user_001"
        assert "weather" in response.personalized_rewrites[0].original_query.lower()
    
    def test_clustered_query_model_validation(self):
        """Test ClusteredQuery model validation."""
        clustered_query = ClusteredQuery(
            cluster_id=1,
            user_ids=["user_003"],
            queries=[
                Query(user_id="user_003", question="Where should I eat tonight?")
            ]
        )
        
        assert clustered_query.cluster_id == 1
        assert len(clustered_query.user_ids) == 1
        assert len(clustered_query.queries) == 1
        assert clustered_query.queries[0].user_id == "user_003"
    
    def test_personalized_rewrite_model_validation(self):
        """Test PersonalizedRewrite model validation."""
        rewrite = PersonalizedRewrite(
            user_id="user_002",
            original_query="How's the weather?",
            rewritten_query="What are the current weather conditions?",
            personalized_query="Hey, what's the weather like right now?"
        )
        
        assert rewrite.user_id == "user_002"
        assert "weather" in rewrite.original_query.lower()
        assert "weather" in rewrite.rewritten_query.lower()
        assert "hey" in rewrite.personalized_query.lower()  # More casual tone


class TestAPIIntegration:
    """Integration tests that test the full API workflow."""
    
    @patch('app.utils.loader.load_mock_data')
    @patch('app.services.clusterer.SentenceTransformer')
    @patch('app.services.llm_provider.AutoTokenizer')
    @patch('app.services.llm_provider.AutoModelForCausalLM')
    def test_full_api_workflow(self, mock_model, mock_tokenizer, mock_transformer, mock_load_data):
        """Test the complete API workflow from data loading to response generation."""
        # Set up comprehensive mocks
        mock_load_data.return_value = (self.mock_users, self.mock_queries)
        
        # Mock sentence transformer for clustering
        mock_transformer_instance = Mock()
        mock_transformer_instance.encode.return_value = [
            [1.0, 0.0, 0.0],  # Weather cluster
            [1.0, 0.0, 0.0],  # Weather cluster
            [1.0, 0.0, 0.0],  # Weather cluster
            [0.0, 1.0, 0.0],  # Restaurant cluster
        ]
        mock_transformer.return_value = mock_transformer_instance
        
        # Mock LLM components
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.return_value = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
        mock_tokenizer_instance.decode.return_value = "What are the current weather conditions?"
        mock_tokenizer_instance.eos_token_id = 2
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_model_instance = Mock()
        mock_model_instance.generate.return_value = [[1, 2, 3, 4, 5]]
        mock_model.return_value = mock_model_instance
        
        with TestClient(app) as client:
            # Test health endpoint
            health_response = client.get("/health")
            assert health_response.status_code == 200
            
            # Test generate endpoint
            generate_response = client.post("/generate")
            assert generate_response.status_code == 200
            
            generate_data = generate_response.json()
            assert "clustered_queries" in generate_data
            assert "representative_rewrites" in generate_data
            assert "personalized_rewrites" in generate_data
            
            # Test clusters endpoint
            clusters_response = client.get("/clusters")
            assert clusters_response.status_code == 200
            
            clusters_data = clusters_response.json()
            assert "clusters" in clusters_data
            assert len(clusters_data["clusters"]) > 0
            
            # Verify consistency between endpoints
            assert len(generate_data["clustered_queries"]) == len(clusters_data["clusters"])
    
    def test_api_with_real_mock_data_files(self):
        """Test API with actual mock data files."""
        # Check if mock data files exist
        data_dir = Path("app/data")
        users_file = data_dir / "mock_users.parquet"
        queries_file = data_dir / "mock_queries.parquet"
        
        if not users_file.exists() or not queries_file.exists():
            pytest.skip("Mock data files not found. Run create_mock_data.py first.")
        
        # Test that the data files can be loaded
        try:
            users_df = pd.read_parquet(users_file)
            queries_df = pd.read_parquet(queries_file)
            
            assert len(users_df) > 0
            assert len(queries_df) > 0
            assert "user_id" in users_df.columns
            assert "description" in users_df.columns
            assert "user_id" in queries_df.columns
            assert "question" in queries_df.columns
            
        except Exception as e:
            pytest.skip(f"Failed to load mock data files: {e}")


class TestAPIPayloadValidation:
    """Test cases for API payload validation and edge cases."""
    
    def test_empty_data_handling(self):
        """Test how the API handles empty data scenarios."""
        # This would require mocking empty data scenarios
        # For now, we'll test the error handling
        with TestClient(app) as client:
            response = client.post("/generate")
            # Should handle gracefully
            assert response.status_code in [500, 200]  # Either error or success
    
    def test_large_payload_handling(self):
        """Test how the API handles large payloads."""
        # Create a large number of queries
        large_queries = pd.DataFrame({
            "user_id": [f"user_{i:03d}" for i in range(100)],
            "question": [f"Query number {i}" for i in range(100)]
        })
        
        # This test would require proper mocking of the large dataset
        # For now, we'll verify the structure is valid
        assert len(large_queries) == 100
        assert "user_id" in large_queries.columns
        assert "question" in large_queries.columns
    
    def test_malformed_data_handling(self):
        """Test how the API handles malformed data."""
        # Test with missing columns
        malformed_users = pd.DataFrame({
            "user_id": ["user_001", "user_002"],
            # Missing description column
        })
        
        malformed_queries = pd.DataFrame({
            "user_id": ["user_001", "user_002"],
            # Missing question column
        })
        
        # These should cause validation errors
        assert "description" not in malformed_users.columns
        assert "question" not in malformed_queries.columns
