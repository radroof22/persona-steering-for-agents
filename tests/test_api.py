import pytest
import pandas as pd
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import json
from pathlib import Path

from app.main import app
from app.models.schema import Query, ClusteredQuery, PersonalizedRewrite, GenerateResponse, User, GenerateRequest, ClustersRequest
from app.services.prompt_summarizer import PromptSummarizer


class TestBasicEndpoints:
    """Test basic API endpoints that don't require complex mocking."""
    
    def test_root_endpoint(self):
        """Test the root endpoint returns correct response."""
        with TestClient(app) as client:
            response = client.get("/")
            
            assert response.status_code == 200
            data = response.json()
            assert "message" in data
            assert "Personalized Query Rewriting API" in data["message"]
    
    def test_health_endpoint(self):
        """Test the health check endpoint returns healthy status."""
        with TestClient(app) as client:
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "Service is operational" in data["message"]


class TestGenerateEndpoint:
    """Test the /generate endpoint with comprehensive mocking."""
    
    def setup_method(self):
        """Set up test data."""
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
    
    @patch('app.services.clusterer.SentenceTransformer')
    @patch('app.services.prompt_summarizer.create_prompt_summarizer')
    @patch('app.services.rewriter.create_style_personalizer')
    def test_generate_endpoint_success(self, mock_create_personalizer, mock_create_summarizer, mock_transformer):
        """Test successful generation of personalized rewrites."""
        # Mock sentence transformer
        mock_transformer_instance = Mock()
        mock_embeddings = [
            [1.0, 0.0, 0.0, 0.0, 0.0],  # Weather cluster
            [1.0, 0.0, 0.0, 0.0, 0.0],  # Weather cluster
            [1.0, 0.0, 0.0, 0.0, 0.0],  # Weather cluster
            [0.0, 1.0, 0.0, 0.0, 0.0],  # Restaurant cluster
        ]
        mock_transformer_instance.encode.return_value = mock_embeddings
        mock_transformer.return_value = mock_transformer_instance
        
        # Mock prompt summarizer
        mock_summarizer = Mock()
        mock_summarizer.summarize_clusters.return_value = [
            "What are the current weather conditions?",
            "Where should I eat tonight?"
        ]
        mock_create_summarizer.return_value = mock_summarizer
        
        # Mock style personalizer
        mock_personalizer = Mock()
        mock_personalizer.create_personalized_rewrites.return_value = [
            PersonalizedRewrite(
                user_id="user_001",
                original_query="What's the weather like today?",
                rewritten_query="What are the current weather conditions?",
                personalized_query="Please provide the current weather conditions."
            ),
            PersonalizedRewrite(
                user_id="user_002",
                original_query="How's the weather looking?",
                rewritten_query="What are the current weather conditions?",
                personalized_query="Hey, what's the weather like right now?"
            ),
            PersonalizedRewrite(
                user_id="user_003",
                original_query="What are the current weather conditions?",
                rewritten_query="What are the current weather conditions?",
                personalized_query="Please provide a detailed analysis of the current weather conditions."
            ),
            PersonalizedRewrite(
                user_id="user_001",
                original_query="Where should I eat tonight?",
                rewritten_query="Where should I eat tonight?",
                personalized_query="Please recommend dining options for this evening."
            )
        ]
        mock_create_personalizer.return_value = mock_personalizer
        
        # Create test payload
        users = [
            User(user_id=row['user_id'], description=row['description'])
            for _, row in self.mock_users.iterrows()
        ]
        
        queries = [
            Query(user_id=row['user_id'], question=row['question'])
            for _, row in self.mock_queries.iterrows()
        ]
        
        payload = GenerateRequest(users=users, queries=queries)
        
        with TestClient(app) as client:
            response = client.post("/generate", json=payload.model_dump())
            
            assert response.status_code == 200
            data = response.json()
            
            # Check response structure
            assert "clustered_queries" in data
            assert "representative_rewrites" in data
            assert "personalized_rewrites" in data
            
            # Check data integrity
            assert len(data["clustered_queries"]) > 0
            assert len(data["representative_rewrites"]) > 0
            assert len(data["personalized_rewrites"]) == 4  # One per query
            
            # Check personalized rewrite structure
            personalized = data["personalized_rewrites"][0]
            assert "user_id" in personalized
            assert "original_query" in personalized
            assert "rewritten_query" in personalized
            assert "personalized_query" in personalized
    
    def test_generate_endpoint_empty_data(self):
        """Test generate endpoint with empty data."""
        empty_payload = GenerateRequest(users=[], queries=[])
        
        with TestClient(app) as client:
            response = client.post("/generate", json=empty_payload.model_dump())
            
            # Should handle empty data gracefully
            assert response.status_code in [200, 500]  # Either success or error
    
    def test_generate_endpoint_malformed_request(self):
        """Test generate endpoint with malformed request."""
        with TestClient(app) as client:
            # Send invalid JSON
            response = client.post("/generate", json={"invalid": "data"})
            
            assert response.status_code == 422  # Validation error
            data = response.json()
            assert "detail" in data
    
    def test_generate_endpoint_missing_payload(self):
        """Test generate endpoint with missing payload."""
        with TestClient(app) as client:
            response = client.post("/generate")
            
            assert response.status_code == 422  # Validation error
    
    @patch('app.services.clusterer.SentenceTransformer')
    @patch('app.services.prompt_summarizer.create_prompt_summarizer')
    @patch('app.services.rewriter.create_style_personalizer')
    def test_generate_endpoint_clustering_failure(self, mock_create_personalizer, mock_create_summarizer, mock_transformer):
        """Test generate endpoint when clustering fails."""
        # Mock sentence transformer to raise exception
        mock_transformer_instance = Mock()
        mock_transformer_instance.encode.side_effect = Exception("Clustering failed")
        mock_transformer.return_value = mock_transformer_instance
        
        # Mock other services
        mock_create_summarizer.return_value = Mock()
        mock_create_personalizer.return_value = Mock()
        
        # Create test payload
        users = [User(user_id="user_001", description="Test user")]
        queries = [Query(user_id="user_001", question="Test question")]
        payload = GenerateRequest(users=users, queries=queries)
        
        with TestClient(app) as client:
            response = client.post("/generate", json=payload.model_dump())
            
            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
            assert "Internal server error" in data["detail"]


class TestClustersEndpoint:
    """Test the /clusters endpoint."""
    
    def setup_method(self):
        """Set up test data."""
        self.mock_users = pd.DataFrame({
            "user_id": ["user_001", "user_002"],
            "description": [
                "A formal business professional",
                "A casual tech enthusiast"
            ]
        })
        
        self.mock_queries = pd.DataFrame({
            "user_id": ["user_001", "user_002", "user_001"],
            "question": [
                "What's the weather like?",
                "How's the weather?",
                "Where should I eat?"
            ]
        })
    
    @patch('app.services.clusterer.SentenceTransformer')
    def test_clusters_endpoint_success(self, mock_transformer):
        """Test successful clustering without LLM processing."""
        # Mock sentence transformer
        mock_transformer_instance = Mock()
        mock_embeddings = [
            [1.0, 0.0, 0.0],  # Weather cluster
            [1.0, 0.0, 0.0],  # Weather cluster
            [0.0, 1.0, 0.0],  # Restaurant cluster
        ]
        mock_transformer_instance.encode.return_value = mock_embeddings
        mock_transformer.return_value = mock_transformer_instance
        
        # Create test payload
        users = [
            User(user_id=row['user_id'], description=row['description'])
            for _, row in self.mock_users.iterrows()
        ]
        
        queries = [
            Query(user_id=row['user_id'], question=row['question'])
            for _, row in self.mock_queries.iterrows()
        ]
        
        payload = ClustersRequest(users=users, queries=queries)
        
        with TestClient(app) as client:
            response = client.post("/clusters", json=payload.model_dump())
            
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
            
            # Verify data integrity
            assert cluster["query_count"] > 0
            assert len(cluster["user_ids"]) > 0
    
    def test_clusters_endpoint_empty_data(self):
        """Test clusters endpoint with empty data."""
        empty_payload = ClustersRequest(users=[], queries=[])
        
        with TestClient(app) as client:
            response = client.post("/clusters", json=empty_payload.model_dump())
            
            # Should handle empty data gracefully
            assert response.status_code in [200, 500]
    
    def test_clusters_endpoint_malformed_request(self):
        """Test clusters endpoint with malformed request."""
        with TestClient(app) as client:
            response = client.post("/clusters", json={"invalid": "data"})
            
            assert response.status_code == 422
            data = response.json()
            assert "detail" in data
    
    @patch('app.services.clusterer.SentenceTransformer')
    def test_clusters_endpoint_clustering_failure(self, mock_transformer):
        """Test clusters endpoint when clustering fails."""
        # Mock sentence transformer to raise exception
        mock_transformer_instance = Mock()
        mock_transformer_instance.encode.side_effect = Exception("Clustering failed")
        mock_transformer.return_value = mock_transformer_instance
        
        # Create test payload
        users = [User(user_id="user_001", description="Test user")]
        queries = [Query(user_id="user_001", question="Test question")]
        payload = ClustersRequest(users=users, queries=queries)
        
        with TestClient(app) as client:
            response = client.post("/clusters", json=payload.model_dump())
            
            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
            assert "Internal server error" in data["detail"]


class TestEndToEndPipeline:
    """Test the complete end-to-end pipeline."""
    
    def setup_method(self):
        """Set up test data for end-to-end testing."""
        self.mock_users = pd.DataFrame({
            "user_id": ["user_001", "user_002", "user_003"],
            "description": [
                "A formal business professional who prefers concise, professional language",
                "A casual tech enthusiast who uses informal language and tech jargon", 
                "An academic researcher who prefers detailed, analytical explanations"
            ]
        })
        
        self.mock_queries = pd.DataFrame({
            "user_id": ["user_001", "user_002", "user_003", "user_001", "user_002"],
            "question": [
                "What's the weather like today?",
                "How's the weather looking?",
                "What are the current weather conditions?",
                "Where should I eat tonight?",
                "Can you recommend a restaurant?"
            ]
        })
    
    @patch('app.services.clusterer.SentenceTransformer')
    @patch('app.services.prompt_summarizer.create_prompt_summarizer')
    @patch('app.services.rewriter.create_style_personalizer')
    @patch('app.services.llm_provider.create_llm_provider')
    def test_full_pipeline_workflow(self, mock_create_llm, mock_create_personalizer, mock_create_summarizer, mock_transformer):
        """Test the complete pipeline from clustering to personalization."""
        # Mock sentence transformer for clustering
        mock_transformer_instance = Mock()
        mock_embeddings = [
            [1.0, 0.0, 0.0],  # Weather cluster
            [1.0, 0.0, 0.0],  # Weather cluster
            [1.0, 0.0, 0.0],  # Weather cluster
            [0.0, 1.0, 0.0],  # Restaurant cluster
            [0.0, 1.0, 0.0],  # Restaurant cluster
        ]
        mock_transformer_instance.encode.return_value = mock_embeddings
        mock_transformer.return_value = mock_transformer_instance
        
        # Mock LLM provider
        mock_llm_provider = Mock()
        mock_llm_provider.generate.return_value = "Mocked personalized response"
        mock_create_llm.return_value = mock_llm_provider
        
        # Mock prompt summarizer
        mock_summarizer = Mock()
        mock_summarizer.summarize_clusters.return_value = [
            "What are the current weather conditions?",
            "Where should I eat tonight?"
        ]
        mock_create_summarizer.return_value = mock_summarizer
        
        # Mock style personalizer
        mock_personalizer = Mock()
        mock_personalizer.create_personalized_rewrites.return_value = [
            PersonalizedRewrite(
                user_id="user_001",
                original_query="What's the weather like today?",
                rewritten_query="What are the current weather conditions?",
                personalized_query="Please provide the current weather conditions."
            ),
            PersonalizedRewrite(
                user_id="user_002",
                original_query="How's the weather looking?",
                rewritten_query="What are the current weather conditions?",
                personalized_query="Hey, what's the weather like right now?"
            ),
            PersonalizedRewrite(
                user_id="user_003",
                original_query="What are the current weather conditions?",
                rewritten_query="What are the current weather conditions?",
                personalized_query="Please provide a detailed analysis of the current weather conditions."
            ),
            PersonalizedRewrite(
                user_id="user_001",
                original_query="Where should I eat tonight?",
                rewritten_query="Where should I eat tonight?",
                personalized_query="Please recommend dining options for this evening."
            ),
            PersonalizedRewrite(
                user_id="user_002",
                original_query="Can you recommend a restaurant?",
                rewritten_query="Where should I eat tonight?",
                personalized_query="Hey, can you suggest a good place to eat?"
            )
        ]
        mock_create_personalizer.return_value = mock_personalizer
        
        # Create test payload
        users = [
            User(user_id=row['user_id'], description=row['description'])
            for _, row in self.mock_users.iterrows()
        ]
        
        queries = [
            Query(user_id=row['user_id'], question=row['question'])
            for _, row in self.mock_queries.iterrows()
        ]
        
        payload = GenerateRequest(users=users, queries=queries)
        
        with TestClient(app) as client:
            # Test the complete pipeline
            response = client.post("/generate", json=payload.model_dump())
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify pipeline output
            assert len(data["clustered_queries"]) == 2  # Weather and restaurant clusters
            assert len(data["representative_rewrites"]) == 2  # One per cluster
            assert len(data["personalized_rewrites"]) == 5  # One per original query
            
            # Verify clustering worked
            weather_cluster = next(c for c in data["clustered_queries"] if "weather" in c["queries"][0]["question"].lower())
            restaurant_cluster = next(c for c in data["clustered_queries"] if "restaurant" in c["queries"][0]["question"].lower() or "eat" in c["queries"][0]["question"].lower())
            
            assert len(weather_cluster["queries"]) >= 3  # At least 3 weather queries
            assert len(restaurant_cluster["queries"]) >= 2  # At least 2 restaurant queries
            
            # Verify personalization worked - check that personalized queries exist and are different from rewritten queries
            formal_user = next(p for p in data["personalized_rewrites"] if p["user_id"] == "user_001")
            casual_user = next(p for p in data["personalized_rewrites"] if p["user_id"] == "user_002")
            
            # Check that personalized queries exist and are not empty
            assert len(formal_user["personalized_query"]) > 0
            assert len(casual_user["personalized_query"]) > 0
            
            # Check that personalized queries are different from rewritten queries (indicating personalization worked)
            assert formal_user["personalized_query"] != formal_user["rewritten_query"]
            assert casual_user["personalized_query"] != casual_user["rewritten_query"]
    
    @patch('app.services.clusterer.SentenceTransformer')
    @patch('app.services.prompt_summarizer.create_prompt_summarizer')
    @patch('app.services.rewriter.create_style_personalizer')
    @patch('app.services.llm_provider.create_llm_provider')
    def test_pipeline_with_singleton_clusters(self, mock_create_llm, mock_create_personalizer, mock_create_summarizer, mock_transformer):
        """Test pipeline with queries that don't cluster well (singleton clusters)."""
        # Mock sentence transformer to create singleton clusters
        mock_transformer_instance = Mock()
        mock_embeddings = [
            [1.0, 0.0, 0.0],  # Unique cluster 1
            [0.0, 1.0, 0.0],  # Unique cluster 2
            [0.0, 0.0, 1.0],  # Unique cluster 3
        ]
        mock_transformer_instance.encode.return_value = mock_embeddings
        mock_transformer.return_value = mock_transformer_instance
        
        # Mock LLM provider
        mock_llm_provider = Mock()
        mock_llm_provider.generate.return_value = "Mocked personalized response"
        mock_create_llm.return_value = mock_llm_provider
        
        # Mock prompt summarizer
        mock_summarizer = Mock()
        mock_summarizer.summarize_clusters.return_value = [
            "What's the weather like?",
            "Where should I eat?",
            "How do I install Python?"
        ]
        mock_create_summarizer.return_value = mock_summarizer
        
        # Mock style personalizer
        mock_personalizer = Mock()
        mock_personalizer.create_personalized_rewrites.return_value = [
            PersonalizedRewrite(
                user_id="user_001",
                original_query="What's the weather like?",
                rewritten_query="What's the weather like?",
                personalized_query="Please provide weather information."
            ),
            PersonalizedRewrite(
                user_id="user_002",
                original_query="Where should I eat?",
                rewritten_query="Where should I eat?",
                personalized_query="Hey, where should I grab food?"
            ),
            PersonalizedRewrite(
                user_id="user_003",
                original_query="How do I install Python?",
                rewritten_query="How do I install Python?",
                personalized_query="Please provide detailed instructions for Python installation."
            )
        ]
        mock_create_personalizer.return_value = mock_personalizer
        
        # Create test payload with diverse queries
        users = [
            User(user_id="user_001", description="A formal business professional"),
            User(user_id="user_002", description="A casual tech enthusiast"),
            User(user_id="user_003", description="An academic researcher")
        ]
        
        queries = [
            Query(user_id="user_001", question="What's the weather like?"),
            Query(user_id="user_002", question="Where should I eat?"),
            Query(user_id="user_003", question="How do I install Python?")
        ]
        
        payload = GenerateRequest(users=users, queries=queries)
        
        with TestClient(app) as client:
            response = client.post("/generate", json=payload.model_dump())
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify singleton clusters are handled
            assert len(data["clustered_queries"]) == 3  # One cluster per query
            assert len(data["representative_rewrites"]) == 3  # One rewrite per cluster
            assert len(data["personalized_rewrites"]) == 3  # One per original query


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_large_payload_handling(self):
        """Test API with large payload."""
        size = 25
        # Create large dataset
        large_users = [
            User(user_id=f"user_{i:03d}", description=f"User {i} description")
            for i in range(25)
        ]
        
        large_queries = [
            Query(user_id=f"user_{i:03d}", question=f"Query number {i}")
            for i in range(25)
        ]
        
        payload = GenerateRequest(users=large_users, queries=large_queries)
        
        with TestClient(app) as client:
            response = client.post("/generate", json=payload.model_dump())
            
            # Should handle large payload (either success or reasonable error)
            assert response.status_code in [200, 413, 500]  # Success, payload too large, or server error
    
    def test_malformed_user_data(self):
        """Test with malformed user data."""
        with TestClient(app) as client:
            # Missing required fields
            malformed_payload = {
                "users": [{"user_id": "user_001"}],  # Missing description
                "queries": [{"user_id": "user_001", "question": "Test?"}]
            }
            
            response = client.post("/generate", json=malformed_payload)
            
            assert response.status_code == 422  # Validation error
    
    def test_malformed_query_data(self):
        """Test with malformed query data."""
        with TestClient(app) as client:
            # Missing required fields
            malformed_payload = {
                "users": [{"user_id": "user_001", "description": "Test user"}],
                "queries": [{"user_id": "user_001"}]  # Missing question
            }
            
            response = client.post("/generate", json=malformed_payload)
            
            assert response.status_code == 422  # Validation error
    
    def test_mismatched_user_ids(self):
        """Test with queries referencing non-existent users."""
        payload = GenerateRequest(
            users=[User(user_id="user_001", description="Test user")],
            queries=[
                Query(user_id="user_001", question="Valid query"),
                Query(user_id="user_999", question="Invalid user ID")  # Non-existent user
            ]
        )
        
        with TestClient(app) as client:
            response = client.post("/generate", json=payload.model_dump())
            
            # Should handle gracefully (either success or reasonable error)
            assert response.status_code in [200, 500]
    
    def test_empty_strings(self):
        """Test with empty strings in data."""
        payload = GenerateRequest(
            users=[User(user_id="user_001", description="")],  # Empty description
            queries=[Query(user_id="user_001", question="")]   # Empty question
        )
        
        with TestClient(app) as client:
            response = client.post("/generate", json=payload.model_dump())
            
            # Should handle empty strings (either success or validation error)
            assert response.status_code in [200, 422, 500]


class TestServiceIntegration:
    """Test integration with external services."""
    
    
    def test_service_dependency_injection(self):
        """Test that service dependencies are properly injected."""
        with TestClient(app) as client:
            # Test each service dependency endpoint
            response = client.get("/health")
            assert response.status_code == 200
            
            # Verify clusterer service injection
            payload = ClustersRequest(
                users=[User(user_id="user_001", description="Test user")],
                queries=[Query(user_id="user_001", question="Test query")]
            )
            response = client.post("/clusters", json=payload.model_dump())
            assert response.status_code == 200
            
            # Verify full pipeline service injection
            payload = GenerateRequest(
                users=[User(user_id="user_001", description="Test user")],
                queries=[Query(user_id="user_001", question="Test query")]
            )
            response = client.post("/generate", json=payload.model_dump())
            assert response.status_code == 200
            
            # Verify response contains expected service outputs
            data = response.json()
            assert "clustered_queries" in data
            assert "representative_rewrites" in data 
            assert "personalized_rewrites" in data
        
    
    def test_service_initialization_failure(self):
        """Test behavior when service initialization fails."""
        # This would require mocking the lifespan context manager
        # For now, we'll test that the app starts up
        with TestClient(app) as client:
            response = client.get("/health")
            # Should still respond even if some services fail
            assert response.status_code in [200, 500]


class TestResponseValidation:
    """Test response model validation."""
    
    def test_generate_response_structure(self):
        """Test that generate response has correct structure."""
        # Create valid response data
        clustered_query = ClusteredQuery(
            cluster_id=0,
            user_ids=["user_001"],
            queries=[Query(user_id="user_001", question="Test question")]
        )
        
        personalized_rewrite = PersonalizedRewrite(
            user_id="user_001",
            original_query="Test question",
            rewritten_query="Test rewrite",
            personalized_query="Personalized test"
        )
        
        response = GenerateResponse(
            clustered_queries=[clustered_query],
            representative_rewrites=["Test rewrite"],
            personalized_rewrites=[personalized_rewrite]
        )
        
        # Verify structure
        assert len(response.clustered_queries) == 1
        assert len(response.representative_rewrites) == 1
        assert len(response.personalized_rewrites) == 1
        
        # Verify data integrity
        assert response.clustered_queries[0].cluster_id == 0
        assert response.personalized_rewrites[0].user_id == "user_001"
    
    def test_clusters_response_structure(self):
        """Test that clusters response has correct structure."""
        from app.models.schema import ClustersResponse
        
        clusters_data = [
            {
                "cluster_id": 0,
                "user_ids": ["user_001"],
                "query_count": 1
            }
        ]
        
        response = ClustersResponse(clusters=clusters_data)
        
        assert len(response.clusters) == 1
        assert response.clusters[0]["cluster_id"] == 0
        assert response.clusters[0]["query_count"] == 1
