import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import List

from backend.services.clusterer import QueryClusterer, create_clusterer
from backend.models.schema import Query, ClusteredQuery


class TestQueryClusterer:
    """Test suite for QueryClusterer class."""
    
    @pytest.fixture
    def sample_queries(self) -> List[Query]:
        """Sample queries for testing."""
        return [
            Query(user_id="user1", question="What is the weather like today?"),
            Query(user_id="user2", question="How's the weather today?"),
            Query(user_id="user3", question="What's the temperature outside?"),
            Query(user_id="user4", question="How do I reset my password?"),
            Query(user_id="user5", question="I forgot my password, help me reset it"),
            Query(user_id="user6", question="What time does the store close?"),
        ]
    
    @pytest.fixture
    def clusterer(self) -> QueryClusterer:
        """Create a QueryClusterer instance for testing."""
        return QueryClusterer(eps=0.15, min_samples=2)
    
    def test_init(self):
        """Test QueryClusterer initialization."""
        clusterer = QueryClusterer(model_name="test-model", eps=0.2, min_samples=3)
        
        assert clusterer.model_name == "test-model"
        assert clusterer.eps == 0.2
        assert clusterer.min_samples == 3
        assert clusterer.model is None
        assert clusterer.embeddings is None
        assert clusterer.cluster_labels is None
    
    @patch('backend.services.clusterer.SentenceTransformer')
    def test_load_model(self, mock_sentence_transformer):
        """Test model loading."""
        clusterer = QueryClusterer()
        
        # Test initial state
        assert clusterer.model is None
        
        # Load model
        clusterer._load_model()
        
        # Verify model was created
        mock_sentence_transformer.assert_called_once_with("all-MiniLM-L6-v2")
        assert clusterer.model is not None
        
        # Test that model is not reloaded
        clusterer._load_model()
        mock_sentence_transformer.assert_called_once()  # Still only called once
    
    @patch('backend.services.clusterer.SentenceTransformer')
    def test_encode_queries(self, mock_sentence_transformer):
        """Test query encoding."""
        # Setup mock
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_sentence_transformer.return_value = mock_model
        
        clusterer = QueryClusterer()
        queries = ["query1", "query2"]
        
        # Encode queries
        embeddings = clusterer.encode_queries(queries)
        
        # Verify results
        assert clusterer.model is not None
        mock_model.encode.assert_called_once_with(queries, show_progress_bar=False)
        assert embeddings.shape == (2, 2)
        assert clusterer.embeddings is not None
        assert clusterer.distance_matrix is not None
    
    @patch('backend.services.clusterer.SentenceTransformer')
    def test_cluster_queries(self, mock_sentence_transformer):
        """Test query clustering."""
        # Setup mocks
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        mock_sentence_transformer.return_value = mock_model
        
        clusterer = QueryClusterer()
        queries = ["query1", "query2", "query3"]
        
        # Mock DBSCAN
        with patch('backend.services.clusterer.DBSCAN') as mock_dbscan:
            mock_dbscan_instance = Mock()
            mock_dbscan_instance.fit_predict.return_value = np.array([0, 0, 1])
            mock_dbscan.return_value = mock_dbscan_instance
            
            # Cluster queries
            cluster_labels = clusterer.cluster_queries(queries)
            
            # Verify results
            assert cluster_labels.shape == (3,)
            assert np.array_equal(cluster_labels, np.array([0, 0, 1]))
            assert clusterer.cluster_labels is not None
    
    @patch.object(QueryClusterer, 'cluster_queries')
    def test_process_queries_normal_clusters(self, mock_cluster_queries, sample_queries, clusterer):
        """Test process_queries with normal clusters (no noise points)."""
        # Mock cluster_queries to return normal cluster labels
        mock_cluster_queries.return_value = np.array([0, 0, 0, 1, 1, 2])
        
        # Process queries
        result = clusterer.process_queries(sample_queries)
        
        # Verify cluster_queries was called
        mock_cluster_queries.assert_called_once_with([q.question for q in sample_queries])
        
        # Verify results
        assert len(result) == 3
        
        # Check cluster 0 (weather queries)
        cluster_0 = result[0]
        assert cluster_0.cluster_id == 0
        assert cluster_0.user_ids == ["user1", "user2", "user3"]
        assert len(cluster_0.queries) == 3
        assert cluster_0.queries[0].question == "What is the weather like today?"
        assert cluster_0.queries[1].question == "How's the weather today?"
        assert cluster_0.queries[2].question == "What's the temperature outside?"
        
        # Check cluster 1 (password queries)
        cluster_1 = result[1]
        assert cluster_1.cluster_id == 1
        assert cluster_1.user_ids == ["user4", "user5"]
        assert len(cluster_1.queries) == 2
        assert cluster_1.queries[0].question == "How do I reset my password?"
        assert cluster_1.queries[1].question == "I forgot my password, help me reset it"
        
        # Check cluster 2 (store hours query)
        cluster_2 = result[2]
        assert cluster_2.cluster_id == 2
        assert cluster_2.user_ids == ["user6"]
        assert len(cluster_2.queries) == 1
        assert cluster_2.queries[0].question == "What time does the store close?"
    
    @patch.object(QueryClusterer, 'cluster_queries')
    def test_process_queries_with_noise_points(self, mock_cluster_queries, sample_queries, clusterer):
        """Test process_queries with noise points (cluster label -1)."""
        # Mock cluster_queries to return some noise points
        mock_cluster_queries.return_value = np.array([0, 0, -1, 1, 1, -1])
        
        # Process queries
        result = clusterer.process_queries(sample_queries)
        
        # Verify cluster_queries was called
        mock_cluster_queries.assert_called_once_with([q.question for q in sample_queries])
        
        # Verify results - should have 4 clusters (2 normal + 2 noise points as singletons)
        assert len(result) == 4
        
        # Check normal clusters first
        normal_clusters = [c for c in result if c.cluster_id in [0, 1]]
        assert len(normal_clusters) == 2
        
        # Check cluster 0
        cluster_0 = next(c for c in result if c.cluster_id == 0)
        assert cluster_0.user_ids == ["user1", "user2"]
        assert len(cluster_0.queries) == 2
        
        # Check cluster 1
        cluster_1 = next(c for c in result if c.cluster_id == 1)
        assert cluster_1.user_ids == ["user4", "user5"]
        assert len(cluster_1.queries) == 2
        
        # Check noise points as singleton clusters
        noise_clusters = [c for c in result if c.cluster_id not in [0, 1]]
        assert len(noise_clusters) == 2
        
        # Verify noise clusters have unique IDs and single queries
        noise_cluster_ids = [c.cluster_id for c in noise_clusters]
        assert len(set(noise_cluster_ids)) == 2  # Unique IDs
        assert all(len(c.queries) == 1 for c in noise_clusters)
        assert all(len(c.user_ids) == 1 for c in noise_clusters)
        
        # Verify the specific noise queries
        user3_cluster = next(c for c in noise_clusters if c.user_ids[0] == "user3")
        user6_cluster = next(c for c in noise_clusters if c.user_ids[0] == "user6")
        assert user3_cluster.queries[0].question == "What's the temperature outside?"
        assert user6_cluster.queries[0].question == "What time does the store close?"
    
    @patch.object(QueryClusterer, 'cluster_queries')
    def test_process_queries_all_noise(self, mock_cluster_queries, sample_queries, clusterer):
        """Test process_queries when all queries are noise points."""
        # Mock cluster_queries to return all noise points
        mock_cluster_queries.return_value = np.array([-1, -1, -1, -1, -1, -1])
        
        # Process queries
        result = clusterer.process_queries(sample_queries)
        
        # Verify results - each query should be its own cluster
        assert len(result) == 6
        
        # Verify each cluster is a singleton
        for i, cluster in enumerate(result):
            assert len(cluster.queries) == 1
            assert len(cluster.user_ids) == 1
            assert cluster.cluster_id >= 0  # Should have positive IDs
        
        # Verify cluster IDs are unique and sequential
        cluster_ids = [c.cluster_id for c in result]
        assert len(set(cluster_ids)) == 6
        assert min(cluster_ids) >= 0
    
    @patch.object(QueryClusterer, 'cluster_queries')
    def test_process_queries_empty_input(self, mock_cluster_queries, clusterer):
        """Test process_queries with empty input."""
        # Process queries
        result = clusterer.process_queries([])
        
        # Verify results
        assert result == []
        # cluster_queries should not be called for empty input
        mock_cluster_queries.assert_not_called()
    
    @patch.object(QueryClusterer, 'cluster_queries')
    def test_process_queries_single_query(self, mock_cluster_queries, clusterer):
        """Test process_queries with a single query."""
        single_query = [Query(user_id="user1", question="Single query")]
        
        # Mock cluster_queries to return single cluster
        mock_cluster_queries.return_value = np.array([0])
        
        # Process queries
        result = clusterer.process_queries(single_query)
        
        # Verify results
        assert len(result) == 1
        assert result[0].cluster_id == 0
        assert result[0].user_ids == ["user1"]
        assert len(result[0].queries) == 1
        assert result[0].queries[0].question == "Single query"
    
    @patch.object(QueryClusterer, 'cluster_queries')
    def test_process_queries_mixed_noise_and_clusters(self, mock_cluster_queries, sample_queries, clusterer):
        """Test process_queries with mixed normal clusters and noise points."""
        # Mock cluster_queries to return mixed results
        mock_cluster_queries.return_value = np.array([0, -1, 0, 1, -1, 1])
        
        # Process queries
        result = clusterer.process_queries(sample_queries)
        
        # Verify results
        assert len(result) == 4  # 2 normal clusters + 2 noise points
        
        # Check normal clusters
        cluster_0 = next(c for c in result if c.cluster_id == 0)
        cluster_1 = next(c for c in result if c.cluster_id == 1)
        
        assert cluster_0.user_ids == ["user1", "user3"]
        assert cluster_1.user_ids == ["user4", "user6"]
        
        # Check noise clusters
        noise_clusters = [c for c in result if c.cluster_id not in [0, 1]]
        assert len(noise_clusters) == 2
        
        # Verify noise clusters correspond to user2 and user5
        noise_user_ids = [c.user_ids[0] for c in noise_clusters]
        assert "user2" in noise_user_ids
        assert "user5" in noise_user_ids


class TestCreateClusterer:
    """Test suite for create_clusterer factory function."""
    
    def test_create_clusterer_default_params(self):
        """Test create_clusterer with default parameters."""
        clusterer = create_clusterer()
        
        assert isinstance(clusterer, QueryClusterer)
        assert clusterer.eps == 0.15
        assert clusterer.min_samples == 2
        assert clusterer.model_name == "all-MiniLM-L6-v2"
    
    def test_create_clusterer_custom_params(self):
        """Test create_clusterer with custom parameters."""
        clusterer = create_clusterer(eps=0.25, min_samples=5)
        
        assert isinstance(clusterer, QueryClusterer)
        assert clusterer.eps == 0.25
        assert clusterer.min_samples == 5
        assert clusterer.model_name == "all-MiniLM-L6-v2"


class TestClustererIntegration:
    """Integration tests for the clusterer service."""
    
    @patch('backend.services.clusterer.SentenceTransformer')
    @patch('backend.services.clusterer.DBSCAN')
    def test_full_clustering_pipeline(self, mock_dbscan, mock_sentence_transformer):
        """Test the full clustering pipeline without mocking cluster_queries."""
        # Setup mocks
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        mock_sentence_transformer.return_value = mock_model
        
        mock_dbscan_instance = Mock()
        mock_dbscan_instance.fit_predict.return_value = np.array([0, 0, 1])
        mock_dbscan.return_value = mock_dbscan_instance
        
        # Create clusterer and test data
        clusterer = QueryClusterer()
        queries = [
            Query(user_id="user1", question="What is the weather?"),
            Query(user_id="user2", question="How's the weather?"),
            Query(user_id="user3", question="Reset password"),
        ]
        
        # Process queries
        result = clusterer.process_queries(queries)
        
        # Verify the full pipeline worked
        assert len(result) == 2
        assert result[0].cluster_id == 0
        assert result[1].cluster_id == 1
        assert len(result[0].queries) == 2
        assert len(result[1].queries) == 1 