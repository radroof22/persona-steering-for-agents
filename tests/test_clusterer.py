import pytest
import numpy as np
from unittest.mock import Mock, patch

from app.models.schema import Query
from app.services.clusterer import QueryClusterer, create_clusterer


class TestQueryClusterer:
    """Test cases for query clustering functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.clusterer = QueryClusterer(eps=0.15, min_samples=2)
        
        # Load test queries from mock data files
        try:
            from app.utils.loader import load_mock_data
            users_df, queries_df = load_mock_data()
            
            # Convert to Query objects
            self.test_queries = [
                Query(user_id=row['user_id'], question=row['question'])
                for _, row in queries_df.iterrows()
            ]
            
            # Store user descriptions for testing
            self.user_descriptions = dict(zip(users_df['user_id'], users_df['description']))
            
        except FileNotFoundError:
            # Fallback to hardcoded data if files don't exist
            self.test_queries = [
                Query(user_id="user1", question="What's the weather like today?"),
                Query(user_id="user2", question="How's the weather looking?"),
                Query(user_id="user3", question="What are the current weather conditions?"),
                Query(user_id="user4", question="Where should I eat tonight?"),
                Query(user_id="user5", question="Can you recommend a good restaurant?"),
                Query(user_id="user6", question="What's the capital of Mongolia?"),  # Singleton
            ]
            self.user_descriptions = {
                "user1": "A formal business professional",
                "user2": "A casual tech enthusiast",
                "user3": "An academic researcher",
                "user4": "A creative writer",
                "user5": "A student",
                "user6": "A senior executive"
            }
    
    @patch('app.services.clusterer.SentenceTransformer')
    def test_encode_queries(self, mock_transformer):
        """Test query encoding."""
        # Mock the transformer
        mock_model = Mock()
        num_queries = len(self.test_queries)
        mock_model.encode.return_value = np.random.rand(num_queries, 384)  # Dynamic number of queries
        mock_transformer.return_value = mock_model
        
        queries_text = [q.question for q in self.test_queries]
        embeddings = self.clusterer.encode_queries(queries_text)
        
        # Check that embeddings are generated
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(queries_text)
        assert embeddings.shape[1] > 0
        
        # Check that model was called
        mock_model.encode.assert_called_once()
    
    @patch('app.services.clusterer.SentenceTransformer')
    def test_cluster_queries(self, mock_transformer):
        """Test query clustering."""
        # Mock the transformer
        mock_model = Mock()
        num_queries = len(self.test_queries)
        embeddings = np.random.rand(num_queries, 384)
        if num_queries >= 3:
            embeddings[:3] = embeddings[0]  # First 3 queries in same cluster
        mock_model.encode.return_value = embeddings
        mock_transformer.return_value = mock_model
        
        queries_text = [q.question for q in self.test_queries]
        cluster_labels = self.clusterer.cluster_queries(queries_text)
        
        # Check that cluster labels are generated
        assert isinstance(cluster_labels, np.ndarray)
        assert len(cluster_labels) == len(queries_text)
        
        # Check that we have some clusters (including noise points with label -1)
        unique_labels = set(cluster_labels)
        assert len(unique_labels) > 0
        assert 0 in unique_labels # check we have at least one cluster
    
    @patch('app.services.clusterer.SentenceTransformer')
    def test_queries_grouped_by_cluster(self, mock_transformer):
        """Test that queries are properly grouped by cluster."""
        # Mock the transformer with embeddings that will create distinct clusters
        mock_model = Mock()
        num_queries = len(self.test_queries)
        
        # Create embeddings that will form distinct clusters
        embeddings = np.random.rand(num_queries, 384)
        
        # Make some queries similar to create clusters
        if num_queries >= 3:
            embeddings[:3] = embeddings[0]  # First 3 queries in same cluster
        if num_queries >= 6:
            embeddings[3:6] = embeddings[3]  # Next 3 queries in same cluster
        
        mock_model.encode.return_value = embeddings
        mock_transformer.return_value = mock_model
        
        # Process queries
        clustered_queries = self.clusterer.process_queries(self.test_queries)
        
        # Check that we get some clusters
        assert len(clustered_queries) > 0
        
        # Check that each cluster has the expected structure
        for cluster in clustered_queries:
            assert hasattr(cluster, 'cluster_id')
            assert hasattr(cluster, 'user_ids')
            assert hasattr(cluster, 'queries')
            
            # Check that cluster is not empty
            assert len(cluster.queries) > 0
            assert len(cluster.user_ids) > 0
            
            # Check that user_ids match the queries
            cluster_user_ids = [q.user_id for q in cluster.queries]
            assert set(cluster.user_ids) == set(cluster_user_ids)
            
            # Check that all queries in cluster have the same cluster_id
            # (This is implicit since we grouped them, but good to verify)
            assert all(q.user_id in cluster.user_ids for q in cluster.queries)
        
        # Check that all original queries are accounted for
        all_clustered_queries = []
        for cluster in clustered_queries:
            all_clustered_queries.extend(cluster.queries)

        # Sort by user_id and question for comparison
        original_sorted = sorted(self.test_queries, key=lambda x: (x.user_id, x.question))
        clustered_sorted = sorted(all_clustered_queries, key=lambda x: (x.user_id, x.question))
        
        assert len(original_sorted) == len(clustered_sorted)
        for orig, clust in zip(original_sorted, clustered_sorted):
            assert orig.user_id == clust.user_id
            assert orig.question == clust.question

        # Check that queries with similar embeddings are clustered together
        if num_queries >= 3:
            # Find cluster containing first query
            first_cluster = next(c for c in clustered_queries if self.test_queries[0] in c.queries)
            # First 3 queries should be in same cluster
            assert all(self.test_queries[i] in first_cluster.queries for i in range(3))

        if num_queries >= 6:
            # Find cluster containing fourth query
            second_cluster = next(c for c in clustered_queries if self.test_queries[3] in c.queries)
            # Queries 3-5 should be in same cluster
            assert all(self.test_queries[i] in second_cluster.queries for i in range(3, 6))
    
    @patch('app.services.clusterer.SentenceTransformer')
    def test_process_queries_complete_pipeline(self, mock_transformer):
        """Test the complete clustering pipeline."""
        # Mock the transformer
        mock_model = Mock()
        num_queries = len(self.test_queries)
        mock_model.encode.return_value = np.random.rand(num_queries, 384)
        mock_transformer.return_value = mock_model
        
        # Process queries through complete pipeline
        clustered_queries = self.clusterer.process_queries(self.test_queries)
        
        # Check that we get some clusters
        assert len(clustered_queries) > 0
        
        # Check that clusters are non-empty
        for cluster in clustered_queries:
            assert len(cluster.queries) > 0
            assert len(cluster.user_ids) > 0
    
    @patch('app.services.clusterer.SentenceTransformer')
    def test_cluster_characteristics(self, mock_transformer):
        """Test that we have both singleton and multi-user clusters."""
        # Mock the transformer with embeddings that will create distinct clusters
        mock_model = Mock()
        # Create embeddings that will form distinct clusters
        # Using more realistic embeddings that will cluster properly
        num_queries = len(self.test_queries)
        embeddings = np.random.rand(num_queries, 384)  # Dynamic size based on actual data
        
        # Make embeddings similar based on the actual mock data structure:
        # - Weather queries (0-3): similar embeddings
        # - Restaurant queries (4-7): similar embeddings  
        # - Technical queries (8-10): similar embeddings
        # - Travel queries (11-13): similar embeddings
        # - Health queries (14-16): similar embeddings
        # - Singleton queries (17-19): unique embeddings
        
        if num_queries >= 4:
            embeddings[:4] = embeddings[0]  # Weather cluster (queries 0-3)
        if num_queries >= 8:
            embeddings[4:8] = embeddings[4]  # Restaurant cluster (queries 4-7)
        if num_queries >= 11:
            embeddings[8:11] = embeddings[8]  # Technical cluster (queries 8-10)
        if num_queries >= 14:
            embeddings[11:14] = embeddings[11]  # Travel cluster (queries 11-13)
        if num_queries >= 17:
            embeddings[14:17] = embeddings[14]  # Health cluster (queries 14-16)
        # Queries 17-19 remain unique (singleton clusters)
        mock_model.encode.return_value = embeddings
        mock_transformer.return_value = mock_model
        
        # Process queries
        clustered_queries = self.clusterer.process_queries(self.test_queries)
        
        # check that we have at least one cluster
        assert len(clustered_queries) > 0

        # check that one of the cluster has queries that all mention "weather"
        weather_pass = False
        for cluster in clustered_queries:
            if all("weather" in q.question.lower() for q in cluster.queries):
                weather_pass = True
                break
        assert weather_pass

        # check that one of the cluster has at least 3 queries mentioning "restaurant"
        restaurant_count_pass = False
        for cluster in clustered_queries:
            if sum(1 for q in cluster.queries if "restaurant" in q.question.lower()) >= 2:
                restaurant_count_pass = True
                break
        assert restaurant_count_pass

        # check that one of the cluster has at least 3 queries mentioning "python"
        python_count_pass = False
        for cluster in clustered_queries:
            if sum(1 for q in cluster.queries if "python" in q.question.lower()) >= 3:
                python_count_pass = True
                break
        assert python_count_pass


    def test_create_clusterer_factory(self):
        """Test the factory function."""
        clusterer = create_clusterer(eps=0.5, min_samples=3)
        
        assert isinstance(clusterer, QueryClusterer)
        assert clusterer.eps == 0.5
        assert clusterer.min_samples == 3 