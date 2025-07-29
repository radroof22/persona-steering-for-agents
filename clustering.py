import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryClusterer:
    """Class for clustering queries using DBSCAN and TF-IDF"""
    
    def __init__(self, eps: float = 1.2, min_samples: int = 2, 
                 max_features: int = 1000, ngram_range: Tuple[int, int] = (1, 2)):
        """
        Initialize the clusterer
        
        Args:
            eps: DBSCAN epsilon parameter
            min_samples: DBSCAN min_samples parameter
            max_features: Maximum features for TF-IDF
            ngram_range: Range of n-grams for TF-IDF
        """
        self.eps = eps
        self.min_samples = min_samples
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = None
        self.dbscan = None
    
    def vectorize_queries(self, queries: List[str]) -> np.ndarray:
        """
        Vectorize queries using TF-IDF
        
        Args:
            queries: List of query strings
            
        Returns:
            TF-IDF matrix
        """
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words='english',
            ngram_range=self.ngram_range
        )
        
        tfidf_matrix = self.vectorizer.fit_transform(queries)
        logger.info(f"Vectorized {len(queries)} queries into {tfidf_matrix.shape[1]} features")
        
        return tfidf_matrix
    
    def cluster_queries(self, queries: List[str]) -> Tuple[List[int], Dict[str, Any]]:
        """
        Cluster queries using DBSCAN
        
        Args:
            queries: List of query strings
            
        Returns:
            Tuple of (cluster_labels, clustering_info)
        """
        if not queries:
            raise ValueError("No queries provided for clustering")
        
        # Vectorize queries
        tfidf_matrix = self.vectorize_queries(queries)
        
        # Perform DBSCAN clustering
        self.dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        clusters = self.dbscan.fit_predict(tfidf_matrix)
        
        # Debug: Print clustering parameters and results
        logger.info(f"DBSCAN parameters: eps={self.eps}, min_samples={self.min_samples}")
        logger.info(f"Clustering result: {len(set(clusters))} unique cluster IDs, {list(set(clusters))}")
        logger.info(f"Cluster distribution: {[list(clusters).count(i) for i in set(clusters)]}")
        
        # Group queries by cluster
        cluster_results = {}
        for i, cluster_id in enumerate(clusters):
            if cluster_id not in cluster_results:
                cluster_results[cluster_id] = []
            cluster_results[cluster_id].append(queries[i])
        
        # Format results
        formatted_clusters = []
        for cluster_id, cluster_queries in cluster_results.items():
            if cluster_id == -1:  # Noise points
                cluster_info = {
                    "cluster_id": "noise",
                    "size": len(cluster_queries),
                    "queries": cluster_queries
                }
            else:
                cluster_info = {
                    "cluster_id": f"cluster_{cluster_id}",
                    "size": len(cluster_queries),
                    "queries": cluster_queries
                }
            formatted_clusters.append(cluster_info)
        
        # Create clustering info
        clustering_info = {
            "total_queries": len(queries),
            "clusters": formatted_clusters,
            "clustering_algorithm": "DBSCAN",
            "parameters": {
                "eps": self.eps,
                "min_samples": self.min_samples,
                "max_features": self.max_features,
                "ngram_range": self.ngram_range
            },
            "n_clusters": len([c for c in clusters if c != -1]),
            "n_noise": len([c for c in clusters if c == -1])
        }
        
        return clusters, clustering_info
    
    def analyze_distance_distribution(self, queries: List[str], sample_size: int = 1000) -> Dict[str, float]:
        """
        Analyze the distribution of pairwise distances to help choose epsilon
        
        Args:
            queries: List of query strings
            sample_size: Number of random pairs to sample (for large datasets)
            
        Returns:
            Dictionary with distance statistics
        """
        tfidf_matrix = self.vectorize_queries(queries)
        dense_matrix = tfidf_matrix.toarray()
        
        # Sample pairs to avoid memory issues with large datasets
        n_queries = dense_matrix.shape[0]
        if n_queries > 100:
            import random
            distances = []
            for _ in range(sample_size):
                i, j = random.sample(range(n_queries), 2)
                dist = np.linalg.norm(dense_matrix[i] - dense_matrix[j])
                distances.append(dist)
        else:
            # Calculate all pairwise distances
            distances = []
            for i in range(n_queries):
                for j in range(i+1, n_queries):
                    dist = np.linalg.norm(dense_matrix[i] - dense_matrix[j])
                    distances.append(dist)
        
        distances = np.array(distances)
        
        return {
            "mean": float(np.mean(distances)),
            "median": float(np.median(distances)),
            "std": float(np.std(distances)),
            "min": float(np.min(distances)),
            "max": float(np.max(distances)),
            "q25": float(np.percentile(distances, 25)),
            "q75": float(np.percentile(distances, 75))
        }
    
    def evaluate_epsilon_values(self, queries: List[str], eps_values: List[float]) -> List[Dict[str, Any]]:
        """
        Evaluate clustering with different epsilon values
        
        Args:
            queries: List of query strings
            eps_values: List of epsilon values to test
            
        Returns:
            List of clustering results for each epsilon value
        """
        results = []
        
        for eps in eps_values:
            # Temporarily set epsilon
            original_eps = self.eps
            self.eps = eps
            
            try:
                clusters, clustering_info = self.cluster_queries(queries)
                
                results.append({
                    'epsilon': eps,
                    'n_clusters': clustering_info['n_clusters'],
                    'n_noise': clustering_info['n_noise'],
                    'n_clustered': len(queries) - clustering_info['n_noise'],
                    'clustering_ratio': (len(queries) - clustering_info['n_noise']) / len(queries),
                    'clusters': clustering_info['clusters']
                })
            finally:
                # Restore original epsilon
                self.eps = original_eps
        
        return results
    
    def print_cluster_summary(self, clustering_info: Dict[str, Any]):
        """
        Print a summary of clustering results
        
        Args:
            clustering_info: Clustering information dictionary
        """
        print("=== QUERY CLUSTERS IDENTIFIED ===")
        print(f"Total queries processed: {clustering_info['total_queries']}")
        print(f"Number of clusters: {clustering_info['n_clusters']}")
        print(f"Noise points: {clustering_info['n_noise']}")
        
        for cluster in clustering_info['clusters']:
            print(f"\n{cluster['cluster_id']} (Size: {cluster['size']}):")
            for query in cluster['queries']:
                print(f"  - {query}")
        print("================================")
    
    def get_optimal_epsilon_recommendations(self, distance_stats: Dict[str, float]) -> Dict[str, List[float]]:
        """
        Get recommendations for epsilon values based on distance statistics
        
        Args:
            distance_stats: Distance statistics from analyze_distance_distribution
            
        Returns:
            Dictionary with epsilon recommendations for different clustering strategies
        """
        mean_dist = distance_stats['mean']
        median_dist = distance_stats['median']
        q25 = distance_stats['q25']
        q75 = distance_stats['q75']
        
        return {
            "conservative": [q25 * 0.5, q25 * 0.75, q25],
            "balanced": [median_dist * 0.5, median_dist * 0.75, median_dist],
            "aggressive": [q75, q75 * 1.25, q75 * 1.5],
            "current_default": [1.2]  # Current FastAPI default
        } 