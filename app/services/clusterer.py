import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from app.models.schema import Query, ClusteredQuery


class QueryClusterer:
    """Clusters similar queries using sentence embeddings and DBSCAN."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", eps: float = 0.15, min_samples: int = 2):
        """
        Initialize the clusterer.
        
        Args:
            model_name: Name of the sentence transformer model
            eps: DBSCAN epsilon parameter for neighborhood size
            min_samples: DBSCAN min_samples parameter
        """
        self.model_name = model_name
        self.eps = eps
        self.min_samples = min_samples
        self.model = None
        self.embeddings = None
        self.cluster_labels = None
        
    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
    
    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """
        Encode queries into embeddings.
        
        Args:
            queries: List of query strings
            
        Returns:
            Array of embeddings
        """
        self._load_model()
        self.embeddings = self.model.encode(queries, show_progress_bar=False)
        # create a square matrix showing distance between each of the embeddings
        self.distance_matrix = cosine_similarity(self.embeddings)
        return self.embeddings
    
    def cluster_queries(self, queries: List[str]) -> np.ndarray:
        """
        Cluster queries using DBSCAN.
        
        Args:
            queries: List of query strings
            
        Returns:
            Array of cluster labels
        """
        # Encode queries
        embeddings = self.encode_queries(queries)
        
        # Apply DBSCAN clustering
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric='cosine')
        self.cluster_labels = dbscan.fit_predict(embeddings)
        
        return self.cluster_labels
    
    def process_queries(self, queries: List[Query]) -> List[ClusteredQuery]:
        """
        Complete clustering pipeline.
        
        Args:
            queries: List of Query objects
            
        Returns:
            List of ClusteredQuery objects
        """
        # Extract query texts
        query_texts = [q.question for q in queries]
        
        # Cluster queries
        cluster_labels = self.cluster_queries(query_texts)
        
        # Group by cluster directly using DBSCAN outputs
        clustered_queries = []
        
        # Process each cluster
        for cluster_id in sorted(set(cluster_labels)):
            if cluster_id == -1:  # Skip noise points
                # treat each noise point as a singleton cluster
                clustered_queries.extend([
                    ClusteredQuery(
                        cluster_id=int(cluster_id),
                        user_ids=[q.user_id],
                        queries=[q]
                    ) for i, q in enumerate(queries) if cluster_labels[i] == -1
                ])
                continue
                
            # Get queries and user_ids for this cluster
            cluster_query_objects = [q for i, q in enumerate(queries) if cluster_labels[i] == cluster_id]
            cluster_user_ids = [q.user_id for q in cluster_query_objects]
            
            # Create ClusteredQuery object
            clustered_query = ClusteredQuery(
                cluster_id=int(cluster_id),
                user_ids=cluster_user_ids,
                queries=cluster_query_objects
            )
            
            clustered_queries.append(clustered_query)
        
        return clustered_queries


def create_clusterer(eps: float = 0.15, min_samples: int = 2) -> QueryClusterer:
    """
    Factory function to create a QueryClusterer instance.
    
    Args:
        eps: DBSCAN epsilon parameter
        min_samples: DBSCAN min_samples parameter
        
    Returns:
        QueryClusterer instance
    """
    return QueryClusterer(eps=eps, min_samples=min_samples) 