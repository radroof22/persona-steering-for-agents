import httpx
import logging
from typing import List, Dict, Any
from config import API_URL

logger = logging.getLogger(__name__)


class APIClient:
    """Client for communicating with the Personalized Query Rewriting API."""
    
    def __init__(self, base_url: str = API_URL):
        self.base_url = base_url
        # Use synchronous client for better Streamlit compatibility
        self.client = httpx.Client(timeout=300.0)
    
    def health_check(self) -> Dict[str, Any]:
        """Check if the API is healthy."""
        try:
            response = self.client.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            logger.error(f"API health check failed: {e}")
            raise
    
    def generate_personalized_rewrites(
        self, 
        users: List[Dict[str, str]], 
        queries: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Generate clustered queries and personalized rewrites.
        
        Args:
            users: List of user dictionaries with user_id and description
            queries: List of query dictionaries with user_id and question
            
        Returns:
            API response with clustered queries and personalized rewrites
        """
        try:
            payload = {
                "users": users,
                "queries": queries
            }
            
            logger.info(f"Sending request to {self.base_url}/generate with {len(users)} users and {len(queries)} queries")
            
            response = self.client.post(
                f"{self.base_url}/generate",
                json=payload
            )
            response.raise_for_status()
            
            logger.info("Successfully received response from generate endpoint")
            return response.json()
            
        except httpx.TimeoutException as e:
            logger.error(f"API request timed out after 5 minutes: {e}")
            raise Exception("Request timed out after 5 minutes. The operation may still be processing on the server.")
        except httpx.RequestError as e:
            logger.error(f"API request failed: {e}")
            raise Exception(f"API request failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in generate_personalized_rewrites: {e}")
            raise
    
    def get_clusters(
        self, 
        users: List[Dict[str, str]], 
        queries: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Get clustered queries without LLM processing.
        
        Args:
            users: List of user dictionaries with user_id and description
            queries: List of query dictionaries with user_id and question
            
        Returns:
            API response with cluster information
        """
        try:
            payload = {
                "users": users,
                "queries": queries
            }
            
            logger.info(f"Sending request to {self.base_url}/clusters with {len(users)} users and {len(queries)} queries")
            
            response = self.client.post(
                f"{self.base_url}/clusters",
                json=payload
            )
            response.raise_for_status()
            
            logger.info("Successfully received response from clusters endpoint")
            return response.json()
            
        except httpx.TimeoutException as e:
            logger.error(f"API request timed out after 5 minutes: {e}")
            raise Exception("Request timed out after 5 minutes. The operation may still be processing on the server.")
        except httpx.RequestError as e:
            logger.error(f"API request failed: {e}")
            raise Exception(f"API request failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in get_clusters: {e}")
            raise
    
    def close(self):
        """Close the HTTP client."""
        self.client.close() 