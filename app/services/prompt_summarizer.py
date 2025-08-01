from typing import List
import logging
from app.models.schema import ClusteredQuery
from app.services.llm_provider import LLMProvider

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptSummarizer:
    """Service for creating and executing summarization prompts from query clusters."""
    
    def __init__(self, llm_provider: LLMProvider):
        """
        Initialize the prompt summarizer.
        
        Args:
            llm_provider: LLMProvider instance for generating responses
        """
        self.llm_provider = llm_provider
    
    def summarize_clusters(self, clustered_queries: List[ClusteredQuery]) -> List[str]:
        """
        Generate summarization responses for query clusters.
        
        Args:
            clustered_queries: List[ClusteredQuery] containing query clusters
            
        Returns:
            List[str] of summarized responses for each cluster
        """
        summarized_responses = []
        
        for i, cluster in enumerate(clustered_queries):
            logger.info(f"Processing cluster {i+1}/{len(clustered_queries)}")
            response = self._create_cluster_summary(cluster, i)
            summarized_responses.append(response)
            
        return summarized_responses
    
    def _create_cluster_summary(self, cluster: ClusteredQuery, cluster_index: int) -> str:
        """
        Create a comprehensive summary for a single cluster.
        
        Args:
            cluster: ClusteredQuery object containing the cluster data
            cluster_index: Index of the cluster for logging
            
        Returns:
            str containing the summarized response
        """
        # Collect all specific details from cluster queries
        cluster_details = "\n".join([
            f"- {query.question}" 
            for query in cluster.queries
        ])
        
        # Create a prompt that ensures the summary captures all specifics
        summary_prompt = f"""Combine these related queries into one comprehensive query. It should be one sentence but include all the details from the queries:

Related queries:
{cluster_details}

Comprehensive query:"""
        
        response = self.llm_provider.generate(summary_prompt)
        logger.debug(f"Cluster {cluster_index+1} summary: {response[:100]}...")
        
        return response


def create_prompt_summarizer(llm_provider: LLMProvider) -> PromptSummarizer:
    """
    Factory function to create a PromptSummarizer instance.
    
    Args:
        llm_provider: LLMProvider instance
        
    Returns:
        PromptSummarizer instance
    """
    return PromptSummarizer(llm_provider) 