from typing import List
import logging
from backend.models.schema import ClusteredQuery
from backend.services.llm_provider import LLMProvider

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
    
    def summarize_cluster(self, cluster: ClusteredQuery) -> str:
        """
        Generate a summarized query for a single cluster.
        
        Args:
            cluster: ClusteredQuery containing the cluster data
            
        Returns:
            str: Summarized query that represents all queries in the cluster
        """
        logger.info(f"Summarizing cluster {cluster.cluster_id} with {len(cluster.queries)} queries")
        
        # Extract all query texts from the cluster
        query_texts = [query.question for query in cluster.queries]
        
        # Create a prompt to summarize the queries
        summarization_prompt = f"""
        Analyze the following related user queries and create a single, comprehensive question that captures the main intent and information needs of all queries:

        Queries:
        {chr(10).join([f"- {query}" for query in query_texts])}

        Please provide a single, well-formed question that:
        1. Captures the common theme across all queries
        2. Is clear and specific enough to get a comprehensive answer
        3. Maintains the technical level and domain context
        4. Can be answered in a way that addresses all the original queries

        Summarized Question:""" if len(cluster.queries) > 1 else cluster.queries[0].question
        
        # Generate the summarized query
        summarized_query = self.llm_provider.generate(summarization_prompt)
        
        logger.info(f"Generated summarized query for cluster {cluster.cluster_id}")
        return summarized_query.strip()

    def summarize_cluster_queries(self, clustered_queries: List[ClusteredQuery]) -> List[str]:
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