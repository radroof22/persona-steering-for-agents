import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Optional
import logging
from app.models.schema import ClusteredQuery
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMProvider:
    """Provides access to Hugging Face LLM for query rewriting."""
    
    def __init__(self, model_name: str = "NousResearch/Hermes-2-Pro-Mistral-7B"):
        """
        Initialize the LLM provider.
        
        Args:
            model_name: Name of the Hugging Face model to use
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def _load_model(self) -> None:
        """Load the tokenizer and model."""
        if self.tokenizer is None or self.model is None:
            logger.info(f"Loading model {self.model_name} on {self.device}...")
            
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    low_cpu_mem_usage=True
                )
                
                # Set pad token if not present
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    
                logger.info("Model loaded successfully!")
                
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                raise
    
    def rewrite_query(self, query: str, max_length: int = 512) -> str:
        """
        Rewrite a query using the LLM.
        
        Args:
            query: The original query to rewrite
            max_length: Maximum length for the generated response
            
        Returns:
            The rewritten query
        """
        self._load_model()
        
        try:
            # Tokenize input
            inputs = self.tokenizer(query, return_tensors="pt", truncation=True, max_length=max_length)
            
            # Move to device
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the assistant's response (everything after the prompt)
            assistant_response = response[len(query):].strip()
            
            # Clean up the response
            if assistant_response.startswith("Here's the rewritten query:"):
                assistant_response = assistant_response[len("Here's the rewritten query:"):].strip()
            
            # Remove quotes if present
            if assistant_response.startswith('"') and assistant_response.endswith('"'):
                assistant_response = assistant_response[1:-1]
            
            return assistant_response
            
        except Exception as e:
            logger.error(f"Error rewriting query: {e}")
            # Return original query as fallback
            return query
    
    def rewrite_multiple_queries(self, clustered_queries: List[ClusteredQuery]) -> List[str]:
        """
        Rewrite representative queries to ensure they capture all specific details from their cluster members.
        
        Args:
            clustered_queries: List[ClusteredQuery] containing query clusters
            
        Returns:
            List[str] of comprehensive rewritten queries that preserve all cluster-specific information
        """
        rewritten_queries = []
        
        for i, cluster in enumerate(clustered_queries):
            logger.info(f"Processing cluster {i+1}/{len(clustered_queries)}")
            rewritten = self._create_comprehensive_rewrite(cluster, i)
            rewritten_queries.append(rewritten)
            
        return rewritten_queries

    def _create_comprehensive_rewrite(self, cluster: ClusteredQuery, cluster_index: int) -> str:
        """
        Create a comprehensive rewrite for a single cluster that captures all specific details.
        
        Args:
            cluster: ClusteredQuery object containing the cluster data
            cluster_index: Index of the cluster for logging
            
        Returns:
            str containing the comprehensive rewritten query
        """
        # Collect all specific details from cluster queries
        cluster_details = "\n".join([
            f"- {query.question}" 
            for query in cluster.queries
        ])
        
        # Create a prompt that ensures the rewritten query captures all specifics
        comprehensive_prompt = f"""Analyze these related queries and create a single comprehensive query that captures all specific details:

        Related Queries:
        {cluster_details}
        
        Create a single query that:
        1. Combines all the related queries into one comprehensive query
        2. Incorporates all specific details, constraints, and requirements from the related queries
        3. Ensures no information is lost from any of the specific queries
        4. Maintains clarity and natural language flow
        
        Provide only the rewritten query, nothing else."""
        
        rewritten = self.rewrite_query(comprehensive_prompt)
        logger.debug(f"Cluster {cluster_index+1} rewritten query: {rewritten[:100]}...")
        
        return rewritten


def create_llm_provider(model_name: Optional[str] = None) -> LLMProvider:
    """
    Factory function to create an LLMProvider instance.
    
    Args:
        model_name: Optional model name override
        
    Returns:
        LLMProvider instance
    """
    if model_name is None:
        model_name = "NousResearch/Hermes-2-Pro-Mistral-7B"
    
    return LLMProvider(model_name=model_name) 