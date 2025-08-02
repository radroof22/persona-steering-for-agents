from typing import List, Dict, Optional
from app.models.schema import Query, PersonalizedRewrite
from app.services.llm_provider import LLMProvider
import logging

logger = logging.getLogger(__name__)


class StylePersonalizer:
    """Personalizes rewritten queries based on user style preferences."""
    
    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        """
        Initialize the style personalizer.
        
        Args:
            llm_provider: LLM provider for personalization
        """
        self.llm_provider = llm_provider or LLMProvider()
    
    def _create_personalization_prompt(self, rewritten_query: str, user_description: str) -> str:
        """
        Create a prompt for personalizing the rewritten query.
        
        Args:
            rewritten_query: The LLM-rewritten query
            user_description: User's style description
            
        Returns:
            Formatted prompt for personalization
        """
        prompt = f"""Rewrite the following text in the tone and style of a user described as: "{user_description}"

Text to rewrite: "{rewritten_query}"

Provide only the personalized version, nothing else."""
        
        return prompt
    
    def personalize_response(
        self, 
        original_query: str,
        user_description: str, 
        summarized_query: str, 
        summarized_response: str
    ) -> str:
        """
        Personalize a response for a specific user and original query.
        
        Args:
            original_query: The user's original question
            user_description: User's description for personalization
            summarized_query: The summarized query for the cluster
            summarized_response: The LLM response to the summarized query
            
        Returns:
            Personalized response tailored to the user and original query
        """
        logger.info(f"Personalizing response for user query: {original_query[:50]}...")
        
        personalization_prompt = f"""
        You are a helpful assistant that personalizes responses based on user context.
        
        User Description: {user_description}
        Original User Question: {original_query}
        Cluster Summary Question: {summarized_query}
        General Response: {summarized_response}
        
        Please provide a personalized response that:
        1. Directly addresses the user's original question
        2. Is tailored to their background and expertise level
        3. Uses the general response as a foundation but adapts it for this specific user
        4. Maintains the tone and style appropriate for the user's description
        5. Focuses on aspects most relevant to the user's context
        
        Personalized Response:"""
        
        personalized_response = self.llm_provider.generate(personalization_prompt)
        
        logger.info(f"Generated personalized response")
        return personalized_response.strip()

    def personalize_query(self, rewritten_query: str, user_description: str) -> str:
        """
        Personalize a rewritten query based on user description.
        
        Args:
            rewritten_query: The LLM-rewritten query
            user_description: User's style description
            
        Returns:
            Personalized query
        """
        try:
            # Create personalization prompt
            prompt = self._create_personalization_prompt(rewritten_query, user_description)
            
            # Use the LLM provider to generate personalized version
            personalized = self.llm_provider.generate(prompt)
            
            return personalized
            
        except Exception as e:
            logger.error(f"Error personalizing query: {e}")
            # Return rewritten query as fallback
            return rewritten_query
    
    def create_personalized_rewrites(
        self, 
        queries: List[Query], 
        rewritten_queries: List[str], 
        user_descriptions: Dict[str, str]
    ) -> List[PersonalizedRewrite]:
        """
        Create personalized rewrites for all users.
        
        Args:
            queries: List of original queries
            rewritten_queries: List of LLM-rewritten queries
            user_descriptions: Mapping of user_id to description
            
        Returns:
            List of PersonalizedRewrite objects
        """
        personalized_rewrites = []
        
        for query, rewritten_query in zip(queries, rewritten_queries):
            user_description = user_descriptions.get(query.user_id, "A general user")
            
            personalized_query = self.personalize_query(rewritten_query, user_description)
            
            personalized_rewrite = PersonalizedRewrite(
                user_id=query.user_id,
                original_query=query.question,
                rewritten_query=rewritten_query,
                personalized_query=personalized_query
            )
            
            personalized_rewrites.append(personalized_rewrite)
        
        return personalized_rewrites


def create_style_personalizer(llm_provider: Optional[LLMProvider] = None) -> StylePersonalizer:
    """
    Factory function to create a StylePersonalizer instance.
    
    Args:
        llm_provider: Optional LLM provider
        
    Returns:
        StylePersonalizer instance
    """
    return StylePersonalizer(llm_provider=llm_provider) 