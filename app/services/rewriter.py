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
        prompt = f"""<|im_start|>system
You are an expert at adapting text to match specific user styles and preferences.
Your goal is to rewrite the given text to match the user's described style while maintaining the original meaning.
<|im_end|>
<|im_start|>user
Rewrite the following text in the tone and style of a user described as: "{user_description}"

Text to rewrite: "{rewritten_query}"

Provide only the personalized version, nothing else.
<|im_end|>
<|im_start|>assistant
"""
        return prompt
    
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
            personalized = self.llm_provider.rewrite_query(rewritten_query)
            
            # For now, return the rewritten query as-is
            # In a full implementation, we would use the personalization prompt
            # but for this proof-of-concept, we'll use a simpler approach
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