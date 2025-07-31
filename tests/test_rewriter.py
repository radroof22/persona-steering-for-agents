import pytest
from unittest.mock import Mock, patch

from app.models.schema import Query, PersonalizedRewrite
from app.services.rewriter import StylePersonalizer, create_style_personalizer
from app.services.llm_provider import LLMProvider


class TestStylePersonalizer:
    """Test cases for style personalization functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm_provider = Mock(spec=LLMProvider)
        self.personalizer = StylePersonalizer(self.mock_llm_provider)
        
        # Load test data from mock files
        try:
            from app.utils.loader import load_mock_data
            users_df, queries_df = load_mock_data()
            
            # Convert to Query objects (take first few for testing)
            self.test_queries = [
                Query(user_id=row['user_id'], question=row['question'])
                for _, row in queries_df.head(4).iterrows()
            ]
            
            # Create rewritten queries based on actual queries
            self.test_rewritten_queries = [
                "What are the current weather conditions?",
                "What is the weather forecast for today?",
                "What are the current weather conditions?",
                "Where should I eat tonight?"
            ]
            
            # Use actual user descriptions from mock data
            self.test_user_descriptions = dict(zip(users_df['user_id'], users_df['description']))
            
        except FileNotFoundError:
            # Fallback to hardcoded data if files don't exist
            self.test_queries = [
                Query(user_id="user1", question="What's the weather like?"),
                Query(user_id="user2", question="How's the weather?"),
            ]
            
            self.test_rewritten_queries = [
                "What are the current weather conditions?",
                "What is the weather forecast for today?",
            ]
            
            self.test_user_descriptions = {
                "user1": "A formal business professional who prefers concise, professional language",
                "user2": "A casual tech enthusiast who uses informal language and tech jargon",
            }
    
    def test_create_personalization_prompt(self):
        """Test personalization prompt creation."""
        rewritten_query = "What are the current weather conditions?"
        user_description = "A formal business professional"
        
        prompt = self.personalizer._create_personalization_prompt(rewritten_query, user_description)
        
        # Check that prompt contains both elements
        assert rewritten_query in prompt
        assert user_description in prompt
        assert "Rewrite the following text" in prompt
    
    @patch.object(StylePersonalizer, '_create_personalization_prompt')
    def test_personalize_query_success(self, mock_create_prompt):
        """Test successful query personalization."""
        # Mock the prompt creation
        mock_create_prompt.return_value = "test prompt"
        
        # Mock the LLM provider response
        self.mock_llm_provider.rewrite_query.return_value = "Personalized weather query"
        
        rewritten_query = "What are the current weather conditions?"
        user_description = "A formal business professional"
        
        result = self.personalizer.personalize_query(rewritten_query, user_description)
        
        # Check that LLM provider was called
        self.mock_llm_provider.rewrite_query.assert_called_once_with(rewritten_query)
        
        # Check result
        assert result == "Personalized weather query"
    
    def test_personalize_query_fallback(self):
        """Test query personalization with fallback on error."""
        # Mock the LLM provider to raise an exception
        self.mock_llm_provider.rewrite_query.side_effect = Exception("LLM error")
        
        rewritten_query = "What are the current weather conditions?"
        user_description = "A formal business professional"
        
        result = self.personalizer.personalize_query(rewritten_query, user_description)
        
        # Should return original rewritten query as fallback
        assert result == rewritten_query
    
    def test_create_personalized_rewrites(self):
        """Test creating personalized rewrites for multiple users."""
        # Mock the LLM provider
        self.mock_llm_provider.rewrite_query.side_effect = [
            "Formal weather query",
            "Casual weather query"
        ]
        
        personalized_rewrites = self.personalizer.create_personalized_rewrites(
            self.test_queries,
            self.test_rewritten_queries,
            self.test_user_descriptions
        )
        
        # Check that we get the expected number of rewrites
        assert len(personalized_rewrites) == len(self.test_queries)
        
        # Check structure of each rewrite
        for rewrite in personalized_rewrites:
            assert isinstance(rewrite, PersonalizedRewrite)
            assert hasattr(rewrite, 'user_id')
            assert hasattr(rewrite, 'original_query')
            assert hasattr(rewrite, 'rewritten_query')
            assert hasattr(rewrite, 'personalized_query')
            
            # Check that user_id is in our test data
            assert rewrite.user_id in self.test_user_descriptions
            
            # Check that original and rewritten queries match
            assert rewrite.original_query in [q.question for q in self.test_queries]
            assert rewrite.rewritten_query in self.test_rewritten_queries
    
    def test_create_personalized_rewrites_missing_user(self):
        """Test creating personalized rewrites with missing user description."""
        # Add a query for a user not in descriptions
        queries_with_missing_user = self.test_queries + [
            Query(user_id="user3", question="What's the time?")
        ]
        
        rewritten_queries_with_missing = self.test_rewritten_queries + [
            "What is the current time?"
        ]
        
        # Mock the LLM provider
        self.mock_llm_provider.rewrite_query.side_effect = [
            "Formal weather query",
            "Casual weather query",
            "General time query"
        ]
        
        personalized_rewrites = self.personalizer.create_personalized_rewrites(
            queries_with_missing_user,
            rewritten_queries_with_missing,
            self.test_user_descriptions
        )
        
        # Should still create rewrites for all queries
        assert len(personalized_rewrites) == len(queries_with_missing_user)
        
        # Check that missing user gets default description
        user3_rewrite = next(r for r in personalized_rewrites if r.user_id == "user3")
        assert user3_rewrite is not None
    
    def test_create_personalized_rewrites_empty_input(self):
        """Test creating personalized rewrites with empty input."""
        personalized_rewrites = self.personalizer.create_personalized_rewrites(
            [], [], {}
        )
        
        assert len(personalized_rewrites) == 0
    
    def test_create_style_personalizer_factory(self):
        """Test the factory function."""
        # Test with default LLM provider
        personalizer = create_style_personalizer()
        assert isinstance(personalizer, StylePersonalizer)
        
        # Test with custom LLM provider
        mock_llm = Mock(spec=LLMProvider)
        personalizer = create_style_personalizer(mock_llm)
        assert isinstance(personalizer, StylePersonalizer)
        assert personalizer.llm_provider == mock_llm


class TestPersonalizedRewriteModel:
    """Test cases for the PersonalizedRewrite Pydantic model."""
    
    def test_personalized_rewrite_creation(self):
        """Test creating a PersonalizedRewrite instance."""
        rewrite = PersonalizedRewrite(
            user_id="user1",
            original_query="What's the weather?",
            rewritten_query="What are the current weather conditions?",
            personalized_query="Please provide the current weather conditions."
        )
        
        assert rewrite.user_id == "user1"
        assert rewrite.original_query == "What's the weather?"
        assert rewrite.rewritten_query == "What are the current weather conditions?"
        assert rewrite.personalized_query == "Please provide the current weather conditions."
    
    def test_personalized_rewrite_validation(self):
        """Test PersonalizedRewrite validation."""
        # Should not raise an exception for valid data
        PersonalizedRewrite(
            user_id="user1",
            original_query="test",
            rewritten_query="test",
            personalized_query="test"
        )
        
        # Should raise validation error for missing required fields
        with pytest.raises(ValueError):
            PersonalizedRewrite(
                user_id="user1",
                original_query="test",
                # Missing rewritten_query and personalized_query
            ) 