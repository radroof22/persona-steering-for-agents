import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict

from backend.services.rewriter import StylePersonalizer, create_style_personalizer
from backend.models.schema import Query, PersonalizedRewrite
from backend.services.llm_provider import LLMProvider


class TestStylePersonalizer:
    """Test suite for StylePersonalizer class."""
    
    @pytest.fixture
    def sample_queries(self) -> List[Query]:
        """Sample queries for testing."""
        return [
            Query(user_id="user1", question="What is the weather like today?"),
            Query(user_id="user2", question="How do I reset my password?"),
            Query(user_id="user3", question="What time does the store close?"),
        ]
    
    @pytest.fixture
    def sample_rewritten_queries(self) -> List[str]:
        """Sample rewritten queries for testing."""
        return [
            "What is the current weather condition today?",
            "How can I reset my password?",
            "What are the store's operating hours?",
        ]
    
    @pytest.fixture
    def sample_user_descriptions(self) -> Dict[str, str]:
        """Sample user descriptions for testing."""
        return {
            "user1": "A casual, friendly person who prefers simple explanations",
            "user2": "A technical professional who likes detailed, precise information",
            "user3": "A busy parent who needs quick, practical answers",
        }
    
    @pytest.fixture
    def mock_llm_provider(self) -> Mock:
        """Create a mock LLM provider."""
        return Mock(spec=LLMProvider)
    
    @pytest.fixture
    def style_personalizer(self, mock_llm_provider) -> StylePersonalizer:
        """Create a StylePersonalizer instance with mocked LLM provider."""
        return StylePersonalizer(llm_provider=mock_llm_provider)
    
    def test_init_with_llm_provider(self, mock_llm_provider):
        """Test StylePersonalizer initialization with LLM provider."""
        personalizer = StylePersonalizer(llm_provider=mock_llm_provider)
        
        assert personalizer.llm_provider is mock_llm_provider
    
    def test_init_without_llm_provider(self):
        """Test StylePersonalizer initialization without LLM provider."""
        with patch('backend.services.rewriter.LLMProvider') as mock_llm_provider_class:
            mock_provider = Mock(spec=LLMProvider)
            mock_llm_provider_class.return_value = mock_provider
            
            personalizer = StylePersonalizer()
            
            assert personalizer.llm_provider is mock_provider
            mock_llm_provider_class.assert_called_once()
    
    def test_create_personalization_prompt(self, style_personalizer):
        """Test personalization prompt creation."""
        rewritten_query = "What is the weather like?"
        user_description = "A casual user"
        
        prompt = style_personalizer._create_personalization_prompt(rewritten_query, user_description)
        
        expected_prompt = f"""Rewrite the following text in the tone and style of a user described as: "{user_description}"

Text to rewrite: "{rewritten_query}"

Provide only the personalized version, nothing else."""
        
        assert prompt == expected_prompt
    
    @patch.object(StylePersonalizer, 'personalize_query')
    def test_create_personalized_rewrites_normal_case(
        self, 
        mock_personalize_query, 
        style_personalizer, 
        sample_queries, 
        sample_rewritten_queries, 
        sample_user_descriptions
    ):
        """Test create_personalized_rewrites with normal case."""
        # Mock personalize_query to return personalized versions
        mock_personalize_query.side_effect = [
            "Hey, what's the weather like today?",
            "Please provide detailed steps for password reset procedure.",
            "Quick: when does the store close?",
        ]
        
        # Create personalized rewrites
        result = style_personalizer.create_personalized_rewrites(
            sample_queries, 
            sample_rewritten_queries, 
            sample_user_descriptions
        )
        
        # Verify personalize_query was called for each query
        assert mock_personalize_query.call_count == 3
        
        # Verify calls to personalize_query
        expected_calls = [
            ((sample_rewritten_queries[0], sample_user_descriptions["user1"]),),
            ((sample_rewritten_queries[1], sample_user_descriptions["user2"]),),
            ((sample_rewritten_queries[2], sample_user_descriptions["user3"]),),
        ]
        mock_personalize_query.assert_has_calls(expected_calls)
        
        # Verify results
        assert len(result) == 3
        
        # Check first result
        assert result[0].user_id == "user1"
        assert result[0].original_query == "What is the weather like today?"
        assert result[0].rewritten_query == "What is the current weather condition today?"
        assert result[0].personalized_query == "Hey, what's the weather like today?"
        
        # Check second result
        assert result[1].user_id == "user2"
        assert result[1].original_query == "How do I reset my password?"
        assert result[1].rewritten_query == "How can I reset my password?"
        assert result[1].personalized_query == "Please provide detailed steps for password reset procedure."
        
        # Check third result
        assert result[2].user_id == "user3"
        assert result[2].original_query == "What time does the store close?"
        assert result[2].rewritten_query == "What are the store's operating hours?"
        assert result[2].personalized_query == "Quick: when does the store close?"
    
    @patch.object(StylePersonalizer, 'personalize_query')
    def test_create_personalized_rewrites_missing_user_description(
        self, 
        mock_personalize_query, 
        style_personalizer, 
        sample_queries, 
        sample_rewritten_queries
    ):
        """Test create_personalized_rewrites with missing user descriptions."""
        # Use incomplete user descriptions
        incomplete_descriptions = {
            "user1": "A casual user",
            # user2 missing
            "user3": "A busy parent",
        }
        
        # Mock personalize_query to return personalized versions
        mock_personalize_query.side_effect = [
            "Hey, what's the weather like?",
            "How can I reset my password?",  # Should use default description
            "Quick store hours?",
        ]
        
        # Create personalized rewrites
        result = style_personalizer.create_personalized_rewrites(
            sample_queries, 
            sample_rewritten_queries, 
            incomplete_descriptions
        )
        
        # Verify personalize_query was called for each query
        assert mock_personalize_query.call_count == 3
        
        # Verify calls to personalize_query (user2 should get default description)
        expected_calls = [
            ((sample_rewritten_queries[0], "A casual user"),),
            ((sample_rewritten_queries[1], "A general user"),),  # Default description
            ((sample_rewritten_queries[2], "A busy parent"),),
        ]
        mock_personalize_query.assert_has_calls(expected_calls)
        
        # Verify results
        assert len(result) == 3
        assert result[1].user_id == "user2"
        assert result[1].personalized_query == "How can I reset my password?"
    
    @patch.object(StylePersonalizer, 'personalize_query')
    def test_create_personalized_rewrites_empty_input(
        self, 
        mock_personalize_query, 
        style_personalizer
    ):
        """Test create_personalized_rewrites with empty input."""
        # Create personalized rewrites with empty lists
        result = style_personalizer.create_personalized_rewrites(
            [], 
            [], 
            {}
        )
        
        # Verify personalize_query was not called
        mock_personalize_query.assert_not_called()
        
        # Verify empty result
        assert result == []
    
    @patch.object(StylePersonalizer, 'personalize_query')
    def test_create_personalized_rewrites_single_query(
        self, 
        mock_personalize_query, 
        style_personalizer
    ):
        """Test create_personalized_rewrites with single query."""
        single_query = [Query(user_id="user1", question="What is the weather?")]
        single_rewritten = ["What is the current weather condition?"]
        descriptions = {"user1": "A casual user"}
        
        # Mock personalize_query
        mock_personalize_query.return_value = "Hey, what's the weather like?"
        
        # Create personalized rewrites
        result = style_personalizer.create_personalized_rewrites(
            single_query, 
            single_rewritten, 
            descriptions
        )
        
        # Verify personalize_query was called once
        mock_personalize_query.assert_called_once_with(
            "What is the current weather condition?", 
            "A casual user"
        )
        
        # Verify result
        assert len(result) == 1
        assert result[0].user_id == "user1"
        assert result[0].original_query == "What is the weather?"
        assert result[0].rewritten_query == "What is the current weather condition?"
        assert result[0].personalized_query == "Hey, what's the weather like?"
    
    @patch.object(StylePersonalizer, 'personalize_query')
    def test_create_personalized_rewrites_mismatched_lengths(
        self, 
        mock_personalize_query, 
        style_personalizer, 
        sample_queries, 
        sample_user_descriptions
    ):
        """Test create_personalized_rewrites with mismatched query and rewritten lengths."""
        # Use fewer rewritten queries than original queries
        fewer_rewritten = [
            "What is the current weather condition today?",
            "How can I reset my password?",
        ]
        
        # Mock personalize_query
        mock_personalize_query.side_effect = [
            "Hey, what's the weather like?",
            "Please provide detailed steps for password reset.",
        ]
        
        # Create personalized rewrites (should only process first 2 queries)
        result = style_personalizer.create_personalized_rewrites(
            sample_queries, 
            fewer_rewritten, 
            sample_user_descriptions
        )
        
        # Verify personalize_query was called only for available rewritten queries
        assert mock_personalize_query.call_count == 2
        
        # Verify results (should only have 2 results)
        assert len(result) == 2
        assert result[0].user_id == "user1"
        assert result[1].user_id == "user2"
    
    def test_create_personalized_rewrites_personalize_query_exception(
        self, 
        style_personalizer, 
        sample_queries, 
        sample_rewritten_queries, 
        sample_user_descriptions,
        mock_llm_provider
    ):
        """Test create_personalized_rewrites when personalize_query raises an exception."""
        # Mock LLM provider to raise an exception for the second query
        mock_llm_provider.generate.side_effect = [
            "Hey, what's the weather like?",  # First call succeeds
            Exception("LLM service unavailable"),  # Second call fails
            "Quick store hours?",  # Third call succeeds
        ]
        
        # Create personalized rewrites
        result = style_personalizer.create_personalized_rewrites(
            sample_queries, 
            sample_rewritten_queries, 
            sample_user_descriptions
        )
        
        # Verify LLM provider was called for all queries
        assert mock_llm_provider.generate.call_count == 3
        
        # Verify results (second query should fall back to rewritten query)
        assert len(result) == 3
        assert result[0].personalized_query == "Hey, what's the weather like?"
        assert result[1].personalized_query == "How can I reset my password?"  # Fallback to rewritten
        assert result[2].personalized_query == "Quick store hours?"
    
    def test_create_personalized_rewrites_all_exceptions(
        self, 
        style_personalizer, 
        sample_queries, 
        sample_rewritten_queries, 
        sample_user_descriptions,
        mock_llm_provider
    ):
        """Test create_personalized_rewrites when all personalize_query calls fail."""
        # Mock LLM provider to raise exceptions for all queries
        mock_llm_provider.generate.side_effect = [
            Exception("LLM service unavailable"),
            Exception("Network error"),
            Exception("Timeout"),
        ]
        
        # Create personalized rewrites
        result = style_personalizer.create_personalized_rewrites(
            sample_queries, 
            sample_rewritten_queries, 
            sample_user_descriptions
        )
        
        # Verify LLM provider was called for all queries
        assert mock_llm_provider.generate.call_count == 3
        
        # Verify results (all should fall back to rewritten queries)
        assert len(result) == 3
        assert result[0].personalized_query == "What is the current weather condition today?"
        assert result[1].personalized_query == "How can I reset my password?"
        assert result[2].personalized_query == "What are the store's operating hours?"


class TestPersonalizeQuery:
    """Test suite for personalize_query method."""
    
    @pytest.fixture
    def mock_llm_provider(self) -> Mock:
        """Create a mock LLM provider."""
        return Mock(spec=LLMProvider)
    
    @pytest.fixture
    def style_personalizer(self, mock_llm_provider) -> StylePersonalizer:
        """Create a StylePersonalizer instance with mocked LLM provider."""
        return StylePersonalizer(llm_provider=mock_llm_provider)
    
    def test_personalize_query_success(self, style_personalizer, mock_llm_provider):
        """Test successful query personalization."""
        rewritten_query = "What is the weather like?"
        user_description = "A casual user"
        expected_personalized = "Hey, what's the weather like today?"
        
        # Mock LLM provider response
        mock_llm_provider.generate.return_value = expected_personalized
        
        # Personalize query
        result = style_personalizer.personalize_query(rewritten_query, user_description)
        
        # Verify LLM provider was called
        mock_llm_provider.generate.assert_called_once()
        
        # Verify the prompt contains expected content
        call_args = mock_llm_provider.generate.call_args[0][0]
        assert user_description in call_args
        assert rewritten_query in call_args
        
        # Verify result
        assert result == expected_personalized
    
    def test_personalize_query_exception_fallback(self, style_personalizer, mock_llm_provider):
        """Test personalize_query falls back to rewritten query on exception."""
        rewritten_query = "What is the weather like?"
        user_description = "A casual user"
        
        # Mock LLM provider to raise exception
        mock_llm_provider.generate.side_effect = Exception("LLM service unavailable")
        
        # Personalize query
        result = style_personalizer.personalize_query(rewritten_query, user_description)
        
        # Verify LLM provider was called
        mock_llm_provider.generate.assert_called_once()
        
        # Verify fallback to rewritten query
        assert result == rewritten_query


class TestPersonalizeResponse:
    """Test suite for personalize_response method."""
    
    @pytest.fixture
    def mock_llm_provider(self) -> Mock:
        """Create a mock LLM provider."""
        return Mock(spec=LLMProvider)
    
    @pytest.fixture
    def style_personalizer(self, mock_llm_provider) -> StylePersonalizer:
        """Create a StylePersonalizer instance with mocked LLM provider."""
        return StylePersonalizer(llm_provider=mock_llm_provider)
    
    def test_personalize_response_success(self, style_personalizer, mock_llm_provider):
        """Test successful response personalization."""
        original_query = "What's the weather like?"
        user_description = "A casual user"
        summarized_query = "What is the current weather condition?"
        summarized_response = "The weather is sunny with a temperature of 75°F."
        expected_personalized = "Hey! It's a beautiful sunny day at 75°F - perfect weather!"
        
        # Mock LLM provider response
        mock_llm_provider.generate.return_value = expected_personalized
        
        # Personalize response
        result = style_personalizer.personalize_response(
            original_query, 
            user_description, 
            summarized_query, 
            summarized_response
        )
        
        # Verify LLM provider was called
        mock_llm_provider.generate.assert_called_once()
        
        # Verify the prompt contains expected content
        call_args = mock_llm_provider.generate.call_args[0][0]
        assert user_description in call_args
        assert original_query in call_args
        assert summarized_query in call_args
        assert summarized_response in call_args
        
        # Verify result
        assert result == expected_personalized.strip()


class TestCreateStylePersonalizer:
    """Test suite for create_style_personalizer factory function."""
    
    def test_create_style_personalizer_with_llm_provider(self):
        """Test create_style_personalizer with LLM provider."""
        mock_llm_provider = Mock(spec=LLMProvider)
        
        personalizer = create_style_personalizer(llm_provider=mock_llm_provider)
        
        assert isinstance(personalizer, StylePersonalizer)
        assert personalizer.llm_provider is mock_llm_provider
    
    def test_create_style_personalizer_without_llm_provider(self):
        """Test create_style_personalizer without LLM provider."""
        with patch('backend.services.rewriter.LLMProvider') as mock_llm_provider_class:
            mock_provider = Mock(spec=LLMProvider)
            mock_llm_provider_class.return_value = mock_provider
            
            personalizer = create_style_personalizer()
            
            assert isinstance(personalizer, StylePersonalizer)
            assert personalizer.llm_provider is mock_provider
            mock_llm_provider_class.assert_called_once() 