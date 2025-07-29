import logging
from typing import Optional
from database import ParquetDatabase
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

logger = logging.getLogger(__name__)

class QueryAnswerer:
    """Class for answering queries using T5 model"""
    
    def __init__(self, db: ParquetDatabase, model_name: str = "t5-base"):
        self.db = db
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the T5 model for question answering"""
        try:
            logger.info(f"Loading T5 model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            
            # Use GPU if available
            if torch.cuda.is_available():
                self.model = self.model.to('cuda')
                logger.info("T5 model loaded on GPU")
            else:
                logger.info("T5 model loaded on CPU")
                
        except Exception as e:
            logger.error(f"Error loading T5 model: {e}")
            # Fallback to a smaller model if the main one fails
            try:
                logger.info("Trying fallback model: t5-small")
                self.model_name = "t5-small"
                self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
                self.model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
                if torch.cuda.is_available():
                    self.model = self.model.to('cuda')
            except Exception as e2:
                logger.error(f"Error loading fallback model: {e2}")
                raise Exception("Failed to load any T5 model for query answering")
    
    def answer_query(self, query: str, user_id: Optional[str] = None) -> str:
        """
        Answer a query using T5 model
        
        Args:
            query: The query to answer
            user_id: Optional user ID for personalized responses
            
        Returns:
            Answer to the query
        """
        try:
            if not self.model or not self.tokenizer:
                logger.error("T5 model not loaded")
                return "I'm sorry, I'm unable to answer questions at the moment due to a technical issue."
            
            # Format the query for T5
            formatted_query = self._format_query_for_t5(query)
            
            # Tokenize input
            inputs = self.tokenizer(
                formatted_query,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            
            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            # Generate answer
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=256,
                    num_beams=4,
                    length_penalty=2.0,
                    early_stopping=True,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
            
            # Decode answer
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up the answer
            answer = self._clean_answer(answer)
            
            # If user_id is provided, try to personalize the response
            if user_id:
                answer = self._personalize_response(answer, user_id)
            
            logger.info(f"Query answered successfully. Query: {query[:50]}... -> Answer: {answer[:100]}...")
            return answer
            
        except Exception as e:
            logger.error(f"Error answering query: {e}")
            return "I'm sorry, I encountered an error while trying to answer your question. Please try again."
    
    def _format_query_for_t5(self, query: str) -> str:
        """
        Format the query for T5 model input
        
        Args:
            query: Original query
            
        Returns:
            Formatted query for T5
        """
        # T5 uses task prefixes. For question answering, we can use "question:"
        formatted_query = f"question: {query}"
        
        # Add context if it's a specific type of question
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['what is', 'what are', 'define', 'explain']):
            formatted_query = f"explain: {query}"
        elif any(word in query_lower for word in ['how to', 'how do', 'steps', 'process']):
            formatted_query = f"how to: {query}"
        elif any(word in query_lower for word in ['why', 'reason', 'cause']):
            formatted_query = f"why: {query}"
        elif any(word in query_lower for word in ['when', 'time', 'date']):
            formatted_query = f"when: {query}"
        elif any(word in query_lower for word in ['where', 'location', 'place']):
            formatted_query = f"where: {query}"
        elif any(word in query_lower for word in ['who', 'person', 'people']):
            formatted_query = f"who: {query}"
        
        return formatted_query
    
    def _clean_answer(self, answer: str) -> str:
        """
        Clean and format the generated answer
        
        Args:
            answer: Raw answer from model
            
        Returns:
            Cleaned answer
        """
        # Remove any task prefixes that might have been included
        prefixes_to_remove = ['answer:', 'response:', 'explain:', 'how to:', 'why:', 'when:', 'where:', 'who:']
        for prefix in prefixes_to_remove:
            if answer.lower().startswith(prefix):
                answer = answer[len(prefix):].strip()
        
        # Ensure proper sentence ending
        if not answer.endswith(('.', '!', '?')):
            answer += '.'
        
        # Remove any obvious artifacts or incomplete sentences
        sentences = answer.split('.')
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 3:  # Minimum meaningful sentence length
                cleaned_sentences.append(sentence)
        
        cleaned_answer = '. '.join(cleaned_sentences)
        if cleaned_answer:
            cleaned_answer += '.'
        
        return cleaned_answer if cleaned_answer else "I don't have enough information to answer that question."
    
    def _personalize_response(self, answer: str, user_id: str) -> str:
        """
        Personalize the response based on user's style (if available)
        
        Args:
            answer: Original answer
            user_id: User ID for personalization
            
        Returns:
            Personalized answer
        """
        try:
            # Get user data
            users = self.db.get_all_users()
            user_data = next((user for user in users if user['user_id'] == user_id), None)
            
            if not user_data:
                return answer
            
            # For now, return the original answer
            # In a more advanced implementation, you could use the ResponseRewriter here
            # to adapt the answer to the user's style
            return answer
            
        except Exception as e:
            logger.error(f"Error personalizing response for user {user_id}: {e}")
            return answer 