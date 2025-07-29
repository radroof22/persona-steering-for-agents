import re
from typing import List, Dict, Any, Optional
from database import ParquetDatabase
import logging
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

logger = logging.getLogger(__name__)

class ResponseRewriter:
    """Class for rewriting responses to match user's familiar text style using Hugging Face models"""
    
    def __init__(self, db: ParquetDatabase, model_name: str = "t5-base"):
        self.db = db
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the Hugging Face model for text generation"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            
            # Use GPU if available
            if torch.cuda.is_available():
                self.model = self.model.to('cuda')
                logger.info("Model loaded on GPU")
            else:
                logger.info("Model loaded on CPU")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
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
                raise Exception("Failed to load any text generation model")
    
    def extract_writing_style(self, text_corpus: str) -> Dict[str, Any]:
        """
        Extract writing style characteristics from user's text corpus
        
        Args:
            text_corpus: User's sample text corpus
            
        Returns:
            Dictionary containing style characteristics
        """
        if not text_corpus:
            return {}
        
        # Analyze sentence length
        sentences = re.split(r'[.!?]+', text_corpus)
        sentences = [s.strip() for s in sentences if s.strip()]
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        # Analyze word length
        words = re.findall(r'\b\w+\b', text_corpus.lower())
        avg_word_length = sum(len(w) for w in words) / len(words) if words else 0
        
        # Analyze vocabulary complexity (unique words ratio)
        unique_words = set(words)
        vocabulary_ratio = len(unique_words) / len(words) if words else 0
        
        # Analyze punctuation usage
        punctuation_count = len(re.findall(r'[,.!?;:]', text_corpus))
        punctuation_ratio = punctuation_count / len(words) if words else 0
        
        # Analyze paragraph structure
        paragraphs = [p.strip() for p in text_corpus.split('\n\n') if p.strip()]
        avg_paragraph_length = sum(len(p.split()) for p in paragraphs) / len(paragraphs) if paragraphs else 0
        
        # Detect formal vs informal language
        formal_indicators = ['therefore', 'furthermore', 'moreover', 'consequently', 'thus', 'hence']
        informal_indicators = ['like', 'you know', 'basically', 'actually', 'literally', 'totally']
        
        formal_count = sum(1 for word in words if word in formal_indicators)
        informal_count = sum(1 for word in words if word in informal_indicators)
        
        formality_score = (formal_count - informal_count) / len(words) if words else 0
        
        # Detect technical vs simple language
        technical_indicators = ['algorithm', 'implementation', 'optimization', 'framework', 'architecture']
        technical_count = sum(1 for word in words if word in technical_indicators)
        technical_ratio = technical_count / len(words) if words else 0
        
        return {
            'avg_sentence_length': avg_sentence_length,
            'avg_word_length': avg_word_length,
            'vocabulary_ratio': vocabulary_ratio,
            'punctuation_ratio': punctuation_ratio,
            'avg_paragraph_length': avg_paragraph_length,
            'formality_score': formality_score,
            'technical_ratio': technical_ratio,
            'total_words': len(words),
            'total_sentences': len(sentences),
            'total_paragraphs': len(paragraphs)
        }
    
    def rewrite_response(self, original_response: str, user_id: str) -> str:
        """
        Rewrite a response to match the user's familiar writing style
        
        Args:
            original_response: The original response to rewrite
            user_id: ID of the user whose style to match
            
        Returns:
            Rewritten response matching user's style
        """
        try:
            # Get user data
            users = self.db.get_all_users()
            user_data = next((user for user in users if user['user_id'] == user_id), None)
            
            if not user_data:
                logger.warning(f"User {user_id} not found, returning original response")
                return original_response
            
            sample_text_corpus = user_data.get('sample_text_corpus')
            if not sample_text_corpus:
                logger.warning(f"No sample text corpus for user {user_id}, returning original response")
                return original_response
            
            # Extract user's writing style
            user_style = self.extract_writing_style(sample_text_corpus)
            if not user_style:
                return original_response
            
            # Rewrite the response based on user's style using Hugging Face model
            rewritten_response = self._rewrite_with_model(original_response, user_style, sample_text_corpus)
            
            logger.info(f"Successfully rewrote response for user {user_id}")
            return rewritten_response
            
        except Exception as e:
            logger.error(f"Error rewriting response for user {user_id}: {e}")
            return original_response
    
    def _rewrite_with_model(self, text: str, user_style: Dict[str, Any], sample_text: str) -> str:
        """
        Rewrite text using Hugging Face model based on user's writing style
        
        Args:
            text: Original text to transform
            user_style: User's writing style characteristics
            sample_text: User's sample text corpus for context
            
        Returns:
            Transformed text using the model
        """
        try:
            if not self.model or not self.tokenizer:
                logger.error("Model not loaded, returning original text")
                return text
            
            # Create a style prompt based on user characteristics
            style_prompt = self._create_style_prompt(user_style, sample_text)
            
            # Create the input for the model
            # For T5, we use a prefix task format
            input_text = f"rewrite in style: {text}"
            
            # Add style context to the input
            if style_prompt:
                input_text = f"style: {style_prompt} rewrite: {text}"
            
            # Tokenize input
            inputs = self.tokenizer(
                input_text,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            
            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            # Generate output
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
            
            # Decode output
            rewritten_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up the output
            rewritten_text = self._clean_generated_text(rewritten_text, text)
            
            logger.info(f"Model rewrite completed. Original: {text[:50]}... -> Rewritten: {rewritten_text[:150]}...")
            return rewritten_text
            
        except Exception as e:
            logger.error(f"Error in model-based rewriting: {e}")
            return text
    
    def _create_style_prompt(self, user_style: Dict[str, Any], sample_text: str) -> str:
        """
        Create a style prompt based on user's writing characteristics
        
        Args:
            user_style: User's writing style characteristics
            sample_text: User's sample text corpus
            
        Returns:
            Style prompt string
        """
        style_descriptions = []
        
        # Add formality level
        formality_score = user_style.get('formality_score', 0)
        if formality_score > 0.01:
            style_descriptions.append("formal academic tone")
        elif formality_score < -0.01:
            style_descriptions.append("casual conversational tone")
        else:
            style_descriptions.append("neutral professional tone")
        
        # Add vocabulary level
        vocabulary_ratio = user_style.get('vocabulary_ratio', 0.7)
        if vocabulary_ratio < 0.6:
            style_descriptions.append("simple vocabulary")
        elif vocabulary_ratio > 0.8:
            style_descriptions.append("advanced vocabulary")
        
        # Add sentence complexity
        avg_sentence_length = user_style.get('avg_sentence_length', 15)
        if avg_sentence_length < 12:
            style_descriptions.append("short sentences")
        elif avg_sentence_length > 20:
            style_descriptions.append("complex sentences")
        
        # Add technical level
        technical_ratio = user_style.get('technical_ratio', 0)
        if technical_ratio > 0.05:
            style_descriptions.append("technical terminology")
        
        # Extract key phrases from sample text (first 100 words)
        sample_words = sample_text.split()[:100]
        sample_text_sample = ' '.join(sample_words)
        
        # Create the final prompt
        style_prompt = f"Write in a style that is {', '.join(style_descriptions)}. "
        style_prompt += f"Use similar language patterns as: {sample_text_sample[:200]}..."
        
        return style_prompt
    
    def _clean_generated_text(self, generated_text: str, original_text: str) -> str:
        """
        Clean and validate generated text
        
        Args:
            generated_text: Text generated by the model
            original_text: Original input text
            
        Returns:
            Cleaned text
        """
        # Remove any prefixes that might have been added
        if generated_text.startswith("rewrite:"):
            generated_text = generated_text[8:].strip()
        
        # Remove any style prefixes
        if generated_text.startswith("style:"):
            # Find the actual content after style description
            parts = generated_text.split("rewrite:", 1)
            if len(parts) > 1:
                generated_text = parts[1].strip()
        
        # Ensure the text is not empty
        if not generated_text.strip():
            return original_text
        
        # Ensure proper sentence ending
        if not generated_text.endswith(('.', '!', '?')):
            generated_text += '.'
        
        # Remove any duplicate sentences or obvious artifacts
        sentences = generated_text.split('.')
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 3:  # Minimum meaningful sentence length
                cleaned_sentences.append(sentence)
        
        cleaned_text = '. '.join(cleaned_sentences)
        if cleaned_text:
            cleaned_text += '.'
        
        return cleaned_text if cleaned_text else original_text
    

    
    def get_user_style_analysis(self, user_id: str) -> Dict[str, Any]:
        """
        Get detailed analysis of user's writing style
        
        Args:
            user_id: ID of the user to analyze
            
        Returns:
            Dictionary with style analysis
        """
        try:
            users = self.db.get_all_users()
            user_data = next((user for user in users if user['user_id'] == user_id), None)
            
            if not user_data:
                return {"error": "User not found"}
            
            sample_text_corpus = user_data.get('sample_text_corpus')
            if not sample_text_corpus:
                return {"error": "No sample text corpus available"}
            
            style_analysis = self.extract_writing_style(sample_text_corpus)
            
            # Add interpretation
            interpretation = {
                'sentence_complexity': 'Simple' if style_analysis['avg_sentence_length'] < 15 else 'Complex',
                'vocabulary_level': 'Basic' if style_analysis['vocabulary_ratio'] < 0.6 else 'Advanced',
                'formality_level': 'Formal' if style_analysis['formality_score'] > 0 else 'Informal',
                'technical_level': 'Technical' if style_analysis['technical_ratio'] > 0.05 else 'Non-technical'
            }
            
            return {
                'user_id': user_id,
                'style_characteristics': style_analysis,
                'interpretation': interpretation,
                'sample_text_length': len(sample_text_corpus)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing user style for {user_id}: {e}")
            return {"error": str(e)} 