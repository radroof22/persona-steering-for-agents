import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMProvider:
    """Provides access to Hugging Face LLM for inference."""
    
    def __init__(self, model_name: str = "gpt2"):
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
    
    def generate(self, prompt: str, max_new_tokens: int = 50, temperature: float = 0.8) -> str:
        """
        Generate a response using the LLM based on the given prompt.
        
        Args:
            prompt: The prompt to send to the model
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature for generation
            
        Returns:
            The generated response
        """
        self._load_model()
        
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            # Move to device
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (after the prompt)
            generated_text = full_response[len(prompt):].strip()
            
            # Clean up the response
            if generated_text:
                # Remove any trailing punctuation or incomplete sentences
                lines = generated_text.split('\n')
                first_line = lines[0].strip()
                
                # Remove quotes if present
                if first_line.startswith('"') and first_line.endswith('"'):
                    first_line = first_line[1:-1]
                
                return first_line
            else:
                # Fallback to a simple response if no generation
                return "Unable to generate response"
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            # Return fallback response
            return "Error generating response"


def create_llm_provider(model_name: Optional[str] = None) -> LLMProvider:
    """
    Factory function to create an LLMProvider instance.
    
    Args:
        model_name: Optional model name override
        
    Returns:
        LLMProvider instance
    """
    if model_name is None:
        model_name = "gpt2"
    
    return LLMProvider(model_name=model_name) 