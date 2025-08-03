"""
Llama 3.1 8B model wrapper for F1 chatbot
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from loguru import logger
from typing import Optional
import gc

from .config import MODEL_NAME, MAX_RESPONSE_LENGTH, TEMPERATURE


class LlamaChatbot:
    """
    Wrapper for Llama 3.1 8B-Instruct model
    """
    
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        """Load the Llama model and tokenizer"""
        try:
            logger.info(f"Loading Llama model on {self.device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with appropriate precision
            if self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                )
                self.model.to(self.device)
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            
            logger.info("Llama model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading Llama model: {e}")
            raise
    
    def generate_response(self, prompt: str, context: str = "") -> str:
        """
        Generate response using Llama with F1 context
        
        Args:
            prompt: User question
            context: Retrieved F1 context
            
        Returns:
            Generated response
        """
        try:
            # Format the prompt for Llama 3.1 Instruct
            formatted_prompt = self._format_prompt(prompt, context)
            
            # Generate response
            response = self.pipeline(
                formatted_prompt,
                max_length=len(self.tokenizer.encode(formatted_prompt)) + MAX_RESPONSE_LENGTH,
                temperature=TEMPERATURE,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                truncation=True,
                return_full_text=False  # Only return the generated part
            )
            
            # Extract generated text
            if response and len(response) > 0:
                generated_text = response[0]["generated_text"]
                return generated_text.strip()
            else:
                return "I apologize, but I couldn't generate a response. Please try again."
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error while generating a response. Please try again."
    
    def _format_prompt(self, question: str, context: str) -> str:
        """
        Format prompt for Llama 3.1 Instruct model
        
        Args:
            question: User question
            context: F1 context information
            
        Returns:
            Formatted prompt
        """
        system_message = """You are a knowledgeable Formula 1 expert assistant. Use the provided context to answer questions about F1 races, drivers, teams, and news. Be accurate, informative, and engaging. If the context doesn't contain enough information to answer the question, say so politely."""
        
        if context.strip():
            user_message = f"""Context information:
{context}

Question: {question}

Please provide a helpful and accurate answer based on the context above."""
        else:
            user_message = f"""Question: {question}

Please provide a helpful answer about Formula 1."""
        
        # Llama 3.1 Instruct format with proper chat template
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        # Use the tokenizer's chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                return formatted_prompt
            except Exception as e:
                logger.warning(f"Chat template failed, using fallback: {e}")
        
        # Fallback format for Llama 3.1
        formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        return formatted_prompt
    
    def cleanup(self):
        """Clean up model resources"""
        try:
            if self.model:
                del self.model
            if self.tokenizer:
                del self.tokenizer
            if self.pipeline:
                del self.pipeline
                
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            gc.collect()
            logger.info("Model cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Global model instance
_llama_instance = None


def get_llama_model() -> LlamaChatbot:
    """Get or create global Llama model instance"""
    global _llama_instance
    
    if _llama_instance is None:
        _llama_instance = LlamaChatbot()
    
    return _llama_instance


# Backward compatibility alias
def get_mistral_model() -> LlamaChatbot:
    """Backward compatibility - returns Llama model"""
    return get_llama_model()