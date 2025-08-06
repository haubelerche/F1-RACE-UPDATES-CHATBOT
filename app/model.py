"""
F1 RAG Chatbot Model
Simple implementation using Llama-3.2-1B for F1 race queries
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class F1ChatbotModel:
    """Simple F1 chatbot model using Llama-3.2-1B"""
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        
    def load_model(self):
        """Load the Llama model and tokenizer"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Add pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
                
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def generate_response(self, context: str, question: str, max_length: int = 512) -> str:
        """Generate response based on context and question"""
        if not self.model or not self.tokenizer:
            self.load_model()
        
        # Create a focused prompt for F1 queries
        prompt = f"""You are an F1 expert assistant. Use the provided context to answer questions about Formula 1 racing.

Context: {context}

Question: {question}

Answer: """
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding=True
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the answer part
            if "Answer: " in response:
                answer = response.split("Answer: ")[-1].strip()
            else:
                answer = response.strip()
                
            return answer
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm sorry, I encountered an error while processing your question."