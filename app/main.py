"""
F1 RAG Chatbot Main Application
Simple RAG chatbot for F1 queries using Llama-3.2-1B and retrieval
"""

import logging
import os
from typing import Optional
from .model import F1ChatbotModel
from .retriever import F1DataRetriever

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class F1RagChatbot:
    """Main F1 RAG Chatbot class"""
    
    def __init__(self, data_path: str = "data"):
        self.data_path = data_path
        self.model = F1ChatbotModel()
        self.retriever = F1DataRetriever(data_path=data_path)
        self._initialized = False
        
    def initialize(self):
        """Initialize the chatbot components"""
        if self._initialized:
            return
            
        logger.info("Initializing F1 RAG Chatbot...")
        
        # Initialize retriever (this will load data and create embeddings)
        self.retriever.create_embeddings()
        
        # Load model
        self.model.load_model()
        
        self._initialized = True
        logger.info("F1 RAG Chatbot initialized successfully!")
    
    def chat(self, question: str) -> str:
        """Main chat function"""
        if not self._initialized:
            self.initialize()
        
        try:
            # Get relevant context from retriever
            logger.info(f"Processing question: {question[:50]}...")
            context = self.retriever.get_context(question)
            
            # Generate response using the model
            response = self.model.generate_response(
                context=context,
                question=question,
                max_length=300
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return "I'm sorry, I encountered an error while processing your question. Please try again."
    
    def get_system_info(self) -> dict:
        """Get system information"""
        return {
            "model_device": self.model.device,
            "documents_loaded": len(self.retriever.documents) if self.retriever.documents else 0,
            "initialized": self._initialized
        }

def create_chatbot(data_path: str = "data") -> F1RagChatbot:
    """Factory function to create chatbot instance"""
    return F1RagChatbot(data_path=data_path)

# Quick test function
def quick_test():
    """Quick test of the chatbot"""
    chatbot = create_chatbot()
    
    test_questions = [
        "Who got pole position in Hungary?",
        "Tell me about Leclerc's performance",
        "What happened in the latest F1 race?"
    ]
    
    print("=== F1 RAG Chatbot Quick Test ===")
    print(f"System Info: {chatbot.get_system_info()}")
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        try:
            answer = chatbot.chat(question)
            print(f"Answer: {answer}")
        except Exception as e:
            print(f"Error: {e}")
        print("-" * 50)

if __name__ == "__main__":
    quick_test()
