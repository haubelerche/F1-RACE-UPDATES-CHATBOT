"""
Main F1 RAG Chatbot application
"""
from typing import Optional
from loguru import logger

from .model import get_mistral_model
from .retriever import get_retriever
from .config import APP_TITLE


class F1Chatbot:
    """
    Main F1 RAG Chatbot class
    """
    
    def __init__(self):
        self.model = None
        self.retriever = None
        self._initialize()
    
    def _initialize(self):
        """Initialize model and retriever"""
        try:
            logger.info("Initializing F1 Chatbot...")
            
            # Initialize retriever first (faster)
            self.retriever = get_retriever()
            
            # Initialize model (slower)
            self.model = get_mistral_model()
            
            logger.info("F1 Chatbot initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing chatbot: {e}")
            raise
    
    def answer_question(self, question: str) -> str:
        """
        Answer a question about F1 using RAG
        
        Args:
            question: User question
            
        Returns:
            Generated answer
        """
        try:
            logger.info(f"Processing question: {question[:100]}...")
            
            # Step 1: Retrieve relevant context
            context = self.retriever.get_context(question)
            
            # Step 2: Generate response with context
            answer = self.model.generate_response(question, context)
            
            logger.info("Question answered successfully")
            return answer
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return "I apologize, but I encountered an error while processing your question. Please try again."
    
    def get_system_info(self) -> dict:
        """Get system information"""
        try:
            retriever_info = self.retriever.get_collection_info()
            return {
                "title": APP_TITLE,
                "model": "Llama 3.1 8B-Instruct",
                "retriever": retriever_info,
                "status": "ready"
            }
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return {"status": "error"}
    
    def update_knowledge_base(self, documents: list):
        """Update the knowledge base with new documents"""
        try:
            self.retriever.add_documents(documents)
            logger.info(f"Knowledge base updated with {len(documents)} documents")
            return True
        except Exception as e:
            logger.error(f"Error updating knowledge base: {e}")
            return False


# Global chatbot instance
_chatbot_instance = None


def get_chatbot() -> F1Chatbot:
    """Get or create global chatbot instance"""
    global _chatbot_instance
    
    if _chatbot_instance is None:
        _chatbot_instance = F1Chatbot()
    
    return _chatbot_instance


def main():
    """Main function for testing"""
    chatbot = get_chatbot()
    
    # Test questions
    test_questions = [
        "Who won the last race?",
        "What's the current championship standings?",
        "Tell me about Mercedes F1 team",
        "What happened in the Monaco Grand Prix?"
    ]
    
    for question in test_questions:
        print(f"\nQ: {question}")
        answer = chatbot.answer_question(question)
        print(f"A: {answer}")


if __name__ == "__main__":
    main()