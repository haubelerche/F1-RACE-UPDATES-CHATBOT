"""
ChromaDB-based retriever for F1 knowledge base
"""
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from loguru import logger
import pandas as pd
import json

from .config import (
    CHROMA_PERSIST_DIR, 
    COLLECTION_NAME, 
    EMBEDDING_MODEL,
    MAX_CONTEXT_LENGTH
)


class F1Retriever:
    """
    Retriever for F1 knowledge base using ChromaDB
    """
    
    def __init__(self):
        self.client = None
        self.collection = None
        self.embedding_model = None
        self._initialize()
    
    def _initialize(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=CHROMA_PERSIST_DIR,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Initialize embedding model
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(name=COLLECTION_NAME)
                logger.info(f"Loaded existing collection: {COLLECTION_NAME}")
            except:
                self.collection = self.client.create_collection(
                    name=COLLECTION_NAME,
                    metadata={"description": "F1 race data and news"}
                )
                logger.info(f"Created new collection: {COLLECTION_NAME}")
                
        except Exception as e:
            logger.error(f"Error initializing retriever: {e}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add documents to the knowledge base
        
        Args:
            documents: List of document dictionaries with 'content', 'metadata', etc.
        """
        try:
            texts = []
            metadatas = []
            ids = []
            
            for i, doc in enumerate(documents):
                content = doc.get('content', '')
                metadata = doc.get('metadata', {})
                doc_id = doc.get('id', f"doc_{i}")
                
                if content.strip():
                    texts.append(content)
                    metadatas.append(metadata)
                    ids.append(doc_id)
            
            if texts:
                # Generate embeddings
                embeddings = self.embedding_model.encode(texts).tolist()
                
                # Add to collection
                self.collection.add(
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
                
                logger.info(f"Added {len(texts)} documents to knowledge base")
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            
        Returns:
            List of relevant documents with metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()
            
            # Search collection
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            retrieved_docs = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    retrieved_docs.append({
                        'content': doc,
                        'metadata': results['metadatas'][0][i] if results['metadatas'][0] else {},
                        'distance': results['distances'][0][i] if results['distances'][0] else 1.0
                    })
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents for query: {query[:50]}...")
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def get_context(self, query: str, max_length: int = MAX_CONTEXT_LENGTH) -> str:
        """
        Get formatted context string for the query
        
        Args:
            query: User query
            max_length: Maximum context length
            
        Returns:
            Formatted context string
        """
        try:
            # Retrieve relevant documents
            docs = self.retrieve(query)
            
            # Format context
            context_parts = []
            current_length = 0
            
            for doc in docs:
                content = doc['content']
                metadata = doc['metadata']
                
                # Format document
                if metadata.get('source'):
                    doc_text = f"Source: {metadata['source']}\n{content}\n"
                else:
                    doc_text = f"{content}\n"
                
                # Check length limit
                if current_length + len(doc_text) > max_length:
                    break
                
                context_parts.append(doc_text)
                current_length += len(doc_text)
            
            return "\n---\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error getting context: {e}")
            return ""
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            count = self.collection.count()
            return {
                "name": COLLECTION_NAME,
                "document_count": count,
                "embedding_model": EMBEDDING_MODEL
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}
    
    def reset_collection(self):
        """Reset the collection (delete all documents)"""
        try:
            self.client.delete_collection(name=COLLECTION_NAME)
            self.collection = self.client.create_collection(
                name=COLLECTION_NAME,
                metadata={"description": "F1 race data and news"}
            )
            logger.info("Collection reset successfully")
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")


# Global retriever instance
_retriever_instance = None


def get_retriever() -> F1Retriever:
    """Get or create global retriever instance"""
    global _retriever_instance
    
    if _retriever_instance is None:
        _retriever_instance = F1Retriever()
    
    return _retriever_instance