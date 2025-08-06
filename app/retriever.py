"""
F1 Data Retriever
Simple RAG implementation for F1 data retrieval using sentence transformers
"""

import json
import pickle
import os
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

class F1DataRetriever:
    """Simple retriever for F1 data using semantic search"""
    
    def __init__(self, data_path: str = "data", model_name: str = "all-MiniLM-L6-v2"):
        self.data_path = data_path
        self.model_name = model_name
        self.encoder = None
        self.documents = []
        self.embeddings = None
        
    def load_encoder(self):
        """Load the sentence transformer model"""
        if not self.encoder:
            logger.info(f"Loading encoder: {self.model_name}")
            self.encoder = SentenceTransformer(self.model_name)
            
    def load_data(self):
        """Load F1 data from JSON files"""
        documents = []
        
        # Load from raw data files
        raw_data_path = os.path.join(self.data_path, "raw")
        if os.path.exists(raw_data_path):
            for filename in os.listdir(raw_data_path):
                if filename.endswith(".json"):
                    file_path = os.path.join(raw_data_path, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                for item in data:
                                    if isinstance(item, dict):
                                        # Extract text content for embedding
                                        text_content = self._extract_text_content(item)
                                        if text_content:
                                            documents.append({
                                                'content': text_content,
                                                'metadata': item.get('metadata', {}),
                                                'source': filename
                                            })
                    except Exception as e:
                        logger.warning(f"Error loading {filename}: {e}")
                        continue
        
        self.documents = documents
        logger.info(f"Loaded {len(documents)} documents")
        
    def _extract_text_content(self, item: Dict) -> str:
        """Extract text content from data item"""
        content_parts = []
        
        # Try different content fields
        if 'content' in item:
            content_parts.append(str(item['content']))
        
        # Add metadata information
        metadata = item.get('metadata', {})
        if 'title' in metadata:
            content_parts.append(f"Title: {metadata['title']}")
        if 'category' in metadata:
            content_parts.append(f"Category: {metadata['category']}")
        
        return " ".join(content_parts)
    
    def create_embeddings(self):
        """Create embeddings for all documents"""
        if not self.documents:
            self.load_data()
            
        if not self.encoder:
            self.load_encoder()
            
        # Check if embeddings already exist
        embeddings_file = os.path.join(self.data_path, "embeddings", "simple_embeddings.pkl")
        documents_file = os.path.join(self.data_path, "embeddings", "documents.pkl")
        
        if os.path.exists(embeddings_file) and os.path.exists(documents_file):
            logger.info("Loading existing embeddings...")
            try:
                with open(embeddings_file, 'rb') as f:
                    self.embeddings = pickle.load(f)
                with open(documents_file, 'rb') as f:
                    self.documents = pickle.load(f)
                logger.info(f"Loaded {len(self.documents)} documents with embeddings")
                return
            except Exception as e:
                logger.warning(f"Error loading existing embeddings: {e}")
        
        # Create new embeddings
        logger.info("Creating new embeddings...")
        texts = [doc['content'] for doc in self.documents]
        self.embeddings = self.encoder.encode(texts, show_progress_bar=True)
        
        # Save embeddings
        os.makedirs(os.path.join(self.data_path, "embeddings"), exist_ok=True)
        with open(embeddings_file, 'wb') as f:
            pickle.dump(self.embeddings, f)
        with open(documents_file, 'wb') as f:
            pickle.dump(self.documents, f)
            
        logger.info("Embeddings created and saved")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """Retrieve most relevant documents for a query"""
        if not self.encoder:
            self.load_encoder()
            
        if self.embeddings is None:
            self.create_embeddings()
        
        # Encode query
        query_embedding = self.encoder.encode([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top-k most similar documents
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return relevant documents
        relevant_docs = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum similarity threshold
                relevant_docs.append(self.documents[idx]['content'])
        
        return relevant_docs[:top_k] if relevant_docs else ["No relevant F1 data found."]
    
    def get_context(self, query: str, max_context_length: int = 2000) -> str:
        """Get context for RAG by retrieving relevant documents"""
        relevant_docs = self.retrieve(query, top_k=3)
        
        # Combine documents into context
        context = "\n\n".join(relevant_docs)
        
        # Truncate if too long
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
            
        return context
