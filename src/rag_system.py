import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import openai
from sentence_transformers import SentenceTransformer
from .vector_store import VectorStore, create_vector_store
from .document_processor import DocumentProcessor
from config.settings import settings


@dataclass
class RAGResponse:
    """Response from the RAG system."""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    metadata: Dict[str, Any]


class RAGSystem:
    """Retrieval-Augmented Generation system."""
    
    def __init__(self, vector_store: VectorStore = None):
        # Initialize OpenAI client
        openai.api_key = settings.openai_api_key
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        
        # Initialize document processor
        self.document_processor = DocumentProcessor(settings.embedding_model)
        
        # Initialize vector store
        if vector_store is None:
            vector_store_kwargs = {}
            if settings.vector_db_type.lower() == "pinecone":
                vector_store_kwargs = {
                    "api_key": settings.pinecone_api_key,
                    "environment": settings.pinecone_environment
                }
            self.vector_store = create_vector_store(settings.vector_db_type, **vector_store_kwargs)
        else:
            self.vector_store = vector_store
    
    def add_document(self, file_path: str) -> Dict[str, Any]:
        """Add a document to the knowledge base."""
        try:
            # Process document
            processed_doc = self.document_processor.process_document(file_path)
            
            # Add to vector store
            doc_ids = self.vector_store.add_documents([processed_doc])
            
            return {
                "status": "success",
                "filename": processed_doc['filename'],
                "chunks_added": len(doc_ids),
                "doc_ids": doc_ids,
                "metadata": processed_doc['metadata']
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def add_document_from_url(self, url: str) -> Dict[str, Any]:
        """Add content from a URL to the knowledge base."""
        try:
            # Process URL content
            processed_doc = self.document_processor.process_url(url)
            
            # Add to vector store
            doc_ids = self.vector_store.add_documents([processed_doc])
            
            return {
                "status": "success",
                "url": url,
                "chunks_added": len(doc_ids),
                "doc_ids": doc_ids,
                "metadata": processed_doc['metadata']
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def query(self, question: str, conversation_history: List[Dict[str, str]] = None, top_k: int = None) -> RAGResponse:
        """Query the RAG system."""
        if top_k is None:
            top_k = settings.top_k_results
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(question).tolist()
            
            # Search for relevant documents
            search_results = self.vector_store.search(query_embedding, top_k)
            
            if not search_results:
                return RAGResponse(
                    answer="I don't have any relevant information to answer your question.",
                    sources=[],
                    confidence=0.0,
                    metadata={"error": "No relevant documents found"}
                )
            
            # Build context from search results
            context = self._build_context(search_results)
            
            # Generate response using LLM
            response = self._generate_response(question, context, conversation_history)
            
            # Calculate confidence based on search scores
            confidence = self._calculate_confidence(search_results)
            
            return RAGResponse(
                answer=response,
                sources=search_results,
                confidence=confidence,
                metadata={
                    "context_length": len(context),
                    "sources_count": len(search_results)
                }
            )
        
        except Exception as e:
            return RAGResponse(
                answer=f"I encountered an error while processing your question: {str(e)}",
                sources=[],
                confidence=0.0,
                metadata={"error": str(e)}
            )
    
    def _build_context(self, search_results: List[Dict[str, Any]]) -> str:
        """Build context string from search results."""
        context_parts = []
        for i, result in enumerate(search_results):
            context_parts.append(f"[Source {i+1}] {result['filename']}:\n{result['text']}")
        
        return "\n\n".join(context_parts)
    
    def _generate_response(self, question: str, context: str, conversation_history: List[Dict[str, str]] = None) -> str:
        """Generate response using OpenAI GPT."""
        
        # Build conversation messages
        messages = [
            {
                "role": "system",
                "content": """You are a helpful AI assistant that answers questions based on the provided context. 
                Follow these guidelines:
                1. Answer questions using only the information provided in the context
                2. Be accurate and concise
                3. If the context doesn't contain enough information, say so clearly
                4. Cite relevant sources when appropriate
                5. Maintain a professional and helpful tone
                """
            }
        ]
        
        # Add conversation history if provided
        if conversation_history:
            for msg in conversation_history[-5:]:  # Last 5 messages for context
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Add current question with context
        user_message = f"Context:\n{context}\n\nQuestion: {question}"
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        # Generate response
        response = openai.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    def _calculate_confidence(self, search_results: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on search results."""
        if not search_results:
            return 0.0
        
        # Average of top search scores
        scores = [result['score'] for result in search_results]
        return sum(scores) / len(scores)
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        return {
            "total_documents": self.vector_store.get_document_count(),
            "vector_db_type": settings.vector_db_type,
            "embedding_model": settings.embedding_model,
            "llm_model": settings.llm_model
        }
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document from the knowledge base."""
        return self.vector_store.delete_document(document_id)


class ConversationManager:
    """Manages conversation context and history."""
    
    def __init__(self, max_history_length: int = 10):
        self.conversations = {}
        self.max_history_length = max_history_length
    
    def add_message(self, conversation_id: str, role: str, content: str, metadata: Dict[str, Any] = None):
        """Add a message to the conversation history."""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        
        message = {
            "role": role,
            "content": content,
            "metadata": metadata or {}
        }
        
        self.conversations[conversation_id].append(message)
        
        # Keep only recent messages
        if len(self.conversations[conversation_id]) > self.max_history_length:
            self.conversations[conversation_id] = self.conversations[conversation_id][-self.max_history_length:]
    
    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, str]]:
        """Get conversation history for a given conversation ID."""
        return self.conversations.get(conversation_id, [])
    
    def clear_conversation(self, conversation_id: str):
        """Clear conversation history for a given conversation ID."""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
    
    def get_all_conversations(self) -> List[str]:
        """Get all conversation IDs."""
        return list(self.conversations.keys())
