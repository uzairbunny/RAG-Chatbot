import os
import json
import pickle
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss
import chromadb
from sentence_transformers import SentenceTransformer

try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False


class VectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Add documents to the vector store."""
        pass
    
    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    def delete_document(self, document_id: str) -> bool:
        """Delete a document from the vector store."""
        pass
    
    @abstractmethod
    def get_document_count(self) -> int:
        """Get the total number of documents in the store."""
        pass


class FAISSVectorStore(VectorStore):
    """FAISS-based vector store for local deployment."""
    
    def __init__(self, dimension: int = 384, index_file: str = "faiss_index.bin"):
        self.dimension = dimension
        self.index_file = index_file
        self.metadata_file = f"{index_file}.metadata"
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.documents = {}
        self.id_counter = 0
        
        # Load existing index if available
        self.load_index()
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Add documents to FAISS index."""
        embeddings = []
        doc_ids = []
        
        for doc in documents:
            for chunk in doc['chunks']:
                doc_id = f"{doc['filename']}_{chunk['id']}_{self.id_counter}"
                self.id_counter += 1
                
                # Normalize embedding for cosine similarity
                embedding = np.array(chunk['embedding'], dtype=np.float32)
                embedding = embedding / np.linalg.norm(embedding)
                
                embeddings.append(embedding)
                doc_ids.append(doc_id)
                
                self.documents[doc_id] = {
                    'text': chunk['text'],
                    'filename': doc['filename'],
                    'file_type': doc['file_type'],
                    'chunk_id': chunk['id'],
                    'metadata': doc.get('metadata', {})
                }
        
        # Add to FAISS index
        embeddings_array = np.vstack(embeddings)
        self.index.add(embeddings_array)
        
        # Save index
        self.save_index()
        
        return doc_ids
    
    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar documents using FAISS."""
        if self.index.ntotal == 0:
            return []
        
        # Normalize query embedding
        query_embedding = np.array(query_embedding, dtype=np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        results = []
        doc_ids = list(self.documents.keys())
        
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # Valid result
                doc_id = doc_ids[idx]
                doc = self.documents[doc_id]
                results.append({
                    'id': doc_id,
                    'text': doc['text'],
                    'filename': doc['filename'],
                    'file_type': doc['file_type'],
                    'score': float(scores[0][i]),
                    'metadata': doc['metadata']
                })
        
        return results
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document (not efficiently supported in FAISS)."""
        if document_id in self.documents:
            del self.documents[document_id]
            self.save_index()
            return True
        return False
    
    def get_document_count(self) -> int:
        """Get the total number of documents."""
        return len(self.documents)
    
    def save_index(self):
        """Save FAISS index and metadata to disk."""
        faiss.write_index(self.index, self.index_file)
        with open(self.metadata_file, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'id_counter': self.id_counter
            }, f)
    
    def load_index(self):
        """Load FAISS index and metadata from disk."""
        if os.path.exists(self.index_file):
            self.index = faiss.read_index(self.index_file)
        
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'rb') as f:
                data = pickle.load(f)
                self.documents = data.get('documents', {})
                self.id_counter = data.get('id_counter', 0)


class ChromaDBVectorStore(VectorStore):
    """ChromaDB-based vector store."""
    
    def __init__(self, collection_name: str = "documents", persist_directory: str = "./chroma_db"):
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Add documents to ChromaDB."""
        doc_ids = []
        embeddings = []
        texts = []
        metadatas = []
        
        for doc in documents:
            for chunk in doc['chunks']:
                doc_id = f"{doc['filename']}_{chunk['id']}"
                doc_ids.append(doc_id)
                embeddings.append(chunk['embedding'])
                texts.append(chunk['text'])
                metadatas.append({
                    'filename': doc['filename'],
                    'file_type': doc['file_type'],
                    'chunk_id': chunk['id'],
                    **doc.get('metadata', {})
                })
        
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=doc_ids
        )
        
        return doc_ids
    
    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar documents using ChromaDB."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        search_results = []
        if results['documents']:
            for i in range(len(results['documents'][0])):
                search_results.append({
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'filename': results['metadatas'][0][i]['filename'],
                    'file_type': results['metadatas'][0][i]['file_type'],
                    'score': 1 - results['distances'][0][i],  # Convert distance to similarity
                    'metadata': results['metadatas'][0][i]
                })
        
        return search_results
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document from ChromaDB."""
        try:
            self.collection.delete(ids=[document_id])
            return True
        except Exception:
            return False
    
    def get_document_count(self) -> int:
        """Get the total number of documents."""
        return self.collection.count()


class PineconeVectorStore(VectorStore):
    """Pinecone-based vector store."""
    
    def __init__(self, api_key: str, environment: str, index_name: str = "documents"):
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone is not installed. Install with: pip install pinecone-client")
        
        self.index_name = index_name
        pinecone.init(api_key=api_key, environment=environment)
        
        # Create index if it doesn't exist
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                dimension=384,  # Default for sentence-transformers/all-MiniLM-L6-v2
                metric="cosine"
            )
        
        self.index = pinecone.Index(index_name)
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Add documents to Pinecone."""
        vectors = []
        doc_ids = []
        
        for doc in documents:
            for chunk in doc['chunks']:
                doc_id = f"{doc['filename']}_{chunk['id']}"
                doc_ids.append(doc_id)
                
                vectors.append({
                    'id': doc_id,
                    'values': chunk['embedding'],
                    'metadata': {
                        'text': chunk['text'],
                        'filename': doc['filename'],
                        'file_type': doc['file_type'],
                        'chunk_id': chunk['id'],
                        **doc.get('metadata', {})
                    }
                })
        
        # Upsert in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            self.index.upsert(vectors=batch)
        
        return doc_ids
    
    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar documents using Pinecone."""
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        search_results = []
        for match in results.matches:
            search_results.append({
                'id': match.id,
                'text': match.metadata['text'],
                'filename': match.metadata['filename'],
                'file_type': match.metadata['file_type'],
                'score': match.score,
                'metadata': match.metadata
            })
        
        return search_results
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document from Pinecone."""
        try:
            self.index.delete(ids=[document_id])
            return True
        except Exception:
            return False
    
    def get_document_count(self) -> int:
        """Get the total number of documents."""
        stats = self.index.describe_index_stats()
        return stats.total_vector_count


def create_vector_store(vector_db_type: str, **kwargs) -> VectorStore:
    """Factory function to create vector store instances."""
    if vector_db_type.lower() == "faiss":
        return FAISSVectorStore(**kwargs)
    elif vector_db_type.lower() == "chromadb":
        return ChromaDBVectorStore(**kwargs)
    elif vector_db_type.lower() == "pinecone":
        return PineconeVectorStore(**kwargs)
    else:
        raise ValueError(f"Unsupported vector database type: {vector_db_type}")
