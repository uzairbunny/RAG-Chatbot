import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import PyPDF2
from docx import Document as DocxDocument
from bs4 import BeautifulSoup
import requests
from sentence_transformers import SentenceTransformer
import tiktoken


class DocumentProcessor:
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def extract_text_from_file(self, file_path: str, file_type: Optional[str] = None) -> str:
        """Extract text from various file types."""
        if not file_type:
            file_type = Path(file_path).suffix.lower()
        
        try:
            if file_type in ['.pdf']:
                return self._extract_from_pdf(file_path)
            elif file_type in ['.docx', '.doc']:
                return self._extract_from_docx(file_path)
            elif file_type in ['.txt', '.md']:
                return self._extract_from_text(file_path)
            elif file_type in ['.html', '.htm']:
                return self._extract_from_html(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        except Exception as e:
            raise Exception(f"Error extracting text from {file_path}: {str(e)}")
    
    def _extract_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF files."""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    
    def _extract_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX files."""
        doc = DocxDocument(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    
    def _extract_from_text(self, file_path: str) -> str:
        """Extract text from plain text files."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    
    def _extract_from_html(self, file_path: str) -> str:
        """Extract text from HTML files."""
        with open(file_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file.read(), 'html.parser')
            return soup.get_text().strip()
    
    def chunk_text(self, text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks."""
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        start = 0
        chunk_id = 0
        
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            chunks.append({
                'id': chunk_id,
                'text': chunk_text.strip(),
                'start_token': start,
                'end_token': end,
                'token_count': len(chunk_tokens)
            })
            
            chunk_id += 1
            start = end - chunk_overlap
            
            if start >= len(tokens):
                break
        
        return chunks
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        return self.embedding_model.encode(texts).tolist()
    
    def process_document(self, file_path: str, chunk_size: int = 500, chunk_overlap: int = 50) -> Dict[str, Any]:
        """Process a document: extract text, chunk it, and generate embeddings."""
        file_type = Path(file_path).suffix.lower()
        filename = Path(file_path).name
        
        # Extract text
        text = self.extract_text_from_file(file_path, file_type)
        
        # Create chunks
        chunks = self.chunk_text(text, chunk_size, chunk_overlap)
        
        # Generate embeddings
        chunk_texts = [chunk['text'] for chunk in chunks]
        embeddings = self.generate_embeddings(chunk_texts)
        
        # Add embeddings to chunks
        for i, chunk in enumerate(chunks):
            chunk['embedding'] = embeddings[i]
        
        return {
            'filename': filename,
            'file_type': file_type,
            'content': text,
            'chunks': chunks,
            'metadata': {
                'total_tokens': len(self.tokenizer.encode(text)),
                'total_chunks': len(chunks),
                'chunk_size': chunk_size,
                'chunk_overlap': chunk_overlap
            }
        }
    
    def process_url(self, url: str, chunk_size: int = 500, chunk_overlap: int = 50) -> Dict[str, Any]:
        """Process content from a URL."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text().strip()
            
            # Create chunks
            chunks = self.chunk_text(text, chunk_size, chunk_overlap)
            
            # Generate embeddings
            chunk_texts = [chunk['text'] for chunk in chunks]
            embeddings = self.generate_embeddings(chunk_texts)
            
            # Add embeddings to chunks
            for i, chunk in enumerate(chunks):
                chunk['embedding'] = embeddings[i]
            
            return {
                'filename': f"web_content_{url.replace('/', '_').replace(':', '')}",
                'file_type': 'html',
                'content': text,
                'chunks': chunks,
                'metadata': {
                    'url': url,
                    'total_tokens': len(self.tokenizer.encode(text)),
                    'total_chunks': len(chunks),
                    'chunk_size': chunk_size,
                    'chunk_overlap': chunk_overlap
                }
            }
        except Exception as e:
            raise Exception(f"Error processing URL {url}: {str(e)}")
