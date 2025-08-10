#!/usr/bin/env python3
"""
Quick start script for the RAG Chatbot
This script helps you get started quickly with minimal configuration.
"""

import os
import sys
from pathlib import Path

def main():
    print("""
    🤖 RAG CHATBOT - QUICK START
    ===========================
    
    This script will help you get the RAG chatbot up and running quickly!
    """)
    
    # Check if .env exists
    env_path = Path(".env")
    if not env_path.exists():
        print("📝 Setting up environment configuration...")
        
        # Get OpenAI API key from user
        api_key = input("\n🔑 Please enter your OpenAI API key (required): ").strip()
        
        if not api_key:
            print("❌ OpenAI API key is required to run the chatbot.")
            sys.exit(1)
        
        # Create .env file
        with open(env_path, 'w') as f:
            f.write(f"OPENAI_API_KEY={api_key}\n")
            f.write("VECTOR_DB_TYPE=faiss\n")
            f.write("HOST=127.0.0.1\n")
            f.write("PORT=8000\n")
        
        print("✅ Environment configuration created!")
    else:
        print("✅ Environment configuration found!")
    
    # Check if requirements are installed
    print("\n📦 Checking dependencies...")
    try:
        import fastapi
        import uvicorn
        import sentence_transformers
        import faiss
        print("✅ Key dependencies are installed!")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Please install requirements: pip install -r requirements.txt")
        sys.exit(1)
    
    # Test basic functionality
    print("\n🧪 Running basic tests...")
    try:
        # Add project root to path
        project_root = Path(__file__).parent
        sys.path.insert(0, str(project_root))
        
        # Test imports
        from src.rag_system import RAGSystem
        from src.vector_store import create_vector_store
        
        # Test vector store creation
        vector_store = create_vector_store("faiss", dimension=384)
        
        print("✅ Basic functionality test passed!")
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        print("💡 Please check your installation and try again.")
        sys.exit(1)
    
    # Upload sample document
    print("\n📄 Setting up sample knowledge base...")
    try:
        from src.document_processor import DocumentProcessor
        
        processor = DocumentProcessor()
        sample_doc = Path("docs/sample_faq.md")
        
        if sample_doc.exists():
            result = processor.process_document(str(sample_doc))
            doc_ids = vector_store.add_documents([result])
            print(f"✅ Sample FAQ loaded ({len(doc_ids)} chunks)")
        else:
            print("⚠️  Sample FAQ not found, continuing without sample data")
        
    except Exception as e:
        print(f"⚠️  Could not load sample document: {e}")
        print("   You can upload documents later through the web interface")
    
    # Start the application
    print("\n🚀 Starting RAG Chatbot...")
    print("=" * 50)
    print("📍 Web Interface: http://127.0.0.1:8000")
    print("📍 API Docs: http://127.0.0.1:8000/docs")
    print("📍 Upload documents and start chatting!")
    print("=" * 50)
    print("Press Ctrl+C to stop the server")
    print("")
    
    try:
        # Import and start the main application
        from src.main import main as start_main
        start_main()
    except KeyboardInterrupt:
        print("\n\n👋 RAG Chatbot stopped. Thanks for using it!")
    except Exception as e:
        print(f"\n❌ Error starting the application: {e}")
        print("💡 Try running: python src/main.py")


if __name__ == "__main__":
    main()
