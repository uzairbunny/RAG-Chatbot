#!/usr/bin/env python3
"""
Simple test script for the RAG chatbot system
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_basic_functionality():
    """Test basic RAG functionality with sample document."""
    print("🧪 Testing RAG Chatbot Basic Functionality")
    print("=" * 50)
    
    try:
        # Test imports
        print("1. Testing imports...")
        from src.rag_system import RAGSystem, ConversationManager
        from src.vector_store import create_vector_store
        print("✅ Imports successful")
        
        # Test environment
        print("2. Checking environment...")
        from config.settings import settings
        
        # Create a minimal .env for testing if it doesn't exist
        env_path = project_root / '.env'
        if not env_path.exists():
            print("⚠️  No .env file found. Creating minimal test configuration...")
            with open(env_path, 'w') as f:
                f.write("OPENAI_API_KEY=test_key_for_testing\n")
                f.write("VECTOR_DB_TYPE=faiss\n")
            print("✅ Test .env created (you'll need to add a real OpenAI API key)")
        else:
            print("✅ Environment configuration found")
        
        # Test vector store creation
        print("3. Testing vector store...")
        try:
            vector_store = create_vector_store("faiss", dimension=384)
            print("✅ FAISS vector store created successfully")
        except Exception as e:
            print(f"❌ Vector store creation failed: {e}")
            return False
        
        # Test document processor
        print("4. Testing document processor...")
        from src.document_processor import DocumentProcessor
        
        processor = DocumentProcessor()
        
        # Test with sample FAQ document
        sample_doc_path = project_root / 'docs' / 'sample_faq.md'
        if sample_doc_path.exists():
            try:
                result = processor.process_document(str(sample_doc_path))
                print(f"✅ Document processed: {result['metadata']['total_chunks']} chunks created")
                
                # Test adding to vector store
                doc_ids = vector_store.add_documents([result])
                print(f"✅ {len(doc_ids)} chunks added to vector store")
                
            except Exception as e:
                print(f"❌ Document processing failed: {e}")
                return False
        else:
            print("⚠️  Sample document not found, skipping document processing test")
        
        # Test conversation manager
        print("5. Testing conversation manager...")
        conv_manager = ConversationManager()
        conv_manager.add_message("test_conv", "user", "Hello")
        conv_manager.add_message("test_conv", "assistant", "Hi there!")
        
        history = conv_manager.get_conversation_history("test_conv")
        if len(history) == 2:
            print("✅ Conversation manager working correctly")
        else:
            print("❌ Conversation manager test failed")
            return False
        
        print("\n🎉 All basic functionality tests passed!")
        print("\n📝 Next Steps:")
        print("1. Add your OpenAI API key to the .env file")
        print("2. Run: python src/main.py")
        print("3. Open http://localhost:8000 in your browser")
        print("4. Upload documents and start chatting!")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you've installed all requirements: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_web_app_imports():
    """Test if web app can be imported."""
    print("\n🌐 Testing Web Application Imports")
    print("=" * 40)
    
    try:
        from src.web_app import app
        print("✅ FastAPI app imports successfully")
        
        # Test if we can get the routes
        routes = [route.path for route in app.routes]
        expected_routes = ["/", "/chat", "/upload-document", "/health"]
        
        for route in expected_routes:
            if route in routes:
                print(f"✅ Route {route} found")
            else:
                print(f"⚠️  Route {route} not found")
        
        return True
    except Exception as e:
        print(f"❌ Web app import failed: {e}")
        return False


def check_requirements():
    """Check if all required packages are installed."""
    print("\n📦 Checking Required Packages")
    print("=" * 35)
    
    required_packages = [
        'fastapi', 'uvicorn', 'openai', 'sentence-transformers',
        'faiss-cpu', 'chromadb', 'PyPDF2', 'python-docx',
        'beautifulsoup4', 'requests', 'pandas', 'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("💡 Install with: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All required packages are installed!")
        return True


if __name__ == "__main__":
    print("🚀 RAG Chatbot Test Suite")
    print("=" * 60)
    
    # Check requirements first
    requirements_ok = check_requirements()
    
    if requirements_ok:
        # Run basic functionality tests
        basic_tests_ok = test_basic_functionality()
        
        # Test web app
        web_app_ok = test_web_app_imports()
        
        if basic_tests_ok and web_app_ok:
            print("\n🎉 All tests passed! Your RAG chatbot is ready to go!")
        else:
            print("\n⚠️  Some tests failed. Please check the errors above.")
    else:
        print("\n❌ Please install missing requirements first.")
    
    print("\n" + "=" * 60)
