#!/usr/bin/env python3
"""
RAG Chatbot - Main Application
A comprehensive RAG-powered chatbot with multi-channel support
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag_system import RAGSystem, ConversationManager
from src.slack_integration import start_slack_bot
from src.whatsapp_integration import start_whatsapp_bot
from config.settings import settings


def initialize_system():
    """Initialize the RAG system and conversation manager."""
    print("🚀 Initializing RAG Chatbot System...")
    
    # Initialize RAG system
    rag_system = RAGSystem()
    print(f"✅ RAG System initialized with {settings.vector_db_type} vector store")
    
    # Initialize conversation manager
    conversation_manager = ConversationManager()
    print("✅ Conversation Manager initialized")
    
    return rag_system, conversation_manager


def start_integrations(rag_system: RAGSystem, conversation_manager: ConversationManager):
    """Start all channel integrations."""
    print("\n📱 Starting channel integrations...")
    
    # Start Slack bot
    slack_bot = start_slack_bot(rag_system, conversation_manager)
    if slack_bot:
        print("✅ Slack integration started")
    
    # Start WhatsApp bot
    whatsapp_bot = start_whatsapp_bot(rag_system, conversation_manager)
    if whatsapp_bot:
        print("✅ WhatsApp integration started")
    
    return slack_bot, whatsapp_bot


def start_web_app():
    """Start the FastAPI web application."""
    import uvicorn
    from src.web_app import app
    
    print(f"\n🌐 Starting web application on {settings.host}:{settings.port}")
    
    # Start the web app
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level="info"
    )


def display_startup_info():
    """Display startup information and instructions."""
    print("\n" + "="*60)
    print("🤖 RAG CHATBOT - SUCCESSFULLY STARTED!")
    print("="*60)
    print(f"📊 Vector Database: {settings.vector_db_type}")
    print(f"🧠 LLM Model: {settings.llm_model}")
    print(f"🔧 Embedding Model: {settings.embedding_model}")
    print("\n📍 Available Endpoints:")
    print(f"   • Web Interface: http://{settings.host}:{settings.port}")
    print(f"   • API Docs: http://{settings.host}:{settings.port}/docs")
    print(f"   • Health Check: http://{settings.host}:{settings.port}/health")
    
    if settings.slack_bot_token:
        print(f"   • Slack Events: http://{settings.host}:3000/slack/events")
        print(f"   • Slack Commands: http://{settings.host}:3000/slack/commands")
    
    if settings.twilio_account_sid:
        print(f"   • WhatsApp Webhook: http://{settings.host}:3001/whatsapp/webhook")
    
    print("\n📝 Usage Instructions:")
    print("1. Open the web interface to upload documents and chat")
    print("2. Use the API endpoints for programmatic access")
    print("3. Configure Slack/WhatsApp webhooks for channel integration")
    print("\n💡 Tips:")
    print("• Upload PDF, DOCX, TXT, MD, or HTML files")
    print("• Add content from URLs")
    print("• View source attribution and confidence scores")
    print("• Conversations are context-aware across channels")
    print("="*60)


def main():
    """Main application entry point."""
    try:
        # Display banner
        print("""
    ██████╗  █████╗  ██████╗      ██████╗██╗  ██╗ █████╗ ████████╗██████╗  ██████╗ ████████╗
    ██╔══██╗██╔══██╗██╔════╝     ██╔════╝██║  ██║██╔══██╗╚══██╔══╝██╔══██╗██╔═══██╗╚══██╔══╝
    ██████╔╝███████║██║  ███╗    ██║     ███████║███████║   ██║   ██████╔╝██║   ██║   ██║   
    ██╔══██╗██╔══██║██║   ██║    ██║     ██╔══██║██╔══██║   ██║   ██╔══██╗██║   ██║   ██║   
    ██║  ██║██║  ██║╚██████╔╝    ╚██████╗██║  ██║██║  ██║   ██║   ██████╔╝╚██████╔╝   ██║   
    ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝      ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═════╝  ╚═════╝    ╚═╝   
    """)
        
        # Initialize system
        rag_system, conversation_manager = initialize_system()
        
        # Start channel integrations
        start_integrations(rag_system, conversation_manager)
        
        # Display startup info
        display_startup_info()
        
        # Start web application (this blocks)
        start_web_app()
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down RAG Chatbot...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error starting RAG Chatbot: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
