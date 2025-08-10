from pydantic import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    # API Keys
    openai_api_key: str
    pinecone_api_key: Optional[str] = None
    pinecone_environment: Optional[str] = None
    
    # Slack Integration
    slack_bot_token: Optional[str] = None
    slack_signing_secret: Optional[str] = None
    
    # WhatsApp/Twilio Integration
    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    twilio_whatsapp_number: Optional[str] = None
    
    # Database
    database_url: str = "sqlite:///./chatbot.db"
    
    # Application Settings
    vector_db_type: str = "faiss"  # Options: pinecone, faiss, chromadb
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "gpt-3.5-turbo"
    max_context_length: int = 4000
    top_k_results: int = 3
    
    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000
    
    class Config:
        env_file = ".env"


settings = Settings()
