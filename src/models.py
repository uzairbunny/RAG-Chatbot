from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    content = Column(Text)
    file_type = Column(String)
    upload_date = Column(DateTime, default=datetime.utcnow)
    metadata = Column(Text)  # JSON string for additional metadata
    
    # Relationship with chunks
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    chunk_text = Column(Text)
    chunk_index = Column(Integer)
    embedding_id = Column(String)  # ID in vector database
    
    # Relationship with document
    document = relationship("Document", back_populates="chunks")


class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    channel = Column(String)  # web, slack, whatsapp
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship with messages
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    role = Column(String)  # user, assistant
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    metadata = Column(Text)  # JSON string for additional metadata
    
    # Relationship with conversation
    conversation = relationship("Conversation", back_populates="messages")
