from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import uuid
import tempfile
from pathlib import Path

from .rag_system import RAGSystem, ConversationManager
from config.settings import settings

app = FastAPI(
    title="RAG Chatbot API",
    description="A Retrieval-Augmented Generation chatbot with document ingestion capabilities",
    version="1.0.0"
)

# CORS middleware for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG system and conversation manager
rag_system = RAGSystem()
conversation_manager = ConversationManager()

# Pydantic models for API requests
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    user_id: Optional[str] = "anonymous"

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    sources: List[Dict[str, Any]]
    confidence: float
    metadata: Dict[str, Any]

class URLRequest(BaseModel):
    url: str

class DocumentResponse(BaseModel):
    status: str
    message: str
    details: Optional[Dict[str, Any]] = None

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main chat interface."""
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG Chatbot</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; border-radius: 10px; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            .header { text-align: center; margin-bottom: 30px; }
            .chat-container { border: 1px solid #ddd; height: 400px; overflow-y: auto; padding: 15px; border-radius: 5px; background-color: #f9f9f9; }
            .message { margin-bottom: 15px; padding: 10px; border-radius: 5px; }
            .user-message { background-color: #007bff; color: white; text-align: right; }
            .bot-message { background-color: #e9ecef; color: #333; }
            .input-container { margin-top: 20px; display: flex; gap: 10px; }
            .input-container input { flex-grow: 1; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
            .input-container button { padding: 10px 20px; background-color: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
            .upload-section { margin-top: 20px; padding: 20px; border: 1px solid #ddd; border-radius: 5px; background-color: #f8f9fa; }
            .sources { margin-top: 10px; font-size: 12px; color: #666; }
            .confidence { font-size: 12px; color: #28a745; margin-top: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 RAG Chatbot</h1>
                <p>Ask questions about uploaded documents</p>
            </div>
            
            <div class="upload-section">
                <h3>📁 Upload Documents</h3>
                <input type="file" id="file-input" multiple accept=".pdf,.docx,.txt,.md,.html">
                <button onclick="uploadFiles()">Upload</button>
                <div style="margin-top: 10px;">
                    <input type="url" id="url-input" placeholder="Or enter a URL...">
                    <button onclick="uploadURL()">Add URL</button>
                </div>
                <div id="upload-status"></div>
            </div>
            
            <div class="chat-container" id="chat-container">
                <div class="message bot-message">
                    Hello! I'm your RAG chatbot. Upload some documents and ask me questions about them.
                </div>
            </div>
            
            <div class="input-container">
                <input type="text" id="message-input" placeholder="Type your message..." onkeypress="handleKeyPress(event)">
                <button onclick="sendMessage()">Send</button>
            </div>
        </div>

        <script>
            let conversationId = null;

            function handleKeyPress(event) {
                if (event.key === 'Enter') {
                    sendMessage();
                }
            }

            function addMessage(content, isUser, sources = [], confidence = null) {
                const chatContainer = document.getElementById('chat-container');
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message ' + (isUser ? 'user-message' : 'bot-message');
                
                let messageContent = content;
                
                if (sources && sources.length > 0 && !isUser) {
                    const sourcesList = sources.map(s => `📄 ${s.filename}`).join(', ');
                    messageContent += `<div class="sources">Sources: ${sourcesList}</div>`;
                }
                
                if (confidence !== null && !isUser) {
                    messageContent += `<div class="confidence">Confidence: ${(confidence * 100).toFixed(1)}%</div>`;
                }
                
                messageDiv.innerHTML = messageContent;
                chatContainer.appendChild(messageDiv);
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }

            async function sendMessage() {
                const input = document.getElementById('message-input');
                const message = input.value.trim();
                if (!message) return;

                addMessage(message, true);
                input.value = '';

                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ 
                            message: message, 
                            conversation_id: conversationId 
                        })
                    });

                    const data = await response.json();
                    conversationId = data.conversation_id;
                    addMessage(data.response, false, data.sources, data.confidence);
                } catch (error) {
                    addMessage('Sorry, there was an error processing your request.', false);
                }
            }

            async function uploadFiles() {
                const fileInput = document.getElementById('file-input');
                const files = fileInput.files;
                const statusDiv = document.getElementById('upload-status');
                
                if (files.length === 0) {
                    statusDiv.innerHTML = '<div style="color: red;">Please select files to upload.</div>';
                    return;
                }

                statusDiv.innerHTML = '<div style="color: blue;">Uploading...</div>';

                for (let file of files) {
                    const formData = new FormData();
                    formData.append('file', file);

                    try {
                        const response = await fetch('/upload-document', {
                            method: 'POST',
                            body: formData
                        });

                        const data = await response.json();
                        if (data.status === 'success') {
                            statusDiv.innerHTML += `<div style="color: green;">✅ ${file.name} uploaded successfully</div>`;
                        } else {
                            statusDiv.innerHTML += `<div style="color: red;">❌ Failed to upload ${file.name}: ${data.message}</div>`;
                        }
                    } catch (error) {
                        statusDiv.innerHTML += `<div style="color: red;">❌ Error uploading ${file.name}</div>`;
                    }
                }

                fileInput.value = '';
            }

            async function uploadURL() {
                const urlInput = document.getElementById('url-input');
                const url = urlInput.value.trim();
                const statusDiv = document.getElementById('upload-status');
                
                if (!url) {
                    statusDiv.innerHTML = '<div style="color: red;">Please enter a URL.</div>';
                    return;
                }

                statusDiv.innerHTML = '<div style="color: blue;">Processing URL...</div>';

                try {
                    const response = await fetch('/upload-url', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ url: url })
                    });

                    const data = await response.json();
                    if (data.status === 'success') {
                        statusDiv.innerHTML = '<div style="color: green;">✅ URL content processed successfully</div>';
                    } else {
                        statusDiv.innerHTML = `<div style="color: red;">❌ Failed to process URL: ${data.message}</div>`;
                    }
                } catch (error) {
                    statusDiv.innerHTML = '<div style="color: red;">❌ Error processing URL</div>';
                }

                urlInput.value = '';
            }
        </script>
    </body>
    </html>
    """)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat requests."""
    try:
        # Generate conversation ID if not provided
        if not request.conversation_id:
            request.conversation_id = str(uuid.uuid4())
        
        # Get conversation history
        history = conversation_manager.get_conversation_history(request.conversation_id)
        
        # Add user message to history
        conversation_manager.add_message(
            request.conversation_id, 
            "user", 
            request.message
        )
        
        # Query RAG system
        rag_response = rag_system.query(request.message, history)
        
        # Add assistant response to history
        conversation_manager.add_message(
            request.conversation_id,
            "assistant", 
            rag_response.answer,
            {"sources": [s['id'] for s in rag_response.sources]}
        )
        
        return ChatResponse(
            response=rag_response.answer,
            conversation_id=request.conversation_id,
            sources=rag_response.sources,
            confidence=rag_response.confidence,
            metadata=rag_response.metadata
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-document", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document."""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Process document
            result = rag_system.add_document(tmp_file_path)
            
            if result["status"] == "success":
                return DocumentResponse(
                    status="success",
                    message=f"Document '{file.filename}' processed successfully. Added {result['chunks_added']} chunks.",
                    details=result
                )
            else:
                return DocumentResponse(
                    status="error",
                    message=f"Failed to process document: {result['error']}"
                )
        
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-url", response_model=DocumentResponse)
async def upload_url(request: URLRequest):
    """Process content from a URL."""
    try:
        result = rag_system.add_document_from_url(request.url)
        
        if result["status"] == "success":
            return DocumentResponse(
                status="success",
                message=f"URL content processed successfully. Added {result['chunks_added']} chunks.",
                details=result
            )
        else:
            return DocumentResponse(
                status="error",
                message=f"Failed to process URL: {result['error']}"
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get knowledge base statistics."""
    try:
        stats = rag_system.get_knowledge_base_stats()
        return {
            "status": "success",
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversations")
async def get_conversations():
    """Get all conversation IDs."""
    try:
        conversations = conversation_manager.get_all_conversations()
        return {
            "status": "success",
            "conversations": conversations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/conversations/{conversation_id}")
async def clear_conversation(conversation_id: str):
    """Clear a specific conversation."""
    try:
        conversation_manager.clear_conversation(conversation_id)
        return {
            "status": "success",
            "message": f"Conversation {conversation_id} cleared"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)
