# RAG Chatbot 🤖

A comprehensive Retrieval-Augmented Generation chatbot that can answer customer queries in real time by retrieving answers from a company's internal knowledge base using Large Language Models.

## Features ✨

- **📄 Document Ingestion**: Upload PDF, DOCX, TXT, MD, and HTML files
- **🔍 Smart Search**: Semantic search using vector embeddings
- **💾 Multiple Vector Databases**: Support for FAISS, Pinecone, and ChromaDB
- **🧠 RAG Pipeline**: Advanced retrieval-augmented generation
- **💬 Context-Aware Conversations**: Maintains conversation history
- **🌐 Multi-Channel Integration**: Web, Slack, and WhatsApp support
- **📊 Source Attribution**: Shows which documents were used for answers
- **🎯 Confidence Scores**: Provides confidence ratings for responses

## Architecture 🏗️

```
User Query → Embedding → Vector Search → Context Retrieval → LLM → Response
                ↓
        Vector Database (FAISS/Pinecone/ChromaDB)
                ↑
        Document Processing → Chunking → Embeddings
```

## Quick Start 🚀

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd rag-chatbot

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Copy the environment template and configure your settings:

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```env
# Required
OPENAI_API_KEY=your_openai_api_key

# Optional integrations
PINECONE_API_KEY=your_pinecone_key
SLACK_BOT_TOKEN=xoxb-your-slack-token
TWILIO_ACCOUNT_SID=your_twilio_sid
```

### 3. Run the Application

```bash
# Start the main application
python src/main.py
```

The application will be available at:
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Usage 📖

### Web Interface

1. Open http://localhost:8000 in your browser
2. Upload documents using the file upload section
3. Start asking questions about your documents
4. View source attribution and confidence scores

### API Usage

```python
import requests

# Upload a document
with open('document.pdf', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/upload-document',
        files={'file': f}
    )

# Ask a question
response = requests.post(
    'http://localhost:8000/chat',
    json={
        'message': 'What are the key features?',
        'conversation_id': 'user123'
    }
)

print(response.json())
```

### Slack Integration

1. Create a Slack app at [api.slack.com](https://api.slack.com)
2. Configure bot token scopes: `channels:read`, `chat:write`, `im:read`, `im:write`
3. Set up event subscriptions pointing to: `http://your-server:3000/slack/events`
4. Create a slash command `/rag` pointing to: `http://your-server:3000/slack/commands`

### WhatsApp Integration

1. Set up a Twilio account and WhatsApp sandbox
2. Configure webhook URL: `http://your-server:3001/whatsapp/webhook`
3. Add your Twilio credentials to `.env`

## Configuration Options ⚙️

### Vector Database Options

**FAISS (Default)**
- Local storage
- Good for development and small deployments
- No external dependencies

**Pinecone**
- Cloud-based vector database
- Highly scalable
- Requires API key

**ChromaDB**
- Local persistent database
- Good balance of features and simplicity
- Built-in persistence

### LLM Models

Supported OpenAI models:
- `gpt-3.5-turbo` (default)
- `gpt-4`
- `gpt-4-turbo`

### Embedding Models

Default: `sentence-transformers/all-MiniLM-L6-v2`

Other options:
- `sentence-transformers/all-mpnet-base-v2`
- `text-embedding-ada-002` (OpenAI)

## API Endpoints 🔗

### Document Management
- `POST /upload-document` - Upload a document
- `POST /upload-url` - Process content from URL
- `GET /stats` - Get knowledge base statistics

### Chat
- `POST /chat` - Send a message and get response
- `GET /conversations` - List all conversations
- `DELETE /conversations/{id}` - Clear conversation history

### Health
- `GET /health` - Health check endpoint

## Advanced Configuration 🔧

### Custom Chunking Strategy

```python
from src.document_processor import DocumentProcessor

processor = DocumentProcessor()
result = processor.process_document(
    'document.pdf',
    chunk_size=1000,  # tokens per chunk
    chunk_overlap=100  # overlap between chunks
)
```

### Custom System Prompt

Modify the system prompt in `src/rag_system.py`:

```python
system_prompt = """
You are a helpful customer service assistant.
Answer questions based only on the provided context.
Be friendly and professional in your responses.
"""
```

## Monitoring and Analytics 📊

### Built-in Metrics
- Response confidence scores
- Source document attribution
- Token usage tracking
- Conversation analytics

### Health Monitoring
- Vector database status
- LLM API connectivity
- Memory usage
- Response times

## Security Considerations 🔒

### Data Privacy
- Documents processed locally by default
- Conversation history stored in memory
- API keys stored in environment variables

### Production Deployment
- Use HTTPS for all endpoints
- Implement rate limiting
- Add authentication middleware
- Configure CORS appropriately
- Use secure vector database connections

## Troubleshooting 🔧

### Common Issues

**"No module named 'config'"**
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/rag-chatbot"
```

**Vector database connection errors**
- Check API keys in `.env`
- Verify network connectivity
- Ensure vector database service is running

**Low response quality**
- Upload more relevant documents
- Adjust chunk size and overlap
- Try different embedding models
- Increase top_k retrieval count

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Development 👨‍💻

### Project Structure
```
rag-chatbot/
├── src/
│   ├── document_processor.py   # Document ingestion
│   ├── vector_store.py         # Vector database interfaces
│   ├── rag_system.py          # Core RAG logic
│   ├── web_app.py             # FastAPI web application
│   ├── slack_integration.py   # Slack bot
│   ├── whatsapp_integration.py # WhatsApp bot
│   └── main.py                # Application entry point
├── config/
│   └── settings.py            # Configuration management
├── docs/                      # Sample documents
├── tests/                     # Test files
└── requirements.txt           # Dependencies
```

### Running Tests
```bash
pytest tests/ -v
```

### Code Quality
```bash
# Format code
black src/

# Lint code
flake8 src/

# Type checking
mypy src/
```

## Contributing 🤝

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License 📄

This project is licensed under the MIT License - see the LICENSE file for details.

## Support 💬

- 📧 Email: support@example.com
- 💬 Slack: #rag-chatbot
- 📖 Documentation: https://docs.example.com
- 🐛 Issues: GitHub Issues

## Roadmap 🗺️

### Upcoming Features
- [ ] Multi-language support
- [ ] Advanced analytics dashboard
- [ ] Integration with more messaging platforms
- [ ] Voice message support
- [ ] Custom embedding model training
- [ ] Batch document processing
- [ ] Advanced user management
- [ ] Audit logging
- [ ] Performance optimization

---

Made with ❤️ by the RAG Chatbot Team
