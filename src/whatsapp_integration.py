from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
from flask import Flask, request
from typing import Dict, Any
import threading

from .rag_system import RAGSystem, ConversationManager
from config.settings import settings


class WhatsAppBot:
    """WhatsApp integration using Twilio for the RAG chatbot."""
    
    def __init__(self, rag_system: RAGSystem, conversation_manager: ConversationManager):
        self.rag_system = rag_system
        self.conversation_manager = conversation_manager
        
        # Initialize Twilio client
        if settings.twilio_account_sid and settings.twilio_auth_token:
            self.client = Client(settings.twilio_account_sid, settings.twilio_auth_token)
        else:
            self.client = None
        
        self.app = Flask(__name__)
        self.setup_routes()
    
    def setup_routes(self):
        """Setup Flask routes for WhatsApp webhooks."""
        
        @self.app.route('/whatsapp/webhook', methods=['POST'])
        def whatsapp_webhook():
            """Handle incoming WhatsApp messages."""
            # Get message details
            from_number = request.form.get('From')
            to_number = request.form.get('To')
            body = request.form.get('Body', '').strip()
            
            if not body:
                return '', 200
            
            # Generate conversation ID from phone number
            conversation_id = f"whatsapp_{from_number.replace('whatsapp:', '').replace('+', '')}"
            
            # Process the query
            response_text = self.handle_query(body, from_number, conversation_id)
            
            # Create TwiML response
            response = MessagingResponse()
            response.message(response_text)
            
            return str(response), 200, {'Content-Type': 'text/xml'}
        
        @self.app.route('/whatsapp/status', methods=['POST'])
        def whatsapp_status():
            """Handle WhatsApp message status updates."""
            # You can log message delivery status here if needed
            return '', 200
    
    def handle_query(self, text: str, from_number: str, conversation_id: str) -> str:
        """Process a query and return response."""
        try:
            # Check if client is initialized
            if not self.client:
                return "WhatsApp integration is not properly configured."
            
            # Get conversation history
            history = self.conversation_manager.get_conversation_history(conversation_id)
            
            # Add user message to history
            self.conversation_manager.add_message(conversation_id, "user", text)
            
            # Query RAG system
            rag_response = self.rag_system.query(text, history)
            
            # Add assistant response to history
            self.conversation_manager.add_message(
                conversation_id,
                "assistant",
                rag_response.answer,
                {"sources": [s['id'] for s in rag_response.sources]}
            )
            
            # Format response for WhatsApp (keeping it concise)
            response = rag_response.answer
            
            # Add sources if available (limited to avoid long messages)
            if rag_response.sources:
                response += "\n\n📚 *Sources:*"
                for i, source in enumerate(rag_response.sources[:2]):  # Limit to top 2
                    response += f"\n• {source['filename']}"
            
            # Add confidence
            if rag_response.confidence > 0:
                response += f"\n\n🎯 *Confidence:* {rag_response.confidence:.1%}"
            
            # Truncate if too long (WhatsApp has message limits)
            if len(response) > 1500:
                response = response[:1450] + "...\n\n_Message truncated due to length._"
            
            return response
            
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
    
    def send_message(self, to_number: str, message: str):
        """Send a message to a WhatsApp number."""
        try:
            if not self.client:
                print("Twilio client not initialized")
                return False
            
            # Ensure the number is in WhatsApp format
            if not to_number.startswith('whatsapp:'):
                to_number = f'whatsapp:{to_number}'
            
            message = self.client.messages.create(
                body=message,
                from_=settings.twilio_whatsapp_number,
                to=to_number
            )
            
            return True
            
        except Exception as e:
            print(f"Error sending WhatsApp message: {str(e)}")
            return False
    
    def run(self, port: int = 3001):
        """Run the WhatsApp bot Flask app."""
        self.app.run(host='0.0.0.0', port=port, debug=False)


def start_whatsapp_bot(rag_system: RAGSystem, conversation_manager: ConversationManager):
    """Start the WhatsApp bot in a separate thread."""
    if not settings.twilio_account_sid or not settings.twilio_auth_token:
        print("Twilio credentials not configured. Skipping WhatsApp integration.")
        return None
    
    bot = WhatsAppBot(rag_system, conversation_manager)
    
    # Run in a separate thread
    bot_thread = threading.Thread(target=bot.run, args=(3001,))
    bot_thread.daemon = True
    bot_thread.start()
    
    print("WhatsApp bot started on port 3001")
    return bot


# Utility functions for WhatsApp formatting
def format_whatsapp_message(text: str, max_length: int = 1500) -> str:
    """Format and truncate messages for WhatsApp."""
    if len(text) <= max_length:
        return text
    
    # Find a good place to cut the message
    truncate_point = text.rfind(' ', 0, max_length - 50)
    if truncate_point == -1:
        truncate_point = max_length - 50
    
    return text[:truncate_point] + "...\n\n_Message truncated due to length._"


def add_whatsapp_formatting(text: str) -> str:
    """Add WhatsApp-specific formatting."""
    # Convert markdown-like formatting to WhatsApp format
    text = text.replace('**', '*')  # Bold
    text = text.replace('__', '_')  # Italic
    
    return text
