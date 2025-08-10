import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from flask import Flask, request, jsonify
from typing import Dict, Any
import threading
import uuid

from .rag_system import RAGSystem, ConversationManager
from config.settings import settings


class SlackBot:
    """Slack integration for the RAG chatbot."""
    
    def __init__(self, rag_system: RAGSystem, conversation_manager: ConversationManager):
        self.rag_system = rag_system
        self.conversation_manager = conversation_manager
        self.client = WebClient(token=settings.slack_bot_token)
        self.app = Flask(__name__)
        
        # Setup Flask routes
        self.setup_routes()
    
    def setup_routes(self):
        """Setup Flask routes for Slack events."""
        
        @self.app.route('/slack/events', methods=['POST'])
        def slack_events():
            """Handle Slack events."""
            data = request.json
            
            # Handle URL verification challenge
            if data.get('type') == 'url_verification':
                return jsonify({'challenge': data['challenge']})
            
            # Handle message events
            if data.get('type') == 'event_callback':
                event = data.get('event', {})
                
                if event.get('type') == 'message' and 'bot_id' not in event:
                    # This is a user message, not from a bot
                    self.handle_message(event)
            
            return '', 200
        
        @self.app.route('/slack/commands', methods=['POST'])
        def slack_commands():
            """Handle Slack slash commands."""
            command = request.form.get('command')
            text = request.form.get('text', '')
            user_id = request.form.get('user_id')
            channel_id = request.form.get('channel_id')
            
            if command == '/rag':
                # Handle RAG query command
                response = self.handle_query(text, user_id, channel_id)
                return jsonify({
                    'response_type': 'in_channel',
                    'text': response
                })
            
            return jsonify({'text': 'Unknown command'})
    
    def handle_message(self, event: Dict[str, Any]):
        """Handle incoming Slack messages."""
        try:
            user_id = event.get('user')
            channel_id = event.get('channel')
            text = event.get('text', '').strip()
            
            if not text:
                return
            
            # Generate conversation ID based on channel and user
            conversation_id = f"slack_{channel_id}_{user_id}"
            
            # Process the query
            response = self.handle_query(text, user_id, conversation_id)
            
            # Send response back to Slack
            self.send_message(channel_id, response)
            
        except Exception as e:
            print(f"Error handling Slack message: {str(e)}")
    
    def handle_query(self, text: str, user_id: str, conversation_id: str) -> str:
        """Process a query and return response."""
        try:
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
            
            # Format response with sources
            response = rag_response.answer
            if rag_response.sources:
                sources_text = "\n\n*Sources:*\n"
                for i, source in enumerate(rag_response.sources[:3]):  # Limit to top 3
                    sources_text += f"• {source['filename']}\n"
                response += sources_text
            
            # Add confidence if available
            if rag_response.confidence > 0:
                response += f"\n_Confidence: {rag_response.confidence:.1%}_"
            
            return response
            
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
    
    def send_message(self, channel: str, text: str):
        """Send a message to a Slack channel."""
        try:
            self.client.chat_postMessage(
                channel=channel,
                text=text
            )
        except SlackApiError as e:
            print(f"Error sending Slack message: {e.response['error']}")
    
    def run(self, port: int = 3000):
        """Run the Slack bot Flask app."""
        self.app.run(host='0.0.0.0', port=port, debug=False)


def start_slack_bot(rag_system: RAGSystem, conversation_manager: ConversationManager):
    """Start the Slack bot in a separate thread."""
    if not settings.slack_bot_token:
        print("Slack bot token not configured. Skipping Slack integration.")
        return None
    
    bot = SlackBot(rag_system, conversation_manager)
    
    # Run in a separate thread
    bot_thread = threading.Thread(target=bot.run, args=(3000,))
    bot_thread.daemon = True
    bot_thread.start()
    
    print("Slack bot started on port 3000")
    return bot
