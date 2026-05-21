# rasa_client.py - Rasa Server Client

import requests
from datetime import datetime


class RasaClient:
    """Client for interacting with Rasa server"""

    def __init__(self, server_url="http://localhost:5005"):
        self.server_url = server_url
        self.webhook_url = f"{server_url}/webhooks/rest/webhook"
        self.nlu_url = f"{server_url}/model/parse"
        self.session_id = None
        self.is_connected = False

    def check_connection(self):
        """Check if Rasa server is running"""
        try:
            response = requests.get(f"{self.server_url}/", timeout=5)
            self.is_connected = response.status_code == 200
            return self.is_connected
        except requests.exceptions.RequestException:
            self.is_connected = False
            return False

    def get_session_id(self):
        """Generate or get session ID"""
        if not self.session_id:
            self.session_id = f"user_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        return self.session_id

    def parse_message(self, message):
        """
        Parse message using Rasa NLU to get intent and entities.
        Returns: dict with intent, entities, and confidence
        """
        try:
            response = requests.post(
                self.nlu_url,
                json={"text": message},
                timeout=10
            )
            if response.status_code == 200:
                result = response.json()
                return {
                    "intent": result.get("intent", {}).get("name", "unknown"),
                    "confidence": result.get("intent", {}).get("confidence", 0.0),
                    "entities": result.get("entities", []),
                    "intent_ranking": result.get("intent_ranking", [])
                }
        except requests.exceptions.RequestException as e:
            print(f"NLU parse error: {e}")

        return {
            "intent": "unknown",
            "confidence": 0.0,
            "entities": [],
            "intent_ranking": []
        }

    def send_message(self, message, sender_id=None):
        """
        Send message to Rasa and get response.
        Returns: list of response messages
        """
        if sender_id is None:
            sender_id = self.get_session_id()

        try:
            response = requests.post(
                self.webhook_url,
                json={"sender": sender_id, "message": message},
                timeout=30
            )
            if response.status_code == 200:
                return response.json()
            else:
                return [{"text": "Sorry, I couldn't process that request."}]
        except requests.exceptions.RequestException as e:
            print(f"Rasa request error: {e}")
            return None

    def get_tracker(self, sender_id=None):
        """Get conversation tracker from Rasa"""
        if sender_id is None:
            sender_id = self.get_session_id()

        try:
            response = requests.get(
                f"{self.server_url}/conversations/{sender_id}/tracker",
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
        except requests.exceptions.RequestException:
            pass

        return None

    def reset_conversation(self, sender_id=None):
        """Reset conversation with Rasa"""
        if sender_id is None:
            sender_id = self.get_session_id()

        try:
            response = requests.post(
                f"{self.server_url}/conversations/{sender_id}/tracker/events",
                json={"event": "restart"},
                timeout=10
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False