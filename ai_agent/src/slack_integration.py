import os
import logging

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

class SlackIntegration:
    """
    A class to manage Slack integration for posting messages to Slack channels.

    Args:
        config (dict): Configuration dictionary that might be used for future extensions.
    """
    def __init__(self, config: dict):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.config = config
        # Initialize Slack client
        token = os.getenv("SLACK_BOT_TOKEN")
        if not token:
            raise ValueError("SLACK_BOT_TOKEN environment variable is not set")
        
        self.client = WebClient(token=token)
    
        
    def post_message(self, message: str) -> None:
        """
        Post a message to a Slack channel.
        
        Args:
            message (str): The message to send.
        """
        try:
            response = self.client.chat_postMessage(channel=self.config['slack']['slack_channel'], text=message)
            if not response["ok"]:
                self.logger.error(f"Failed to send message: {response['error']}")
        except SlackApiError as e:
            self.logger.error(f"Error posting message to Slack: {str(e)}")