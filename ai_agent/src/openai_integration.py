import logging

from openai import OpenAI
class OpenAIIntegration:
    """
        Initialize the OpenAIIntegration with configuration.

        Args:
            config (dict): Configuration dictionary with OpenAI settings.
    """
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._validate_config()

    def _validate_config(self):
        """Validate configuration parameters."""
        if 'openai' not in self.config:
            raise ValueError("Configuration must contain 'openai' key.")
        required_keys = ['model', 'temperature']
        for key in required_keys:
            if key not in self.config['openai']:
                raise ValueError(f"Configuration must contain 'openai' with '{key}' key.")
            if not isinstance(self.config['openai'][key], (int, float, str)):
                raise ValueError(f"'{key}' in configuration must be a valid type.")
    def extract_answer(self, question: str, context: str) -> str:
        try:
            response = OpenAI().chat.completions.create(
                model=self.config['openai']['model'],
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the given context. If the question is a word-for-word match in the context, provide the exact answer. If you're not confident about the answer, respond with 'Data Not Available'."},
                    {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
                ],
                temperature=self.config['openai']['temperature'],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"Error extracting answer from OpenAI: {str(e)}")