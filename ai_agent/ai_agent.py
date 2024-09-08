import json
import yaml
import logging
import os
from typing import List, Dict

from .src.pdf_processor import PDFProcessor
from .src.openai_integration import OpenAIIntegration
from .src.slack_integration import SlackIntegration
from .src.vector_store import VectorStore


class AIAgent:
    """
        A class to manage AI operations including PDF processing, question answering, and Slack integration.

        Args:
            pdf_path (str): Path to the PDF file to process.
            slack_channel (str): Slack channel where messages will be posted.
            config_path (str): Path to the YAML configuration file.
    """
    def __init__(self, pdf_path: str):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        config_path = os.path.join(project_root, 'config.yaml')
        self.config = self._load_config(config_path)
        logging.basicConfig(level=self.config['logging']['level'], 
                            format=self.config['logging']['format'], filename=f"{project_root}/log")
        self.logger = logging.getLogger(__name__)
        
        self.pdf_processor = PDFProcessor(pdf_path, self.config)
        self.openai_integration = OpenAIIntegration(self.config)
        self.slack_integration = SlackIntegration(self.config)
        self.vector_store = VectorStore(self.config)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load and validate the configuration from a YAML file."""
        try:
            with open(config_path, 'r') as config_file:
                config = yaml.safe_load(config_file)
                if not all(key in config for key in ['logging']):
                    raise KeyError("Missing required keys in configuration file.")
                return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")
        
    def process_pdf(self) -> None:
        """Extract text from PDF, split it into chunks, and add to the vector store."""
        try:
            text = self.pdf_processor.extract_text()
            chunks = self.pdf_processor.split_into_chunks(text)
            self.vector_store.add_texts(chunks[:8])
            self.logger.info("PDF processing and vector store update complete")
        except Exception as e:
            self.logger.error(f"Error processing PDF: {e}")
            
    def answer_questions(self, questions: List[str]) -> List[Dict[str, str]]:
        """
        Answer a list of questions based on the processed PDF.

        Args:
            questions (List[str]): List of questions to be answered.

        Returns:
            List[Dict[str, str]]: List of dictionaries containing questions and their answers.
        """
        results = []
        for question in questions:
            try:
                self.logger.info(f"Processing question: {question}")
                relevant_chunks = self.vector_store.similarity_search(question)
                context = " ".join([chunk for chunk, _ in relevant_chunks])
                answer = self.openai_integration.extract_answer(question, context)
                if answer.lower().startswith("i don't know") or answer.lower().startswith("i'm not sure"):
                    answer = "Data Not Available"
                results.append({"question": question, "answer": answer})
                self.logger.info(f"Answer for question '{question}': {answer}")
            except Exception as e:
                self.logger.error(f"Error answering question '{question}': {e}")
        return results

    def post_to_slack(self, message: str) -> None:
        """
        Post a message to a Slack channel.

        Args:
            message (str): Message to be posted to Slack.
        """
        try:
            self.slack_integration.post_message(message)
            self.logger.info("Message posted to Slack successfully.")
        except Exception as e:
            self.logger.error(f"Error posting message to Slack: {e}")
            raise

    def run(self, questions: List[str]) -> None:
        try:
            self.logger.info("Processing PDF")
            self.process_pdf()
            results = self.answer_questions(questions)
            output = json.dumps(results, indent=2)
            self.logger.info("Posting results to Slack")
            self.post_to_slack(f"Questions and Answers:\n```\n{output}\n```")
            self.logger.info("AI agent process completed successfully")
            return output
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            self.logger.error(error_message)
            self.post_to_slack(error_message)