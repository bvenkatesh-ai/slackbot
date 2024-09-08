import logging
from typing import List

import PyPDF2
class PDFProcessor:
    """
        Initialize the VectorStore with configuration.
        
        Args:
            config (dict): Configuration dictionary with chunk size and overlap.
    """
    def __init__(self, file_path: str, config: dict):
        self.file_path = file_path
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._validate_config()

    def _validate_config(self):
        """Validate configuration parameters."""
        if 'pdf' not in self.config:
            raise ValueError("Configuration must contain 'pdf' key.")
        if 'chunk_size' not in self.config['pdf'] or 'chunk_overlap' not in self.config['pdf']:
            raise ValueError("Configuration must contain 'chunk_size' and 'chunk_overlap' under 'pdf' key.")
        if not isinstance(self.config['pdf']['chunk_size'], int) or not isinstance(self.config['pdf']['chunk_overlap'], int):
            raise ValueError("'chunk_size' and 'chunk_overlap' must be integers.")
        if self.config['pdf']['chunk_size'] <= 0 or self.config['pdf']['chunk_overlap'] < 0:
            raise ValueError("'chunk_size' must be positive and 'chunk_overlap' must be non-negative.")
        if self.config['pdf']['chunk_size'] < self.config['pdf']['chunk_overlap']:
            raise ValueError("'chunk_size' must be greater than 'chunk_overlap'")
       

    def extract_text(self) -> str:
        """Extract text from a PDF file."""
        self.logger.debug(f"Starting text extraction from file: {self.file_path}")
        try:
            with open(self.file_path, 'rb') as file:
                self.logger.debug("Opening PDF file")
                reader = PyPDF2.PdfReader(file)
                if not reader.pages:
                    self.logger.warning("The PDF file is empty or contains no pages.")
                    return ""
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
            self.logger.debug("Text extraction completed")
            return text
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")

    def split_into_chunks(self, text: str) -> List[str]:
        """Split text into chunks based on configuration."""
        self.logger.debug("Starting text chunking")
        chunks = []
        start = 0 
        chunk_size = self.config['pdf']['chunk_size']
        chunk_overlap = self.config['pdf']['chunk_overlap']   
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            self.logger.debug(f"Chunk created from position {start} to {end}")
            start += chunk_size - chunk_overlap
        self.logger.debug("Text chunking completed")
        return chunks