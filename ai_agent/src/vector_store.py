import logging
from typing import List, Tuple

import numpy as np
import faiss
from openai import OpenAI

class VectorStore:
    """
        Initialize the VectorStore with configuration.
        
        Args:
            config (dict): Configuration dictionary with keys for OpenAI and vector store settings.
    """
    def __init__(self, config: dict):
        self.config = config
        self.index = None
        self.texts = []
        self.logger = logging.getLogger(__name__)
        self._validate_config()

    def _validate_config(self):
        """Validate configuration parameters."""
        if 'openai' not in self.config or 'embedding_model' not in self.config['openai']:
            raise ValueError("Configuration must contain 'openai' with 'embedding_model' key.")
        if 'vector_store' not in self.config or 'dimension' not in self.config['vector_store'] or 'similarity_top_k' not in self.config['vector_store']:
            raise ValueError("Configuration must contain 'vector_store' with 'dimension' and 'similarity_top_k' keys.")
        if not isinstance(self.config['vector_store']['dimension'], int) or self.config['vector_store']['dimension'] <= 0:
            raise ValueError("'dimension' must be a positive integer.")
        if not isinstance(self.config['vector_store']['similarity_top_k'], int) or self.config['vector_store']['similarity_top_k'] <= 0:
            raise ValueError("'similarity_top_k' must be a positive integer.")
    def _get_embedding(self, text: str) -> List[float]:
        """
        Get the embedding vector for a given text using OpenAI API.
        
        Args:
            text (str): The text to embed.
        
        Returns:
            List[float]: The embedding vector.
        """
        try:
            response = OpenAI().embeddings.create(
                input=[text],
                model=self.config['openai']['embedding_model']
            )
            return response.data[0].embedding
        except Exception as e:
            self.logger.error(f"Error getting embedding for text: {text}. Error: {str(e)}", exc_info=True)
            raise

    def add_texts(self, texts: List[str]):
        """
        Add texts to the vector store by embedding them and adding them to the FAISS index.
        
        Args:
            texts (List[str]): The list of texts to add.
        """
        self.texts.extend(texts)
        embeddings = [self._get_embedding(text) for text in texts]
        if not embeddings:
            self.logger.warning("No embeddings were created. No texts were added to the vector store.")
        embeddings = np.array(embeddings).astype("float32")
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.config['vector_store']['dimension'])
        self.index.add(embeddings)
        self.logger.debug(f"Added {embeddings.shape[0]} embeddings to the index.")


    def similarity_search(self, query: str, k: int = None) -> List[Tuple[str, float]]:
        """
        Perform a similarity search to find the top-k most similar texts to the query.
        
        Args:
            query (str): The query text to search for.
            k (int, optional): The number of top results to return. Uses config value if not provided.
        
        Returns:
            List[Tuple[str, float]]: List of tuples where each tuple contains a text and its similarity score.
        """
        if k is None:
            k = self.config['vector_store']['similarity_top_k']
        query_vector = self._get_embedding(query)
        query_vector = np.array(query_vector).astype('float32')
        try:
            distances, indices = self.index.search(np.array([query_vector]).astype('float32'), k)
            results = [(self.texts[i], float(distances[0][j])) for j, i in enumerate(indices[0])]
            self.logger.debug(f"Similarity search completed with {len(results)} results.")
            return results
        except Exception as e:
            self.logger.error(f"Error performing similarity search: {str(e)}", exc_info=True)
            raise