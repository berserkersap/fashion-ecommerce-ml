"""
Module for handling embeddings generation and manipulation using Fashion-CLIP model.
"""

from typing import List, Tuple
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from structlog import get_logger

from ..logging_config import error_log, app_logger

# Initialize Fashion-CLIP model and processor
try:
    model_name = "patrickjohncyh/fashion-clip"
    fashion_processor = CLIPProcessor.from_pretrained(model_name)
    fashion_model = CLIPModel.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
    fashion_model.eval()  # Set to evaluation mode
    app_logger.info(f"Successfully initialized Fashion-CLIP model on {'CUDA' if torch.cuda.is_available() else 'CPU'}")
except Exception as e:
    error_log(e, {"context": "Fashion-CLIP initialization"})
    raise

class EmbeddingGenerator:
    """
    Class for generating and manipulating embeddings using Fashion-CLIP model.
    
    This class provides methods to:
    - Generate embeddings from images
    - Generate embeddings from text
    - Combine image and text embeddings with weights
    """
    
    def __init__(self, image_weight: float = 0.7, text_weight: float = 0.3):
        """
        Initialize the embedding generator with default weights.
        
        Args:
            image_weight (float): Weight for image embeddings (default: 0.7)
            text_weight (float): Weight for text embeddings (default: 0.3)
        """
        self.image_weight = image_weight
        self.text_weight = text_weight
    
    @staticmethod
    def get_image_embedding(image: Image.Image) -> List[float]:
        """
        Generate embeddings for an image using Fashion-CLIP.
        
        Args:
            image (PIL.Image.Image): Input image
            
        Returns:
            List[float]: Normalized image embedding
            
        Raises:
            Exception: If embedding generation fails
        """
        try:
            with torch.no_grad():
                inputs = fashion_processor(images=image, return_tensors="pt")
                image_features = fashion_model.get_image_features(**{k: v.to(fashion_model.device) for k, v in inputs.items()})
                # Normalize embeddings
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                return image_features.cpu().numpy().tolist()[0]
        except Exception as e:
            error_log(e, {"context": "image_embedding_generation"})
            raise
    
    @staticmethod
    def get_text_embedding(text: str) -> List[float]:
        """
        Generate embeddings for text using Fashion-CLIP.
        
        Args:
            text (str): Input text
            
        Returns:
            List[float]: Normalized text embedding
            
        Raises:
            Exception: If embedding generation fails
        """
        try:
            with torch.no_grad():
                inputs = fashion_processor(text=text, return_tensors="pt", padding=True)
                text_features = fashion_model.get_text_features(**{k: v.to(fashion_model.device) for k, v in inputs.items()})
                # Normalize embeddings
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                return text_features.cpu().numpy().tolist()[0]
        except Exception as e:
            error_log(e, {
                "context": "text_embedding_generation",
                "text": text
            })
            raise
    
    def combine_embeddings(
        self,
        image_embedding: List[float],
        text_embedding: List[float],
        weights: Tuple[float, float] = None
    ) -> List[float]:
        """
        Combine image and text embeddings with specified weights.
        
        Args:
            image_embedding (List[float]): Image embedding
            text_embedding (List[float]): Text embedding
            weights (Tuple[float, float], optional): Custom weights for (image, text).
                                                   If None, uses instance weights.
        
        Returns:
            List[float]: Combined and normalized embedding
            
        Raises:
            ValueError: If weights don't sum to 1.0
            Exception: If combination fails
        """
        try:
            if weights is None:
                weights = (self.image_weight, self.text_weight)
            
            image_weight, text_weight = weights
            if abs((image_weight + text_weight) - 1.0) > 1e-6:
                raise ValueError("Weights must sum to 1.0")
            
            # Convert to numpy for efficient computation
            img_emb = np.array(image_embedding)
            txt_emb = np.array(text_embedding)
            
            # Weighted combination
            combined = (image_weight * img_emb) + (text_weight * txt_emb)
            
            # Normalize the combined embedding
            combined_norm = combined / np.linalg.norm(combined)
            
            return combined_norm.tolist()
        except Exception as e:
            error_log(e, {
                "context": "embedding_combination",
                "weights": weights
            })
            raise 