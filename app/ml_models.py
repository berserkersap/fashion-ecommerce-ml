"""
Singleton instances of ML models used across the application.
"""

from vllm import LLM, SamplingParams
from PIL import Image
import io
import base64
from typing import Optional, Union, Dict
import os
import torch
from transformers import AutoProcessor
from optimum.onnxruntime import ORTModelForVision2Seq
import onnxruntime as ort
from .logging_config import app_logger, error_log

class MLModels:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MLModels, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize ML models"""
        self._init_moondream()
        
    def _init_moondream(self):
        """Initialize Moondream model with ONNX Runtime optimization"""
        try:
            # Check if CUDA is available
            if torch.cuda.is_available():
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                app_logger.info("CUDA is available, using GPU acceleration")
            else:
                providers = ['CPUExecutionProvider']
                app_logger.info("CUDA not available, using CPU only")

            # Set up ONNX Runtime session options
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = os.cpu_count()  # Use all CPU cores
            sess_options.enable_mem_pattern = True
            sess_options.enable_cpu_mem_arena = True

            # Load optimized Moondream model from container path or download
            model_path = os.getenv('MOONDREAM_MODEL_PATH', '/app/models/moondream_onnx')
            processor_path = os.getenv('MOONDREAM_PROCESSOR_PATH', '/app/models/moondream')

            if not os.path.exists(model_path):
                app_logger.info("Downloading and optimizing Moondream model...")
                # Export model to ONNX format if not already available
                self.moondream_model = ORTModelForVision2Seq.from_pretrained(
                    "vikhyatk/moondream1",
                    export=True
                )
                # Save optimized model
                os.makedirs(model_path, exist_ok=True)
                self.moondream_model.save_pretrained(model_path)
            else:
                app_logger.info("Loading pre-optimized Moondream model...")
                self.moondream_model = ORTModelForVision2Seq.from_pretrained(
                    model_path,
                    session_options=sess_options,
                    providers=providers
                )

            # Initialize processor
            self.moondream_processor = AutoProcessor.from_pretrained(
                processor_path if os.path.exists(processor_path) else "vikhyatk/moondream1"
            )
            
            # Set generation parameters
            self.generation_config = {
                "max_length": 256,
                "num_beams": 3,
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.2,
                "length_penalty": 1.0
            }
            
            app_logger.info("Successfully initialized optimized Moondream model")
        except Exception as e:
            error_log(e, {"context": "Moondream ONNX initialization"})
            self.moondream_model = None
            self.moondream_processor = None
            raise
    
    def get_image_description(self, image: Image.Image) -> Optional[str]:
        """Get image description using optimized Moondream model"""
        if not self.moondream_model or not self.moondream_processor:
            error_log(Exception("Moondream model not initialized"), {"context": "image_description_generation"})
            return None
            
        try:
            # Prepare image
            inputs = self.moondream_processor(images=image, return_tensors="pt")
            
            # Move inputs to the same device as model if using CUDA
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate description with optimized model
            outputs = self.moondream_model.generate(
                **inputs,
                **self.generation_config
            )
            
            # Decode the output
            description = self.moondream_processor.batch_decode(
                outputs, 
                skip_special_tokens=True
            )[0].strip()
            
            return description
        except Exception as e:
            error_log(e, {"context": "image_description_generation"})
            return None

# Create singleton instance
ml_models = MLModels() 