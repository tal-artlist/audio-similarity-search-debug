import numpy as np
import os
import warnings
import logging
import gc
import contextlib

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings('ignore')

@contextlib.contextmanager
def suppress_essentia_warnings():
    """Context manager to suppress Essentia C++ warnings by redirecting stderr to null."""
    # Save the original stderr file descriptor
    original_stderr_fd = os.dup(2)
    
    try:
        # Open null device
        null_fd = os.open(os.devnull, os.O_WRONLY)
        
        # Redirect stderr to null device
        os.dup2(null_fd, 2)
        os.close(null_fd)
        
        yield
        
    finally:
        # Restore original stderr
        os.dup2(original_stderr_fd, 2)
        os.close(original_stderr_fd)

# Configure TensorFlow with graceful GPU fallback
def configure_tensorflow():
    """Configure TensorFlow with graceful GPU fallback."""
    try:
        import tensorflow as tf
        
        # Try to configure GPU with memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Enable memory growth for all GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logging.info(f"TensorFlow configured with {len(gpus)} GPU(s)")
                return True
            except RuntimeError as e:
                # GPU configuration failed, fall back to CPU
                logging.warning(f"GPU configuration failed, falling back to CPU: {e}")
                tf.config.set_visible_devices([], 'GPU')
                return False
        else:
            logging.info("No GPUs found, using CPU")
            return False
            
    except ImportError:
        return False
    except Exception as e:
        # Handle CUDA/cuDNN library errors gracefully
        if "libcudnn" in str(e) or "libnvrtc" in str(e) or "CUDA" in str(e):
            logging.warning(f"CUDA/cuDNN libraries not available, falling back to CPU: {e}")
            try:
                import tensorflow as tf
                tf.config.set_visible_devices([], 'GPU')
                return False
            except:
                pass
        else:
            logging.warning(f"TensorFlow configuration failed: {e}")
        return False

# Configure TensorFlow before importing Essentia
configure_tensorflow()

from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs
from config import settings

class EssentiaModel:
    def __init__(self):
        """Initialize the Essentia model with proper TensorFlow configuration."""
        self.model = None
        self.loader = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Essentia model with TensorFlow backend."""
        try:
            logging.info("🔧 Initializing EssentiaModel with TensorFlow backend...")
            
            # Initialize the TensorFlow model for Essentia
            self.model = TensorflowPredictEffnetDiscogs(
                graphFilename=settings.ESSENTIA_MODEL_PATH,
                output="PartitionedCall:1"
            )
            self.loader = MonoLoader()
            logging.info("✅ EssentiaModel initialized successfully")
            
        except Exception as e:
            # If there's still a CUDA error, try to force CPU and retry
            if "libcudnn" in str(e) or "libnvrtc" in str(e) or "CUDA" in str(e):
                logging.warning(f"CUDA error during model initialization, forcing CPU: {e}")
                try:
                    import tensorflow as tf
                    tf.config.set_visible_devices([], 'GPU')
                    os.environ["CUDA_VISIBLE_DEVICES"] = ""
                    
                    self.model = TensorflowPredictEffnetDiscogs(
                        graphFilename=settings.ESSENTIA_MODEL_PATH,
                        output="PartitionedCall:1"
                    )
                    self.loader = MonoLoader()
                    logging.info("✅ EssentiaModel initialized successfully on CPU after GPU fallback")
                except Exception as retry_error:
                    logging.error(f"❌ Failed to initialize EssentiaModel even on CPU: {retry_error}")
                    raise
            else:
                logging.error(f"❌ Failed to initialize EssentiaModel: {e}")
                raise

    def normalize(self, v):
        norm = np.linalg.norm(v)
        
        if norm == 0: 
            return v
        return v / norm

    def get_embeddings(self, audio_path: str) -> np.ndarray:
        with suppress_essentia_warnings():
            self.loader.configure(
                filename=audio_path,
                sampleRate=16000,
                resampleQuality=4
            )
            audio = self.loader()
            embeddings = self.model(audio)
            mean_embedding = np.mean(embeddings, axis=0)
            normalized_embeddings = self.normalize(mean_embedding)
            
            # Explicitly delete large audio array and force garbage collection
            del audio
            del embeddings
            gc.collect()
            
            return normalized_embeddings
