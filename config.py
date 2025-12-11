import torch
import os
from pathlib import Path

def get_device_map() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'

device = get_device_map()

class Settings:
    # Project root is the audio-similarity-search folder
    PROJECT_ROOT: Path = Path(__file__).parent
    
    # Model path
    ESSENTIA_MODEL_PATH: str = str(PROJECT_ROOT / "models" / "essentia_model" / "discogs_multi_embeddings-effnet-bs64-1.pb")
    
    # Device
    DEVICE: str = device

settings = Settings()
