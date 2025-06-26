# app/model_loader.py

import torch
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

MODEL_ERROR_BYPASS_FLAG = os.getenv("MODEL_ERROR_BYPASS_FLAG", "False").lower() == "true"

def load_cleaning_model(path: str):
    """
    Loads the CycleGAN model for signature cleaning.
    Returns the model or None if it fails and bypass is enabled.
    """
    print("Attempting to load signature cleaning model...")
    if not os.path.exists(path):
        print(f"ERROR: Cleaning model not found at {path}")
        if MODEL_ERROR_BYPASS_FLAG:
            print("MODEL_ERROR_BYPASS_FLAG is True. Proceeding with placeholder functionality.")
            return None
        raise FileNotFoundError(f"Cleaning model not found at {path}")
    
    try:
        # This is where you would load your actual PyTorch model
        # For demonstration, we'll just return a dummy object.
        # model = torch.load(path)
        # model.eval() 
        model = "DummyCleaningModel" # Replace with actual model loading
        print("Signature cleaning model loaded successfully.")
        return model
    except Exception as e:
        print(f"ERROR: Failed to load cleaning model from {path}. Error: {e}")
        if MODEL_ERROR_BYPASS_FLAG:
            print("MODEL_ERROR_BYPASS_FLAG is True. Proceeding with placeholder functionality.")
            return None
        raise e


def load_matching_model(path: str):
    """
    Loads the Siamese-Transformer model for signature matching.
    Returns the model or None if it fails and bypass is enabled.
    """
    print("Attempting to load signature matching model...")
    if not os.path.exists(path):
        print(f"ERROR: Matching model not found at {path}")
        if MODEL_ERROR_BYPASS_FLAG:
            print("MODEL_ERROR_BYPASS_FLAG is True. Proceeding with placeholder functionality.")
            return None
        raise FileNotFoundError(f"Matching model not found at {path}")
        
    try:
        # This is where you would load your actual PyTorch model
        # For demonstration, we'll just return a dummy object.
        # model = torch.load(path)
        # model.eval()
        model = "DummyMatchingModel" # Replace with actual model loading
        print("Signature matching model loaded successfully.")
        return model
    except Exception as e:
        print(f"ERROR: Failed to load matching model from {path}. Error: {e}")
        if MODEL_ERROR_BYPASS_FLAG:
            print("MODEL_ERROR_BYPASS_FLAG is True. Proceeding with placeholder functionality.")
            return None
        raise e

# --- Model Paths ---
CLEANING_MODEL_PATH = "models_weights/cyclegan_model.pth"
MATCHING_MODEL_PATH = "models_weights/siamese_transformer.pth"

# --- Load Models on Startup ---
# These will be imported into main.py
cleaning_model = load_cleaning_model(CLEANING_MODEL_PATH)
matching_model = load_matching_model(MATCHING_MODEL_PATH)

