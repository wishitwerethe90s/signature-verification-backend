# app/utils.py

import base64
import numpy as np
from PIL import Image
import io
import time
from contextlib import contextmanager

def base64_to_image(base64_string: str) -> Image.Image:
    """
    Decodes a base64 string into a PIL Image.
    Handles strings with or without the 'data:image/...' prefix.
    """
    # Remove the header if it exists
    if "," in base64_string:
        header, encoded_data = base64_string.split(",", 1)
    else:
        encoded_data = base64_string

    # Decode the base64 string
    image_data = base64.b64decode(encoded_data)
    
    # Open the image using Pillow
    image = Image.open(io.BytesIO(image_data))
    return image

def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """
    Encodes a PIL Image into a base64 string.
    """
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{img_str}"

@contextmanager
def timer(description: str, times_dict: dict, key: str):
    """
    A context manager to time a block of code and store the result in a dictionary.
    """
    start_time = time.time()
    yield
    end_time = time.time()
    elapsed_time = end_time - start_time
    times_dict[key] = elapsed_time
    print(f"{description} took: {elapsed_time:.4f} seconds")

# Example of a placeholder function for model inference
def placeholder_clean_image(image: Image.Image) -> Image.Image:
    """
    Placeholder function to "clean" an image.
    In bypass mode, this simply returns the original image.
    """
    # In a real scenario, you would apply your CycleGAN model here.
    # For example: cleaned_tensor = cyclegan_model(transform(image))
    # cleaned_image = to_pil_image(cleaned_tensor)
    time.sleep(0.5) # Simulate processing time
    return image

def placeholder_match_images(image1: Image.Image, image2: Image.Image) -> tuple[str, float]:
    """
    Placeholder function to "match" two images.
    In bypass mode, this generates a random result.
    """
    # In a real scenario, you would use your Siamese-Transformer model here.
    # For example: score = siamese_model(transform(image1), transform(image2))
    time.sleep(0.2) # Simulate processing time
    score = np.random.rand()
    match_status = "match" if score > 0.6 else "no match"
    return match_status, float(score)

