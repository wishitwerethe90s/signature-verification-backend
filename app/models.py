# app/models.py

from pydantic import BaseModel
from typing import List, Optional

class ImagePayload(BaseModel):
    """
    Represents a single image with an ID and base64 encoded data.
    """
    id: str
    data: str

class CleanRequest(BaseModel):
    """
    Request model for the /clean endpoint.
    Expects a list of ImagePayload objects.
    """
    images: List[ImagePayload]

class CleanResponse(BaseModel):
    """
    Response model for the /clean endpoint.
    Returns a list of cleaned images.
    """
    cleaned_images: List[ImagePayload]
    processing_times: dict

class MatchRequest(BaseModel):
    """
    Request model for the /match endpoint.
    Expects two images to be compared.
    """
    image1: ImagePayload
    image2: ImagePayload

class MatchResponse(BaseModel):
    """
    Response model for the /match endpoint.
    Returns the match result and similarity score.
    """
    match: str
    similarity_score: float
    processing_time: float
