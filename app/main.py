# app/main.py

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import asyncio

from .models import CleanRequest, CleanResponse, MatchRequest, MatchResponse, ImagePayload
from .utils import base64_to_image, image_to_base64, timer, placeholder_clean_image, placeholder_match_images
from .model_loader import cleaning_model, matching_model

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Signature Verification API",
    description="A backend service to clean and verify signatures using ML models.",
    version="1.0.0"
)

# --- CORS Middleware ---
# This allows the frontend (running on a different domain/port) to communicate with the backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Event Handlers ---
@app.on_event("startup")
async def startup_event():
    """
    Code to run on application startup.
    Models are loaded via import from model_loader.
    """
    print("Application startup complete.")
    if cleaning_model is None:
        print("WARNING: Running in cleaning model bypass mode.")
    if matching_model is None:
        print("WARNING: Running in matching model bypass mode.")

@app.on_event("shutdown")
async def shutdown_event():
    """
    Code to run on application shutdown.
    """
    print("Application shutting down.")


# --- API Endpoints ---
@app.get("/", summary="Root endpoint to check service status")
async def root():
    """
    A simple endpoint to check if the service is running.
    """
    return {"status": "Signature Verification API is running"}

@app.post("/clean", response_model=CleanResponse, summary="Clean a batch of signature images")
async def clean_signatures(request: CleanRequest):
    """
    Accepts a list of base64 encoded images, cleans them, and returns the results.
    - **Input**: A JSON object with a list of images, each with an ID and base64 data.
    - **Output**: A JSON object with a list of cleaned images and processing times.
    """
    processing_times = {}
    
    async def process_single_image(img_payload: ImagePayload):
        """Asynchronously processes one image."""
        try:
            with timer(f"Image ID {img_payload.id} cleaning", processing_times, img_payload.id):
                input_image = base64_to_image(img_payload.data)

                if cleaning_model is None:
                    # Bypass mode: return the original image
                    cleaned_image = placeholder_clean_image(input_image)
                else:
                    # Real model inference would go here
                    # Example:
                    # transformed_image = transform(input_image)
                    # cleaned_tensor = cleaning_model(transformed_image.unsqueeze(0))
                    # cleaned_image = to_pil_image(cleaned_tensor.squeeze(0))
                    # For now, we use the placeholder
                    cleaned_image = placeholder_clean_image(input_image)

                cleaned_base64 = image_to_base64(cleaned_image)
                return ImagePayload(id=img_payload.id, data=cleaned_base64)
                
        except Exception as e:
            # Log the error and decide how to handle it. 
            # Here, we'll raise an HTTPException to inform the client.
            print(f"Error processing image {img_payload.id}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to process image {img_payload.id}: {str(e)}")

    with timer("Total cleaning time for all images", processing_times, "total"):
        tasks = [process_single_image(img) for img in request.images]
        cleaned_image_payloads = await asyncio.gather(*tasks)

    return CleanResponse(cleaned_images=cleaned_image_payloads, processing_times=processing_times)


@app.post("/match", response_model=MatchResponse, summary="Match two signature images")
async def match_signatures(request: MatchRequest):
    """
    Accepts two base64 encoded images and returns a similarity score and match decision.
    - **Input**: A JSON object with two images (image1, image2).
    - **Output**: A JSON object with match status, similarity score, and processing time.
    """
    processing_times = {}
    try:
        with timer("Total matching time", processing_times, "total"):
            img1 = base64_to_image(request.image1.data)
            img2 = base64_to_image(request.image2.data)

            if matching_model is None:
                # Bypass mode: return random result
                match_status, score = placeholder_match_images(img1, img2)
            else:
                # Real model inference would go here
                # Example:
                # score = matching_model(transform(img1), transform(img2))
                # match_status = "match" if score > THRESHOLD else "no match"
                # For now, we use the placeholder
                match_status, score = placeholder_match_images(img1, img2)

        return MatchResponse(
            match=match_status, 
            similarity_score=score,
            processing_time=processing_times.get("total", 0.0)
        )

    except Exception as e:
        print(f"Error during matching: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to match signatures: {str(e)}")

