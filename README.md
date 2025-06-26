# Signature Verification Backend

This project provides a backend service for a signature verification application. It uses FastAPI and is designed to serve two primary machine learning models: one for cleaning noisy signature images (CycleGAN) and another for verifying signatures (Siamese-Transformer).

The application is built with robustness in mind, featuring asynchronous processing, error handling, and a model bypass mode for frontend development and testing when models are unavailable.

## Features

- **Signature Cleaning**: An endpoint to clean noise from multiple signature images.
- **Signature Matching**: An endpoint to compare two signatures and determine if they match.
- **Asynchronous API**: Built with `async` and `await` for high performance.
- **Model Agnostic**: Can be run in a bypass mode without the actual model weights, returning mock data.
- **CORS Enabled**: Properly configured to allow requests from a frontend application.
- **Performance Timers**: Includes timers to measure model inference times.

## Setup and Installation

1.  **Clone the repository (or set up the folder structure as above).**

2.  **Navigate to the project directory:**

    ```bash
    cd signature_verification_backend
    ```

3.  **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

4.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

5.  **Configure Environment Variables:**
    Create a `.env` file in the project root. This file controls the model loading behavior.

    ```
    # Set to 'True' to bypass model loading errors and use placeholder functions.
    # Set to 'False' or remove to enforce strict model loading.
    MODEL_ERROR_BYPASS_FLAG=True
    ```

6.  **Place Model Weights:**
    Place your trained `.pth` model files into the `models_weights/` directory. Ensure the filenames match those in `app/model_loader.py` or update the paths accordingly.

## How to Run the Server

From the root directory (`signature_verification_backend/`), run the following command:

```bash
uvicorn app.main:app --reload
```

- app.main:app: This tells uvicorn to look for the app object inside the app/main.py file.

- --reload: This enables auto-reloading, so the server will restart whenever you make changes to the code.

The application will be available at http://127.0.0.1:8000.

You can access the auto-generated API documentation at http://127.0.0.1:8000/docs.

## API Documentation

### POST /clean

Cleans a batch of signature images.

- Description: Takes a list of images, processes each one to remove noise, and returns the cleaned versions.

- Payload:

```json
{
  "images": [
    {
      "id": "unique-id-for-image-1",
      "data": "data:image/png;base64,iVBORw0KGgo..."
    },
    {
      "id": "unique-id-for-image-2",
      "data": "data:image/png;base64,iVBORw0KGgo..."
    }
  ]
}
```

Success Response (200 OK):

```json
{
  "cleaned_images": [
    {
      "id": "unique-id-for-image-1",
      "data": "data:image/png;base64,..."
    },
    {
      "id": "unique-id-for-image-2",
      "data": "data:image/png;base64,..."
    }
  ],
  "processing_times": {
    "unique-id-for-image-1": 0.5012,
    "unique-id-for-image-2": 0.5005,
    "total": 1.002
  }
}
```

- Error Response (500 Internal Server Error): If any image fails to process.

### POST /match

Compares two signatures to determine if they match.

- Description: Takes two images, computes a similarity score, and returns a match decision.

- Payload:

```json
{
  "image1": {
    "id": "image-id-1",
    "data": "data:image/png;base64,..."
  },
  "image2": {
    "id": "image-id-2",
    "data": "data:image/png;base64,..."
  }
}
```

Success Response (200 OK):

```json
{
  "match": "no match",
  "similarity_score": 0.5100895393534999,
  "processing_time": 0.2015
}
```

- Error Response (500 Internal Server Error): If the matching process fails.
