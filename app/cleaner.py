# app/cleaner.py

import torch
from PIL import Image
import torchvision.transforms as transforms
import os

# This import works if you copied the 'models' directory from the original repo
# into your 'app' directory as instructed.
from .model_files import networks

class SignatureCleaner:
    """A class to load the pix2pix model and clean signature images."""

    def __init__(self, checkpoint_path: str, gpu_id: int = -1):
        """
        Initializes the SignatureCleaner.
        Args:
            checkpoint_path (str): Path to the trained generator .pth file.
            gpu_id (int): GPU to use. -1 for CPU.
        """
        # --- Device Setup ---
        if gpu_id > -1 and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{gpu_id}')
            print(f"Using GPU: {gpu_id}")
        else:
            self.device = torch.device('cpu')
            print("Using CPU.")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

        # --- Model Definition ---
        # These parameters must match the model you trained.
        # For the default pix2pix model (unet_256):
        # input_nc=3, output_nc=3, num_downs=8, ngf=64, norm_layer=batch, use_dropout=True
        self.model = networks.define_G(
            input_nc=3,
            output_nc=3,
            ngf=64,
            netG='unet_256',
            norm='batch',
            use_dropout=True,
            init_type='normal',
            init_gain=0.02,
            gpu_ids=[gpu_id] if gpu_id > -1 else []
        )
        
        # --- Load Weights ---
        print(f"Loading model from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()  # Set model to evaluation mode
        print("Model loaded successfully.")

        # --- Image Transformations ---
        # This transformation pipeline must match the one used during testing in the repo.
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize to [-1, 1]
        ])

    def _tensor_to_pil(self, tensor_image):
        """Converts an output tensor to a PIL Image."""
        # De-normalize from [-1, 1] to [0, 1]
        image_tensor = tensor_image.detach().cpu().float().squeeze(0)
        image_tensor = (image_tensor + 1) / 2.0
        image_tensor = image_tensor.clamp(0, 1)
        # Convert to PIL Image
        return transforms.ToPILImage()(image_tensor)

    def clean(self, input_image: Image.Image) -> Image.Image:
        """
        Cleans a single signature image.
        Args:
            input_image (PIL.Image): A PIL Image of the dirty signature.
        Returns:
            PIL.Image: A PIL Image of the cleaned signature.
        """
        # Ensure image is RGB and resized to 256x256
        input_image = input_image.convert("RGB").resize((256, 256))

        # Apply transformations and add batch dimension
        input_tensor = self.transform(input_image).unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            output_tensor = self.model(input_tensor)

        # Convert output tensor back to a PIL image
        cleaned_image = self._tensor_to_pil(output_tensor)
        
        return cleaned_image

# --- Example Usage (demonstrates how to use the class) ---
if __name__ == '__main__':
    # 1. Define paths
    MODEL_PATH = "checkpoints/latest_net_G.pth"
    INPUT_IMAGE_PATH = "path/to/your/dirty_signature.png"
    OUTPUT_IMAGE_PATH = "cleaned_signature.png"

    # Check if a dummy input file exists, if not, create one
    if not os.path.exists(INPUT_IMAGE_PATH):
        print(f"Creating a dummy input image at: {INPUT_IMAGE_PATH}")
        os.makedirs(os.path.dirname(INPUT_IMAGE_PATH), exist_ok=True)
        dummy_img = Image.new('RGB', (256, 256), color = 'red')
        dummy_img.save(INPUT_IMAGE_PATH)


    # 2. Initialize the cleaner
    # Use gpu_id=0 for GPU, or -1 for CPU
    cleaner = SignatureCleaner(checkpoint_path=MODEL_PATH, gpu_id=-1)

    # 3. Load the dirty image
    print(f"Loading image from {INPUT_IMAGE_PATH}")
    dirty_image = Image.open(INPUT_IMAGE_PATH)

    # 4. Clean the image
    print("Cleaning image...")
    cleaned_image = cleaner.clean(dirty_image)

    # 5. Save the result
    cleaned_image.save(OUTPUT_IMAGE_PATH)
    print(f"Successfully cleaned image and saved it to {OUTPUT_IMAGE_PATH}")