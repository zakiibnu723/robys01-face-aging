import os
from cog import BasePredictor, Input, Path
import torch
from PIL import Image
from huggingface_hub import hf_hub_download

# Import from the downloaded scripts
from models import UNet
from test_functions import process_image

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Downloading and loading model weights...")
        self.model_path = hf_hub_download(
            repo_id="Robys01/face-aging",
            filename="best_unet_model.pth"
        )
        self.model = UNet()
        self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"), weights_only=False))
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            
        self.model.eval()

    def predict(
        self,
        image: Path = Input(description="Input image of a face"),
        source_age: int = Input(description="Current age of the person", default=20),
        target_age: int = Input(description="Target age of the person", default=60),
    ) -> Path:
        """Run a single prediction on the model"""
        print(f"Processing image for source age {source_age} to target age {target_age}")
        
        # Load image
        img = Image.open(str(image))
        if img.mode not in ["RGB", "L"]:
            img = img.convert("RGB")
            
        # Run inference
        with torch.no_grad():
            processed_image = process_image(self.model, img, source_age, target_age)
        
        # Save output
        output_path = "/tmp/output.jpg"
        processed_image.save(output_path)
        
        return Path(output_path)
