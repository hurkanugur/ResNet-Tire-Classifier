import torch
import config
import torch.nn.functional as F
from dataset import TireDataset
from PIL import Image
from device_manager import DeviceManager
from model import TireClassificationModel
import os

def main():
    # -------------------------
    # Select CUDA (GPU) / MPS (Mac) / CPU
    # -------------------------
    print("-------------------------------------")
    device_manager = DeviceManager()
    device = device_manager.device

    # -------------------------
    # Load dataset normalization params and categorical mappings
    # -------------------------
    dataset = TireDataset()

    # -------------------------
    # Load trained model
    # -------------------------
    model = TireClassificationModel(device=device)
    model.load()
    model.eval()

    # -------------------------
    # Perform inference on images (Defective and Good)
    # -------------------------
    for i in range(10):

        print("-------------------------------------")
        
        image_path = (
            f"{config.DATASET_PATH}/defective/Defective ({i + 1}).jpg"
            if i < 5
            else f"{config.DATASET_PATH}/good/good ({i + 1}).jpg"
        )

        if not os.path.exists(image_path):
            print(f"Image {image_path} not found!")
            continue
        
        # Prepare the image for inference
        image = Image.open(image_path)
        X = dataset.prepare_data_for_inference(image)

        # -------------------------
        # Model inference
        # -------------------------
        X = X.to(device)
        with torch.no_grad():
            outputs = model(X)
            probabilities = F.softmax(outputs, dim=1)
            probability, predicted_class_index = torch.max(probabilities, dim=1)

        # -------------------------
        # Display predictions
        # -------------------------
        print(f"• Image: {os.path.basename(image_path)}")
        print(f"• Predicted Class: {predicted_class_index.item()} (Probability: {probability.item():.4f})")

    # -------------------------
    # Release the memory
    # -------------------------
    print("-------------------------------------")
    device_manager.release_memory()
    print("-------------------------------------")

if __name__ == "__main__":
    main()
