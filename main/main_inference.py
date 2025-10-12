from src.device_manager import DeviceManager
from src.dataset import TireDataset
from src.model import TireClassificationModel
from src.inference import InferencePipeline


def main():
    print("-------------------------------------")
    device_manager = DeviceManager()
    device = device_manager.device

    # Load dataset and model
    dataset = TireDataset()
    model = TireClassificationModel(device=device)
    model.load()

    # Build inference pipeline
    inference_pipeline = InferencePipeline(model, dataset, device)
    app = inference_pipeline.create_gradio_app()

    # Launch the app
    app.launch(share=True)

    print("-------------------------------------")
    device_manager.release_memory()
    print("-------------------------------------")


if __name__ == "__main__":
    main()
