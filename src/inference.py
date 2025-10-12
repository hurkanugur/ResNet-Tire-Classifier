import torch
import torch.nn.functional as F
from PIL import Image
import gradio as gr

from src.dataset import TireDataset
from src.model import TireClassificationModel


class InferencePipeline:
    """
    Handles model loading, image preprocessing, and tire condition classification.
    """

    # ----------------- Initialization -----------------

    def __init__(
            self, 
            model: TireClassificationModel, 
            dataset: TireDataset, 
            device: torch.device
        ):
        
        self.model = model
        self.dataset = dataset
        self.device = device
        self.model.eval()

    # ----------------- Public Methods -----------------

    def predict(self, image_pixels) -> str:
        """
        Perform tire condition classification on an uploaded image.
        Returns formatted prediction text.
        """
        if image_pixels is None:
            return "⚠️ Please upload an image first!"

        # Convert image to PIL and preprocess
        image = Image.fromarray(image_pixels)
        X = self.dataset.prepare_data_for_inference(image).to(self.device)

        # Model inference
        with torch.no_grad():
            outputs = self.model(X)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_index = torch.max(probabilities, dim=1)
            confidence = round(confidence.item() * 100, 2)

        # Generate label
        if predicted_index == 0:
            return f"❌ Defective Tire — {confidence}% confidence"
        elif predicted_index == 1:
            return f"✅ Good Tire — {confidence}% confidence"
        else:
            return "⚠️ Unknown classification!"

    def create_gradio_app(self) -> gr.Blocks:
        """
        Build and return the Gradio interface for interactive tire classification.
        """
        with gr.Blocks(theme=gr.themes.Ocean(), title="Tire Condition Classifier") as demo:
            gr.Markdown(
                """
                # 🧠 Tire Condition Classifier  
                Upload a tire image and let the AI determine whether it’s **Defective** or **Good**.
                """
            )

            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(label="📸 Upload Tire Image", image_mode="RGB")

                with gr.Column(scale=1):
                    gr.Markdown("### 🧾 Prediction Result")
                    output_text = gr.Textbox(
                        label="Classification",
                        placeholder="The model's prediction will appear here...",
                        interactive=False,
                        lines=2,
                        show_copy_button=True,
                    )
                    analyze_btn = gr.Button("🔍 Analyze Tire", variant="primary")
                    clear_btn = gr.Button("🧹 Clear", variant="secondary")

            # Button actions
            analyze_btn.click(
                fn=self.predict,
                inputs=image_input,
                outputs=output_text,
            )

            clear_btn.click(
                fn=lambda: (None, ""),
                inputs=None,
                outputs=[image_input, output_text],
            )

            gr.Markdown(
                """
                ---
                💡 **Tip:** Use clear, high-quality images for best accuracy.  
                📊 Model trained to distinguish between *Defective* and *Good* tire conditions.  

                ---
                👨‍💻 **Developed by [Hürkan Uğur](https://github.com/hurkanugur)**  
                🔗 Source Code: [ResNet-Tire-Classifier](https://github.com/hurkanugur/ResNet-Tire-Classifier)
                """
            )

        return demo
