import torch
import torch.nn.functional as F
from dataset import TireDataset
from PIL import Image
from device_manager import DeviceManager
from model import TireClassificationModel
import gradio as gr


# -------------------------
# Classification function
# -------------------------
def classify_image(model, dataset, device, image_pixels):

    if image_pixels is None:
        return "⚠️ Please select an image first!"

    image = Image.fromarray(image_pixels)
    X = dataset.prepare_data_for_inference(image)
    X = X.to(device)

    with torch.no_grad():
        outputs = model(X)
        probabilities = F.softmax(outputs, dim=1)
        probability, predicted_class_index = torch.max(probabilities, dim=1)
        probability = round(probability[0].item() * 100, 2)

    if predicted_class_index == 0:
        return f"❌ Defective Tire — {probability}% confidence"
    elif predicted_class_index == 1:
        return f"✅ Good Tire — {probability}% confidence"
    else:
        return "⚠️ Unknown classification!"


# -------------------------
# Gradio UI Builder
# -------------------------
def create_gradio_app(model, dataset, device):
    with gr.Blocks(theme=gr.themes.Ocean(), title="Tire Condition Classifier") as demo:
        gr.Markdown(
            """
            # 🧠 Tire Condition Classifier  
            Upload a tire image and let the AI determine whether it’s **Defective** or **Good**.
            """,
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
                submit_btn = gr.Button("🔍 Analyze Tire", variant="primary", scale=1)
                clear_btn = gr.Button("🧹 Clear", variant="secondary")

        submit_btn.click(
            fn=lambda image_pixels: classify_image(model, dataset, device, image_pixels),
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
            📊 Model is trained to distinguish between *Defective* and *Good* tire conditions.  

            ---
            👨‍💻 **Developed by [Hürkan Uğur](https://github.com/hurkanugur)**  
            🔗 Source Code: [ResNet-Tire-Classifier](https://github.com/hurkanugur/ResNet-Tire-Classifier)
            """
        )

    return demo


# -------------------------
# Main entry point
# -------------------------
def main():
    print("-------------------------------------")
    device_manager = DeviceManager()
    device = device_manager.device

    dataset = TireDataset()

    model = TireClassificationModel(device=device)
    model.load()
    model.eval()

    demo = create_gradio_app(model, dataset, device)
    demo.launch(share=True)

    print("-------------------------------------")
    device_manager.release_memory()
    print("-------------------------------------")


if __name__ == "__main__":
    main()
