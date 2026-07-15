"""Gradio web app."""

import gradio as gr

import config
from predict import AnimalPredictor


predictor = AnimalPredictor()


def predict_image(image, top_k: int = 3):
    if image is None:
        return None, "Please upload an image."

    results, is_out_of_scope = predictor.predict_pil(image, top_k=top_k)

    if is_out_of_scope:
        result_text = f"## {config.REJECTION_MESSAGE}\n\n"
        result_text += "这张图片和当前训练集里的品种不匹配，所以不做强行分类。"
        return image, result_text

    result_text = "## Prediction Results\n\n"
    result_text += "| Rank | Class | Confidence |\n"
    result_text += "|------|-------|-------------|\n"
    for i, (class_name, prob) in enumerate(results, 1):
        bar_len = int(prob * 20)
        progress_bar = "#" * bar_len + "." * (20 - bar_len)
        result_text += f"| {i} | **{class_name}** | {prob:.2%} {progress_bar} |\n"

    return image, result_text


with gr.Blocks(
    title="Animal Recognition",
    theme=gr.themes.Soft(),
    css="footer {visibility: hidden}",
) as demo:
    gr.Markdown("# Animal Species Classifier")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="Upload image", type="pil", height=400)
            with gr.Row():
                top_k_slider = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="Top-k")
                submit_btn = gr.Button("Classify", variant="primary", size="lg")

        with gr.Column(scale=1):
            image_output = gr.Image(label="Preview", type="pil", height=400)
            result_output = gr.Markdown(value="Upload an image and click Classify.")

    submit_btn.click(
        fn=predict_image,
        inputs=[image_input, top_k_slider],
        outputs=[image_output, result_output],
    )


if __name__ == "__main__":
    print("Starting animal recognition web app...")
    print(f"Device: {config.DEVICE}")
    print(f"Model path: {config.BEST_MODEL_PATH}")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
