import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_path = "toosharm/spoiler-detector"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

def predict(review_text):
    inputs = tokenizer(review_text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0]
    return {"Not Spoiler": float(probs[0]), "Spoiler": float(probs[1])}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=5, placeholder="Paste a movie review here..."),
    outputs=gr.Label(num_top_classes=2),
    title="Movie Review Spoiler Detector",
    description="Paste a movie review to check if it likely contains spoilers."
)

demo.launch()