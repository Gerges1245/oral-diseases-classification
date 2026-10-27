import json
import numpy as np
import tensorflow as tf
import gradio as gr

IMG_SIZE = 224
model = tf.keras.models.load_model("best_model_small.keras", compile=False)
class_names = json.load(open("class_names.json"))

def predict(image):
    if image is None:
        return None
    # نفس الـ preprocessing بتاع التدريب: resize فقط، والـ pixels من 0 لـ 255
    x = tf.image.resize(image.astype("float32"), (IMG_SIZE, IMG_SIZE))
    x = tf.expand_dims(x, 0)
    probs = model.predict(x, verbose=0)[0]
    return {c: float(p) for c, p in zip(class_names, probs)}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload an oral image"),
    outputs=gr.Label(num_top_classes=3, label="Prediction"),
    title="Oral Diseases Image Classification",
    description=(
        "Classifies oral images into: " + ", ".join(class_names) + ". "
        "Educational demo only, not a substitute for a professional diagnosis."
    ),
)

if __name__ == "__main__":
    demo.launch()