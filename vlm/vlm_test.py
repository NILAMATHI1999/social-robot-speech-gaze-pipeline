import time
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"   # SmolVLM2 500M (Apache 2.0)
processor = AutoProcessor.from_pretrained(MODEL, do_image_splitting=False,          # no tiles: 1 image only (faster)
                                          size={"longest_edge": 512})               # smaller image = fewer tokens
model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.float32)   # CPU: float32

image = Image.open("../mvp/web/camera.jpg")              # last camera picture saved by gaze_trigger
messages = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text", "text": "Is the person holding a phone to their ear? Answer yes or no."}]}]
inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                       return_dict=True, return_tensors="pt")

t = time.perf_counter()
out = model.generate(**inputs, max_new_tokens=5, do_sample=False)
answer = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
print(f"answer: {answer.strip()}   ({time.perf_counter() - t:.2f} s)")
