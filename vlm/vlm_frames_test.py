import time
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

import csv
import sys
FRAMES = sys.argv[1] if len(sys.argv) > 1 else "vlm_frames/"                                  # folder with frames + labels.csv
MODEL = sys.argv[2] if len(sys.argv) > 2 else "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"  # 500M or 2.2B (Apache 2.0)
processor = AutoProcessor.from_pretrained(MODEL, do_image_splitting=False,          # no tiles: 1 image only (faster)
                                         size={"longest_edge": 512})               # smaller image = fewer tokens
model = AutoModelForImageTextToText.from_pretrained(
    MODEL, dtype=torch.bfloat16 if "2.2B" in MODEL else torch.float32)   # 2.2B in float32 needs ~9 GB RAM: too much

QUESTIONS = {"phone_ear": "Is the person holding a phone to their ear? Answer yes or no.",
             "waving":    "Is the person waving a hand? Answer yes or no.",
             "pointing":  "Is the person pointing with a finger? Answer yes or no.",
             "holding":   "Is the person holding an object in their hand? Answer yes or no.",
             "facing":    "Is the person's face turned towards the camera? Answer yes or no."}
rows = list(csv.DictReader(open(FRAMES + "labels.csv")))
# for each row, for each question: Image.open(FRAMES + row["frame"]) -> ask (your lines 11-19) -> compare with row[key]
results = []
for row in rows:
    image = Image.open(FRAMES + row["frame"]).convert("RGB")
    for key, question in QUESTIONS.items():
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question}]}]
        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                               return_dict=True, return_tensors="pt")
        YES, NO = 10539, 5230                                    # token ids of "Yes" and "No" (same in 500M and 2.2B)
        t = time.perf_counter()
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1]               # scores for the first answer word
        p_yes = torch.softmax(logits[[YES, NO]].float(), dim=0)[0].item()   # how sure: "Yes" vs "No" (0..1)
        seconds = time.perf_counter() - t
        said = "yes" if p_yes >= 0.5 else "no"

        results.append([row["frame"], key, row[key], said, said == row[key], round(seconds, 2), round(p_yes, 3)])
        print(row["frame"], key, "expected", row[key], "| said", said, flush=True)

# score per question + save all answers (one file per frames folder and model)
for key in QUESTIONS:
    r = [x for x in results if x[1] == key]
    print(f"{key:10s} correct {sum(x[4] for x in r)}/{len(r)}")
out_name = "results_" + FRAMES.strip("/") + "_" + MODEL.split("/")[-1] + ".csv"
with open(out_name, "w", newline="") as f:
    csv.writer(f).writerows([["frame", "question", "expected", "said", "correct", "seconds", "p_yes"]] + results)
