import cv2
import llama
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

llama_dir = "/mnt/vepfs/zhangyichi/Trustworthy-MLLM/playground/model_weights/LLaMA-7B"

# choose from BIAS-7B, LORA-BIAS-7B, CAPTION-7B.pth
name = "/mnt/vepfs/zhangyichi/Trustworthy-MLLM/playground/model_weights/LLaMA-Adapter-V2/7fa55208379faf2dd862565284101b0e4a2a72114d6490a95e432cf9d9b6c813_BIAS-7B.pth"
# model, preprocess = llama.load("BIAS-7B", llama_dir, device=device)
model, preprocess = llama.load(name, llama_dir, device=device)
model.eval()

prompt = llama.format_prompt('Please introduce this painting.')
img = Image.fromarray(cv2.imread("/mnt/vepfs/zhangyichi/Trustworthy-MLLM/playground/LLaMA-Adapter/docs/logo_v1.png"))
img = preprocess(img).unsqueeze(0).to(device)

result = model.generate(img, [prompt])[0]

print(result)
