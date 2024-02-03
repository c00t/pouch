from PIL import Image

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

from replace_hf import replace_linears_in_hf
device = torch.device("cpu")

model_id = "llava-hf/llava-1.5-7b-hf"
prompt = "USER: <image>\nCan you describe the main content, art style and predominant color palette of this image above? Finally, give some tags start with 'tags:'.\nASSISTANT:"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)

replace_linears_in_hf(model)

processor = AutoProcessor.from_pretrained(model_id)

processor = processor.to(device)

raw_image = Image.open("../profile.jpg")
inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)

output = model.generate(**inputs, max_new_tokens=200, do_sample=False)

print("Generated:", processor.decode(output[0][2:], skip_special_tokens=True))