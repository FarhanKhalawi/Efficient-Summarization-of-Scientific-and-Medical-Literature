# main.py
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("No CUDA GPU detected.")

print("GPU Name:", torch.cuda.get_device_name(0))
print("CUDA Version (PyTorch):", torch.version.cuda)

model_name = "Qwen/Qwen3-30B-A3B-Thinking-2507" 

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, 
)

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("\nLoading model on GPU ...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map={"": 0},   
    dtype=torch.float16,  
)
model.eval()

# check: first parameter on cuda
first_param_device = next(model.parameters()).device
print("First param device:", first_param_device)

prompt = "Give me a short introduction to large language models."
messages = [{"role": "user", "content": prompt}]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

print("\nGenerating on GPU...")
with torch.inference_mode():
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
    )

output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

try:
    index = len(output_ids) - output_ids[::-1].index(151668) 
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("\n--- Thinking content ---\n", thinking_content)
print("\n--- Content ---\n", content)
