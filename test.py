# main.py
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import argparse
import re
import evaluate  

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
    device_map={"": 0},   # force all layers to GPU 0
    dtype=torch.float16,
)
model.eval()

# check: first parameter on cuda
first_param_device = next(model.parameters()).device
print("First param device:", first_param_device)

# --------- Generation ---------
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

# parse optional </think>
try:
    index = len(output_ids) - output_ids[::-1].index(151668)  # token id for </think>
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("\n--- Thinking content ---\n", thinking_content if thinking_content else "")
print("\n--- Content ---\n", content)

# --------- ROUGE evaluation ---------
def split_for_lsum(text: str) -> str:
    """
    ROUGE-Lsum expects sentence boundaries as newlines.
    This lightweight splitter uses punctuation to approximate sentences.
    """
    if not text.strip():
        return ""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    return "\n".join(sentences)

parser = argparse.ArgumentParser()
parser.add_argument("--reference", type=str, default=None, help="Reference summary text.")
parser.add_argument("--reference_file", type=str, default=None, help="Path to a file containing the reference summary.")
parser.add_argument("--use_lsum", action="store_true", help="Apply sentence splitting for ROUGE-Lsum-style scoring.")
args = parser.parse_args()

# Load reference text from CLI or file
reference_text = None
if args.reference_file:
    with open(args.reference_file, "r", encoding="utf-8") as f:
        reference_text = f.read()
elif args.reference:
    reference_text = args.reference

if reference_text is None:
    print("\n[ROUGE] No reference provided. Pass --reference '...' or --reference_file ref.txt")
else:
    # Prepare texts (optionally for Lsum)
    pred_proc = split_for_lsum(content) if args.use_lsum else content
    ref_proc  = split_for_lsum(reference_text) if args.use_lsum else reference_text

    rouge = evaluate.load("rouge")
    scores = rouge.compute(
        predictions=[pred_proc],
        references=[ref_proc],
        use_stemmer=True,
    )

    print("\n=== ROUGE (use_lsum={} ) ===".format(args.use_lsum))
    for k in ("rouge1", "rouge2", "rougeL", "rougeLsum"):
        if k in scores:
            print(f"{k}: {scores[k]:.4f}")
