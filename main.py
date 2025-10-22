# main.py — dataset-ready generation & classification (Qwen 4-bit)
# Usage examples:
#   python main.py --data_csv data/MeDAL/pretrain_subset/test.csv --task generate --save_csv outputs/generate_preds.csv --reference_col REFERENCE --limit_rows 5 --use_lsum
#   python main.py --data_csv data/MeDAL/pretrain_subset/test.csv --task classify --save_csv outputs/classify_preds.csv --limit_rows 5 --eval_accuracy

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import re
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import evaluate  # 🤗 Evaluate metrics (accuracy, rouge)

# ------------------------------
# CLI
# ------------------------------
parser = argparse.ArgumentParser(description="Run Qwen on a CSV for generation or classification.")
parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-30B-A3B-Thinking-2507",
                    help="HF model ID to load.")
parser.add_argument("--data_csv", type=str, default="data/MeDAL/pretrain_subset/test.csv",
                    help="Path to input CSV.")
parser.add_argument("--id_col", type=str, default="ABSTRACT_ID", help="ID column name in CSV.")
parser.add_argument("--text_col", type=str, default="TEXT", help="Text/input column name in CSV.")
parser.add_argument("--label_col", type=str, default="LABEL", help="Label column name for classification.")
parser.add_argument("--reference_col", type=str, default=None,
                    help="Reference text column for ROUGE evaluation (generation mode).")
parser.add_argument("--task", type=str, choices=["generate", "classify"], default="generate",
                    help="Run free-form generation or single-label classification.")
parser.add_argument("--save_csv", type=str, default="outputs/preds.csv", help="Where to save outputs CSV.")

# Generation params
parser.add_argument("--max_new_tokens", type=int, default=192)
parser.add_argument("--temperature", type=float, default=0.2)
parser.add_argument("--top_p", type=float, default=0.95)
parser.add_argument("--seed", type=int, default=42)

# Evaluation toggles (accuracy still opt-in; ROUGE now auto if reference_col provided)
parser.add_argument("--eval_accuracy", action="store_true", help="Compute accuracy in classify mode if labels present.")
parser.add_argument("--use_lsum", action="store_true", help="Apply sentence splitting for ROUGE-Lsum style scoring.")

# Optional memory cap per GPU; leave empty to let Accelerate decide.
parser.add_argument("--gpu_mem_gib", type=int, default=None,
                    help="If set, cap each GPU memory to this many GiB via max_memory.")

# Limit rows — default 5 for fast runs
parser.add_argument("--limit_rows", type=int, default=5,
                    help="Only process the first N rows of the CSV (default: 5).")

args = parser.parse_args()

# ------------------------------
# CUDA checks
# ------------------------------
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("No CUDA GPU detected.")

print("GPU Name:", torch.cuda.get_device_name(0))
print(f"CUDA Version (PyTorch): {torch.version.cuda}")

torch.manual_seed(args.seed)

# ------------------------------
# Model & tokenizer
# ------------------------------
print("\nLoading tokenizer…")
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Build an Accelerate-compatible max_memory mapping (integer GPU ids).
max_memory = None
if args.gpu_mem_gib is not None:
    max_memory = {i: f"{args.gpu_mem_gib}GiB" for i in range(torch.cuda.device_count())}
    max_memory["cpu"] = "48GiB"

print("\nLoading model on GPU …")
model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    quantization_config=bnb_config,
    device_map="auto",
    max_memory=max_memory,           # None -> let Accelerate choose
    dtype=torch.float16,             # modern arg (replaces deprecated torch_dtype)
    low_cpu_mem_usage=True,
    attn_implementation="eager",     # V100-safe (no FlashAttention on SM70)
)
model.eval()

first_param_device = next(model.parameters()).device
print("First param device:", first_param_device)

# ------------------------------
# Helpers
# ------------------------------
END_THINK_TOKEN_ID = 151668  # </think> for Qwen-Thinking models

def parse_thinking_and_content(output_ids: list[int]):
    """Split optional <think>...</think> from visible content."""
    try:
        index = len(output_ids) - output_ids[::-1].index(END_THINK_TOKEN_ID)
    except ValueError:
        index = 0
    thinking = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    return thinking, content

_reasoning_patterns = [
    r"(?is)^\s*(?:hmm[,.\s]|thought|let me|the user wants me to|i need to|analysis|reasoning).*?$",
]

def strip_reasoning(text: str) -> str:
    """Remove likely 'thinking' preambles if the model prints them as plain text."""
    cleaned = text.strip()
    # Drop leading meta-thought lines
    lines = [ln for ln in cleaned.splitlines() if ln.strip()]
    kept = []
    for ln in lines:
        if any(re.match(pat, ln) for pat in _reasoning_patterns):
            continue
        kept.append(ln)
    cleaned = " ".join(kept).strip()
    # Collapse spaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def apply_chat(prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def generate_once(prompt: str) -> str:
    text = apply_chat(prompt)
    model_inputs = tokenizer([text], return_tensors="pt")
    model_inputs = {k: v.to(first_param_device) for k, v in model_inputs.items()}
    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    output_ids = generated_ids[0][len(model_inputs["input_ids"][0]):].tolist()
    _thinking, content = parse_thinking_and_content(output_ids)
    content = strip_reasoning(content)
    # keep only first two sentences for summaries
    if args.task == "generate":
        sents = re.split(r'(?<=[.!?])\s+', content)
        content = " ".join(sents[:2]).strip()
    return content

def split_for_lsum(text: str) -> str:
    """ROUGE-Lsum expects sentence boundaries as newlines."""
    if not text or not str(text).strip():
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", str(text).strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    return "\n".join(sentences)

# ------------------------------
# Load data
# ------------------------------
print("\nReading:", args.data_csv)
df = pd.read_csv(args.data_csv)

if args.text_col not in df.columns:
    raise SystemExit(f"Missing text column: {args.text_col}")

if args.limit_rows:
    df = df.head(args.limit_rows)
    print(f"Processing only the first {len(df)} rows")

# ------------------------------
# Build label set if classify
# ------------------------------
label_set = None
if args.task == "classify":
    if args.label_col not in df.columns:
        raise SystemExit(f"Classification requested but label column not found: {args.label_col}")
    label_set = sorted({str(x).strip() for x in df[args.label_col].dropna().unique()})
    label_list_display = ", ".join(label_set)
    print("Label set:", label_set)

# ------------------------------
# Run
# ------------------------------
outputs = []
for i, row in df.iterrows():
    sample_id = row.get(args.id_col, i)
    text = str(row[args.text_col])

    if args.task == "classify":
        prompt = (
            "You are a strict classifier. "
            f"Choose exactly ONE label from this set: [{label_list_display}].\n\n"
            "Rules:\n"
            "1) Output only the label text—no extra words.\n"
            "2) If unsure, pick the closest label.\n\n"
            f"Text:\n{text}\n\n"
            "Answer with exactly one label from the set:"
        )
        pred = generate_once(prompt).splitlines()[0].strip()
    else:
        prompt = (
            "Summarize the following abstract in 1–2 sentences, focusing on the main findings and outcome measures:\n\n"
            f"{text}"
        )
        pred = generate_once(prompt)

    outputs.append({
        args.id_col: sample_id,
        args.text_col: text,
        (args.label_col if args.label_col in df.columns else "LABEL"): row.get(args.label_col, None),
        "prediction": pred,
    })

# ------------------------------
# Save CSV
# ------------------------------
out_df = pd.DataFrame(outputs)
os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)
out_df.to_csv(args.save_csv, index=False)
print(f"\nSaved predictions -> {args.save_csv}")

# ------------------------------
# Evaluation
# ------------------------------
# 1) Accuracy (opt-in)
if args.task == "classify" and args.eval_accuracy and args.label_col in df.columns:
    try:
        acc = evaluate.load("accuracy")
        preds = [str(x).strip() for x in out_df["prediction"]]
        refs = [str(x).strip() for x in out_df[args.label_col]]
        scores = acc.compute(predictions=preds, references=refs)
        print(f"\n=== Classification accuracy ===\naccuracy: {scores['accuracy']:.4f}")
    except Exception as e:
        print("[warn] accuracy metric not computed:", e)

# 2) ROUGE (AUTO if reference_col provided & exists)
if args.task == "generate":
    if args.reference_col is None:
        print("[ROUGE] reference_col not set — skipping ROUGE.")
    elif args.reference_col not in df.columns:
        print(f"[ROUGE] Reference column '{args.reference_col}' not found in CSV — skipping.")
    else:
        try:
            rouge = evaluate.load("rouge")
            preds_proc = [split_for_lsum(x) if args.use_lsum else x for x in out_df["prediction"]]
            refs_proc  = [split_for_lsum(x) if args.use_lsum else x for x in df[args.reference_col].iloc[:len(out_df)]]
            scores = rouge.compute(predictions=preds_proc, references=refs_proc, use_stemmer=True)
            print(f"\n=== ROUGE (auto, use_lsum={args.use_lsum}) ===")
            for k in ("rouge1", "rouge2", "rougeL", "rougeLsum"):
                if k in scores:
                    print(f"{k}: {scores[k]:.4f}")
            # Save metrics next to predictions
            metrics_path = os.path.splitext(args.save_csv)[0] + "_rouge.json"
            with open(metrics_path, "w") as f:
                json.dump(scores, f, indent=2)
            print(f"[ROUGE] Saved metrics -> {metrics_path}")
        except Exception as e:
            print("[warn] ROUGE not computed:", e)
